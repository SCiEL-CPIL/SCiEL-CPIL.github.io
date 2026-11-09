# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest
    triangle area to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2)
    """
    n = 13
    start_time = time.time()
    timeout = 600  # seconds, keep margin from 630
    
    # Precompute all triangle index combinations
    tri_indices = list(itertools.combinations(range(n), 3))
    tri_count = len(tri_indices)
    
    # Vectorized area computation for all triangles
    def compute_min_area(points):
        """Compute minimum triangle area using vectorized operations."""
        # For each triangle (i,j,k), area = 0.5*|x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        # = 0.5*|cross(p2-p1, p3-p1)|
        idx = np.array(tri_indices)
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Cross product in 2D: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return np.min(areas)
    
    def compute_hull_area(points):
        """Compute convex hull area."""
        try:
            hull = scipy.spatial.ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0
    
    def score(points):
        """Compute the metric: min_triangle_area / hull_area."""
        if not np.all(np.isfinite(points)):
            return -float('inf')
        min_area = compute_min_area(points)
        if min_area <= 0:
            return -float('inf')
        hull_area = compute_hull_area(points)
        if hull_area <= 0:
            return -float('inf')
        return min_area / hull_area
    
    # Objective for scipy: minimize negative score
    def objective(points_flat):
        points = points_flat.reshape(n, 2)
        s = score(points)
        if s == -float('inf'):
            return 1e6
        return -s
    
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = init_points.copy()
    best_score = score(best_points)
    
    # Simulated Annealing
    def simulated_annealing(initial_points, max_iters=5000, t0=0.1, t_end=0.001):
        """SA to explore the non-smooth landscape."""
        current = initial_points.copy()
        current_score = score(current)
        best_sa = current.copy()
        best_sa_score = current_score
        
        # Geometric temperature schedule
        for it in range(max_iters):
            if time.time() - start_time > timeout - 30:
                break
            
            # Propose a random perturbation
            step_size = t0 * (t_end / t0) ** (it / max_iters)
            perturbation = np.random.normal(0, step_size, (n, 2))
            proposed = current + perturbation
            
            proposed_score = score(proposed)
            
            if proposed_score == -float('inf'):
                # Reject invalid
                current = proposed  # Still move, but track best
                continue
            
            # Accept if better or with probability exp((new-old)/T)
            if proposed_score > current_score:
                current = proposed
                current_score = proposed_score
                if current_score > best_sa_score:
                    best_sa = current.copy()
                    best_sa_score = current_score
            else:
                T = t0 * (t_end / t0) ** (it / max_iters)
                if T > 1e-10:
                    delta = proposed_score - current_score
                    if np.random.random() < np.exp(delta / T):
                        current = proposed
                        current_score = proposed_score
        
        return best_sa, best_sa_score
    
    # Local refinement using Powell
    def local_refine(points, maxiter=200):
        """Refine points using Powell's method."""
        try:
            res = scipy.optimize.minimize(
                objective,
                points.flatten(),
                method='Powell',
                options={'maxiter': maxiter, 'ftol': 1e-10, 'xatol': 1e-10}
            )
            refined = res.x.reshape(n, 2)
            refined_score = score(refined)
            return refined, refined_score
        except Exception:
            return points, score(points)
    
    # Multi-start SA + local refinement
    rng = np.random.default_rng(42)
    
    # Start 1: Regular polygon
    sa_result, sa_score = simulated_annealing(init_points, max_iters=3000)
    if sa_score > best_score:
        best_points = sa_result.copy()
        best_score = sa_score
    
    # Local refine the best SA result
    refined, refined_score = local_refine(best_points, maxiter=300)
    if refined_score > best_score:
        best_points = refined.copy()
        best_score = refined_score
    
    # Start 2: Perturbed regular polygon
    if time.time() - start_time < timeout - 60:
        init2 = init_points + rng.normal(0, 0.1, (n, 2))
        sa_result2, sa_score2 = simulated_annealing(init2, max_iters=2000)
        if sa_score2 > best_score:
            best_points = sa_result2.copy()
            best_score = sa_score2
        
        refined2, refined_score2 = local_refine(best_points, maxiter=200)
        if refined_score2 > best_score:
            best_points = refined2.copy()
            best_score = refined_score2
    
    # Start 3: Different initial configuration (slightly irregular)
    if time.time() - start_time < timeout - 60:
        angles3 = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.normal(0, 0.2, n)
        radii3 = 1.0 + rng.normal(0, 0.1, n)
        init3 = np.column_stack([np.cos(angles3) * radii3, np.sin(angles3) * radii3])
        sa_result3, sa_score3 = simulated_annealing(init3, max_iters=2000)
        if sa_score3 > best_score:
            best_points = sa_result3.copy()
            best_score = sa_score3
        
        refined3, refined_score3 = local_refine(best_points, maxiter=200)
        if refined_score3 > best_score:
            best_points = refined3.copy()
            best_score = refined_score3
    
    # Final local refinement
    if time.time() - start_time < timeout - 10:
        final_refined, final_score = local_refine(best_points, maxiter=500)
        if final_score > best_score:
            best_points = final_refined.copy()
            best_score = final_score
    
    return best_points

# EVOLVE-BLOCK-END
