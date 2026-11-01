"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts
    
    def get_covering_info(pts):
        """
        Computes the convex hull, finds the facet with the largest circumradius,
        and returns (radius_in_radians, worst_circumcenter, worst_simplex_indices).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Fallback: return a large radius and a random point
            return np.pi, np.random.randn(3) / np.linalg.norm(np.random.randn(3)), None

        max_radius = 0.0
        worst_center = None
        worst_simplex = None
        
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            c = n / norm_n
            # Ensure c is on the correct side (pointing outwards)
            if np.dot(c, v1) < 0:
                c = -c
                
            # Calculate angular distance from c to v1
            # cos(angle) = c.dot(v1)
            # angle = arccos(c.dot(v1))
            # Alternatively: d = ||c - v1||, angle = 2*arcsin(d/2)
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            
            if radius > max_radius:
                max_radius = radius
                worst_center = c
                worst_simplex = simplex.copy()
                
        if worst_center is None:
            return np.pi, np.random.randn(3) / np.linalg.norm(np.random.randn(3)), None
            
        return max_radius, worst_center, worst_simplex

    def calc_covering_radius(pts):
        radius, _, _ = get_covering_info(pts)
        return radius

    # 1. Initialization: Fibonacci spiral
    current_points = fibonacci_sphere(N).copy()
    best_points = current_points.copy()
    best_radius = calc_covering_radius(current_points)
    
    # 2. Main Optimization Loop: Gradient Descent on Worst Facet Vertices
    #    + Periodic Global Perturbation / Restart
    
    # Strategy:
    # - Compute covering radius and worst facet.
    # - Move vertices of the worst facet towards the circumcenter.
    # - This reduces the local maximum.
    # - Occasionally, if stuck or to escape local minima, add noise or restart from a different initial config.
    
    # We will implement a multi-stage approach:
    # Stage 1: Aggressive local search with decay.
    # Stage 2: Fine-tuning with smaller steps.
    # Stage 3: Random perturbations around the best solution found.
    
    # Hyperparameters
    base_step = 0.05
    decay_rate = 0.995
    min_step = 0.0001
    max_iter = 20000
    
    current_step = base_step
    stuck_count = 0
    threshold = 1e-6
    
    for iter in range(max_iter):
        if time.time() > deadline - 15:
            break
            
        radius, worst_center, worst_simplex = get_covering_info(current_points)
        
        if radius < best_radius - 1e-9:
            best_radius = radius
            best_points = current_points.copy()
            stuck_count = 0
        else:
            stuck_count += 1
            
        if worst_simplex is None:
            break
            
        # Move vertices of the worst simplex towards the circumcenter
        # The direction to move is (worst_center - v)
        # We project this move onto the tangent plane of the sphere at v
        # Actually, simply moving v towards worst_center and normalizing works well enough for small steps.
        
        moved = False
        for idx in worst_simplex:
            v = current_points[idx]
            # Direction to worst center
            dir_vec = worst_center - v
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir < 1e-10:
                continue
            dir_vec /= norm_dir
            
            # Update point
            new_v = v + current_step * dir_vec
            # Normalize
            new_v /= np.linalg.norm(new_v)
            current_points[idx] = new_v
            moved = True
            
        if not moved:
            break
            
        # Decay step size
        current_step *= decay_rate
        
        # If stuck for too long, or step size too small, perturb
        if stuck_count > 50 or current_step < min_step:
            # Add random perturbation to all points
            noise = np.random.randn(N, 3) * 0.05
            current_points = current_points + noise
            current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            current_step = base_step * 0.5
            stuck_count = 0
            
        # Occasional restart from a different initial configuration if progress is very slow
        # This is a heuristic; we check if we've been stuck for a long time
        if stuck_count > 500:
            # Restart with a new Fibonacci sphere with random rotation
            R = np.random.randn(3, 3)
            R, _ = np.linalg.qr(R)
            current_points = fibonacci_sphere(N) @ R.T
            current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            current_step = base_step
            stuck_count = 0

    # 3. Final Polish: Coordinate Descent / Local Search
    #    Try to improve the best solution found so far by moving individual points.
    
    if time.time() < deadline - 5:
        current_best = best_points.copy()
        current_score = best_radius
        step = 0.01
        
        # Try a few iterations of local refinement
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            # Shuffle points to avoid bias
            indices = np.random.permutation(N)
            for j in indices:
                if time.time() > deadline - 5:
                    break
                # Try moving point j in a random direction
                # Or better: move towards the circumcenter of its Voronoi cell (approximated by nearest neighbors)
                # For simplicity, try random small steps
                for _trial in range(10):
                    rand_vec = np.random.randn(3)
                    rand_vec /= np.linalg.norm(rand_vec)
                    
                    old_p = current_best[j].copy()
                    new_p = current_best[j] + step * rand_vec
                    new_p /= np.linalg.norm(new_p)
                    
                    current_best[j] = new_p
                    new_score = calc_covering_radius(current_best)
                    
                    if new_score < current_score - 1e-9:
                        current_score = new_score
                        improved = True
                        # If improved, keep the move and maybe try a bit more in the same direction?
                        # Or just break and try next point
                        break
                    else:
                        current_best[j] = old_p
                        
            if not improved:
                # No improvement in a full pass, reduce step size
                step *= 0.5
                if step < 1e-5:
                    break
        if current_score < best_radius:
            best_points = current_best
            best_radius = current_score

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
