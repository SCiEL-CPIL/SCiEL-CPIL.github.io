import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area relative to the hull area.
    Strategy:
    1. Fix the convex hull as a regular hexagon (known to be optimal or near-optimal for small n).
    2. Optimize the 8 internal points using a smooth soft-min surrogate.
    3. Polish with Simulated Annealing to handle the non-smoothness of the true min-area metric.
    """
    n = 13
    start_time = time.time()
    time_limit = 600  # Leave 30s margin from 630s

    # Pre-compute triangle indices for all C(13,3) = 286 triangles
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    i_idx = tri_indices[:, 0]
    j_idx = tri_indices[:, 1]
    k_idx = tri_indices[:, 2]

    def calculate_score(points):
        """
        Calculate A_min / A_hull.
        """
        if not np.all(np.isfinite(points)):
            return 0.0

        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-12:
                return 0.0
        except Exception:
            return 0.0

        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 0:
            return 0.0
            
        return A_min / A_hull

    def soft_min_objective(params, beta=20.0):
        """
        Smooth surrogate for -A_min.
        params: flattened array of 8 internal points (16 coords).
        We fix the 6 hull points to a regular hexagon.
        """
        # Reconstruct full points array
        points = np.empty((n, 2))
        # Hull points (indices 0-5) fixed to regular hexagon
        # Center at (0,0), radius 1
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        hull_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        points[:6] = hull_pts
        
        # Internal points (indices 6-12) from params
        internal_pts = params.reshape((7, 2)) # Wait, 13-6=7 internal points?
        # Let's re-verify: 13 points total.
        # If we fix 6 hull points, we have 7 internal points.
        # 7 * 2 = 14 parameters.
        
        points[6:] = internal_pts
        
        # Calculate areas
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        # Soft min: -log(sum(exp(-areas/beta))) / beta approximates min(areas)
        # To avoid overflow, shift by max
        min_val = np.min(areas)
        exp_areas = np.exp(-(areas - min_val) / beta)
        soft_min_val = min_val - beta * np.log(np.sum(exp_areas))
        
        # We want to maximize soft_min_val, so minimize -soft_min_val
        # Also, we must ensure the internal points stay inside the hull?
        # The optimizer might push them out. 
        # To keep them "convex region" compliant, we rely on the fact that if they go out, 
        # the hull changes. But we FIXED the hull points. 
        # If an internal point goes outside the hexagon, the "hull" of the 13 points 
        # would no longer be the hexagon. 
        # The metric uses the hull of the ACTUAL 13 points.
        # So if a point goes outside, A_hull increases, potentially lowering the score.
        # However, the soft_min_objective doesn't know about A_hull.
        # It just maximizes the min area of the triangles formed by the 13 points.
        # If a point is far outside, the min area might still be large, but A_hull is huge.
        # So we need to include A_hull in the objective or constrain the points.
        
        # Better approach: 
        # 1. Compute A_hull of the 13 points.
        # 2. Return -(soft_min / A_hull).
        
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull < 1e-12:
                return 1e6
        except Exception:
            return 1e6
            
        score = soft_min_val / A_hull
        return -score

    # Initial guess for internal points: 
    # Use a hexagonal grid pattern scaled to fit inside the unit hexagon.
    # Hexagon vertices at radius 1.
    # Internal points can be placed at a smaller hexagonal lattice.
    
    # Let's generate a few random initializations inside the hexagon
    def random_point_in_hexagon(rng):
        # Rejection sampling
        while True:
            p = rng.uniform(-1, 1, 2)
            # Check if inside hexagon
            # Hexagon defined by |x| <= 1, |y| <= 1, |x + y/2| <= 1, etc?
            # Simpler: use the convex hull check or just sample and check distance to edges.
            # Or just sample in [-0.8, 0.8] square, which is mostly inside.
            # Actually, let's just sample in the bounding box [-1, 1] and check if inside.
            # For speed, let's just use a smaller square [-0.7, 0.7] which is definitely inside the hexagon.
            return p

    best_points = None
    best_score = -np.inf
    rng = np.random.default_rng(42)

    # Phase 1: Optimization with fixed hexagon hull
    # Number of internal points: 7
    num_internal = n - 6
    bounds = [(-1.0, 1.0)] * (num_internal * 2)

    # Try a few random starts
    for _ in range(20):
        if time.time() - start_time > time_limit - 60:
            break
            
        # Random start inside the hexagon
        # We can use the fact that the centroid of the hexagon is (0,0).
        # Let's just pick random points in [-0.8, 0.8]
        init_params = rng.uniform(-0.8, 0.8, (num_internal * 2,))
        
        try:
            res = minimize(
                soft_min_objective,
                init_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            # Reconstruct points
            points = np.empty((n, 2))
            angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            hull_pts = np.column_stack((np.cos(angles), np.sin(angles)))
            points[:6] = hull_pts
            points[6:] = res.x.reshape((num_internal, 2))
            
            score = calculate_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            continue

    # Phase 2: Simulated Annealing Polish
    # This helps escape local minima of the soft-min and handle the sharp min.
    if best_points is not None:
        current_points = best_points.copy()
        current_score = best_score
        
        # SA parameters
        T = 1e-4
        T_min = 1e-7
        cooling_rate = 0.95
        max_iter = 2000
        
        # Pre-compute triangle indices for fast evaluation
        # We already have i_idx, j_idx, k_idx
        
        def evaluate(points):
            try:
                hull = ConvexHull(points)
                A_hull = hull.volume
                if A_hull < 1e-12:
                    return 0.0
            except Exception:
                return 0.0
                
            p1 = points[i_idx]
            p2 = points[j_idx]
            p3 = points[k_idx]
            cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
            areas = 0.5 * np.abs(cross)
            A_min = np.min(areas)
            if A_min < 0:
                return 0.0
            return A_min / A_hull

        for i in range(max_iter):
            if time.time() - start_time > time_limit - 10:
                break
                
            # Perturb one or two random points
            num_perturb = 1 if rng.random() < 0.7 else 2
            indices = rng.choice(n, size=num_perturb, replace=False)
            
            new_points = current_points.copy()
            for idx in indices:
                # Step size proportional to T
                step = T * 100 # Scale step
                delta = rng.normal(0, step, 2)
                new_points[idx] += delta
                
            new_score = evaluate(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Accept with probability exp((new - old) / T)
                # Since we maximize, if new < old, delta_S is negative.
                # P = exp((new_score - current_score) / T)
                delta_S = new_score - current_score
                if delta_S < 0:
                    accept_prob = np.exp(delta_S / T)
                    if rng.random() < accept_prob:
                        current_points = new_points
                        current_score = new_score
            
            T *= cooling_rate
            if T < T_min:
                break

    # Final check and normalization
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    # The score is scale invariant, but let's normalize to unit area hull for consistency?
    # The task asks for coordinates, score is A_min/A_hull.
    # Returning the points as is is fine.
    
    return best_points
