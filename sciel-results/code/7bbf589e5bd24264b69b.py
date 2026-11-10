# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin before 630s kill

    best_score = -1.0
    best_points = None

    def _compute_metric(points: np.ndarray) -> float:
        """Compute (d_min / d_max)^2 for the given points."""
        if points.shape[0] < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        if d_min <= 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def _fast_metric_and_extremes(points: np.ndarray):
        """
        Returns (score, d_min, d_max, min_idx, max_idx)
        min_idx and max_idx are indices into the upper triangle of the distance matrix.
        """
        n_pts = points.shape[0]
        sq_dists = np.sum(points**2, axis=1)[:, np.newaxis] + np.sum(points**2, axis=1)[np.newaxis, :] - 2 * points @ points.T
        np.fill_diagonal(sq_dists, 0.0)
        sq_dists = np.maximum(sq_dists, 0.0)
        
        # Get upper triangle indices
        iu = np.triu_indices(n_pts, 1)
        sq_dists_flat = sq_dists[iu]
        
        if len(sq_dists_flat) == 0:
            return 0.0, 0.0, 0.0, 0, 0
            
        min_pos = np.argmin(sq_dists_flat)
        max_pos = np.argmax(sq_dists_flat)
        
        d_min_sq = sq_dists_flat[min_pos]
        d_max_sq = sq_dists_flat[max_pos]
        
        if d_max_sq <= 0 or d_min_sq <= 0:
            return 0.0, 0.0, 0.0, min_pos, max_pos
            
        d_min = np.sqrt(d_min_sq)
        d_max = np.sqrt(d_max_sq)
        
        ratio = d_min / d_max
        score = ratio * ratio
        
        return score, d_min, d_max, min_pos, max_pos

    def _try_candidate(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            return
        if not np.all(np.isfinite(pts)):
            return
        score, _, _, _, _ = _fast_metric_and_extremes(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def _flat_to_sphere(x):
        """Convert flat vector of (theta, phi) pairs to 3D unit vectors."""
        # x is shape (2*n,)
        # theta in [0, 2*pi), phi in [0, pi]
        theta = x[0::2]
        phi = x[1::2]
        
        # Ensure bounds are respected by wrapping/clipping
        theta = np.mod(theta, 2 * np.pi)
        phi = np.clip(phi, 1e-6, np.pi - 1e-6)
        
        x_coords = np.sin(phi) * np.cos(theta)
        y_coords = np.sin(phi) * np.sin(theta)
        z_coords = np.cos(phi)
        
        pts = np.column_stack([x_coords, y_coords, z_coords])
        return pts

    def _sphere_to_flat(pts):
        """Convert 3D unit vectors to flat vector of (theta, phi) pairs."""
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        theta = np.mod(theta, 2 * np.pi)
        phi = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
        
        x = np.empty(2 * n)
        x[0::2] = theta
        x[1::2] = phi
        return x

    def objective_sphere(x):
        """
        Objective for L-BFGS-B.
        Returns -log(d_min/d_max) = log(d_max) - log(d_min).
        Minimizing this maximizes the ratio.
        """
        pts = _flat_to_sphere(x)
        score, d_min, d_max, _, _ = _fast_metric_and_extremes(pts)
        
        if d_min <= 1e-9 or d_max <= 1e-9:
            return 1e6  # Large penalty
            
        # Use log ratio for better conditioning
        # Maximize log(d_min) - log(d_max)
        # So minimize log(d_max) - log(d_min)
        return np.log(d_max) - np.log(d_min)

    def _local_refinement(points, max_iters=500):
        """
        Local refinement using swap-and-nudge strategy.
        Targets the current min-distance pair (try to increase distance)
        and max-distance pair (try to decrease distance).
        """
        current_pts = points.copy()
        current_score, d_min, d_max, min_idx, max_idx = _fast_metric_and_extremes(current_pts)
        
        # Map flat index back to (i, j)
        n_pts = n
        # Number of elements in upper triangle before row i
        # We need to find i, j such that idx = i*n - i*(i+1)/2 + (j - i - 1)
        def idx_to_pair(idx):
            # Binary search or direct calculation
            # Number of pairs = n*(n-1)/2
            # Let k = i
            # Sum_{m=0}^{i-1} (n-1-m) = i*n - i*(i+1)/2
            # We want largest i such that i*n - i*(i+1)/2 <= idx
            # Approximation: i ~ sqrt(2*idx)
            i = int(np.sqrt(2 * idx))
            while True:
                offset = i * n - i * (i + 1) // 2
                if offset + (n - 1 - i) > idx:
                    break
                i += 1
            offset = i * n - i * (i + 1) // 2
            j = i + 1 + (idx - offset)
            return i, j

        min_i, min_j = idx_to_pair(min_idx)
        max_i, max_j = idx_to_pair(max_idx)

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            
            # 1. Try to increase d_min by moving min_i and min_j apart
            # Move min_i away from min_j
            vec = current_pts[min_i] - current_pts[min_j]
            norm_vec = np.linalg.norm(vec)
            if norm_vec > 1e-9:
                vec /= norm_vec
            else:
                vec = np.random.randn(3)
                vec /= np.linalg.norm(vec)
            
            step_size = 0.05
            for alpha in [step_size, step_size/2, step_size/4]:
                test_pts = current_pts.copy()
                test_pts[min_i] += vec * alpha
                # Normalize
                test_pts[min_i] /= np.linalg.norm(test_pts[min_i])
                
                new_score, _, _, _, _ = _fast_metric_and_extremes(test_pts)
                if new_score > current_score:
                    current_pts = test_pts
                    current_score = new_score
                    improved = True
                    break
            
            # 2. Try to decrease d_max by moving max_i and max_j closer
            vec = current_pts[max_i] - current_pts[max_j]
            norm_vec = np.linalg.norm(vec)
            if norm_vec > 1e-9:
                vec /= norm_vec
            else:
                vec = np.random.randn(3)
                vec /= np.linalg.norm(vec)
            
            for alpha in [step_size, step_size/2, step_size/4]:
                test_pts = current_pts.copy()
                test_pts[max_i] -= vec * alpha # Move towards max_j
                test_pts[max_i] /= np.linalg.norm(test_pts[max_i])
                
                new_score, _, _, _, _ = _fast_metric_and_extremes(test_pts)
                if new_score > current_score:
                    current_pts = test_pts
                    current_score = new_score
                    improved = True
                    break
            
            # 3. Random nudge if stuck
            if not improved:
                idx = np.random.randint(n)
                test_pts = current_pts.copy()
                test_pts[idx] += np.random.randn(3) * 0.05
                test_pts[idx] /= np.linalg.norm(test_pts[idx])
                
                new_score, _, _, _, _ = _fast_metric_and_extremes(test_pts)
                if new_score > current_score:
                    current_pts = test_pts
                    current_score = new_score
                else:
                    # Accept with small probability if not too bad (SA-like)
                    if np.random.rand() < 0.1:
                        current_pts = test_pts
                        current_score = new_score

            _try_candidate(current_pts)
            
            # Update extremes
            _, d_min, d_max, min_idx, max_idx = _fast_metric_and_extremes(current_pts)
            if d_min <= 1e-9 or d_max <= 1e-9:
                break
            min_i, min_j = idx_to_pair(min_idx)
            max_i, max_j = idx_to_pair(max_idx)

        return current_pts

    # Initializations
    np.random.seed(42)
    
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - (i + 0.5) * 2 / n
            radius = np.sqrt(1 - z*z)
            theta = 2 * np.pi * i / golden_ratio
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    start_points = []
    
    # 1. Fibonacci Sphere
    fib_pts = fibonacci_sphere(n)
    start_points.append(fib_pts)
    
    # 2. Random points
    for _ in range(5):
        rp = np.random.randn(n, d)
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        start_points.append(rp)
        
    # 3. Icosahedron + 2 random
    phi = (1 + 5**0.5) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / ico_norms
    extra = np.random.randn(2, d)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    start_points.append(np.vstack([ico, extra]))

    # Optimization Loop
    for i, init_pts in enumerate(start_points):
        if time.time() > deadline - 20:
            break
            
        x0 = _sphere_to_flat(init_pts)
        
        # Define bounds for L-BFGS-B
        # theta: [0, 2*pi], phi: [0, pi]
        bounds = []
        for _ in range(n):
            bounds.append((0, 2 * np.pi))  # theta
            bounds.append((0, np.pi))      # phi
            
        try:
            # L-BFGS-B optimization
            res = minimize(objective_sphere, x0, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e5:
                candidate_pts = _flat_to_sphere(res.x)
                _try_candidate(candidate_pts)
                
                # Local refinement
                if time.time() < deadline - 10:
                    refined = _local_refinement(candidate_pts)
                    _try_candidate(refined)
                    
        except Exception:
            pass

    # Final Fallback
    if best_points is None:
        best_points = fib_pts

    # Ensure the returned points are finite and have the correct shape
    best_points = best_points.reshape(n, d)
    if not np.all(np.isfinite(best_points)):
        best_points = fib_pts

    return best_points

# EVOLVE-BLOCK-END
