# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

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


def _fast_metric(points: np.ndarray) -> float:
    """Faster metric computation using vectorized distances."""
    n = points.shape[0]
    # Compute all pairwise squared distances
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # sq_dists = np.sum(diff**2, axis=2)
    # Use broadcasting
    sq_dists = np.sum(points**2, axis=1)[:, np.newaxis] + np.sum(points**2, axis=1)[np.newaxis, :] - 2 * points @ points.T
    np.fill_diagonal(sq_dists, 0.0)
    sq_dists = np.maximum(sq_dists, 0.0)
    d_min_sq = np.min(sq_dists[sq_dists > 0]) if np.any(sq_dists > 0) else 0.0
    d_max_sq = np.max(sq_dists)
    if d_max_sq <= 0:
        return 0.0
    d_min = np.sqrt(d_min_sq)
    d_max = np.sqrt(d_max_sq)
    if d_min <= 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 615.0  # safety margin before 630s kill

    best_points = None
    best_score = -1.0

    def _try_candidate(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            return
        if not np.all(np.isfinite(pts)):
            return
        score = _fast_metric(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            # Also verify with the exact metric occasionally
            if score > 0.3:
                exact_score = _compute_metric(pts)
                if exact_score > best_score:
                    best_score = exact_score

    # --- Initializations ---
    
    # 1. Random uniform on sphere with different seeds
    np.random.seed(12345)
    for seed in range(30):
        np.random.seed(seed)
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        _try_candidate(pts)
    
    # 2. Icosahedron-based (12 vertices) + 2 more at various positions
    phi = (1 + 5**0.5) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / ico_norms
    
    # Add 2 more points at various locations
    for trial in range(40):
        angle1 = np.random.uniform(0, 2 * np.pi)
        angle2 = np.random.uniform(0, np.pi)
        extra1 = np.array([np.sin(angle2) * np.cos(angle1), np.sin(angle2) * np.sin(angle1), np.cos(angle2)])
        
        angle3 = np.random.uniform(0, 2 * np.pi)
        angle4 = np.random.uniform(0, np.pi)
        extra2 = np.array([np.sin(angle4) * np.cos(angle3), np.sin(angle4) * np.sin(angle3), np.cos(angle4)])
        
        pts = np.vstack([ico, extra1, extra2])
        _try_candidate(pts)
    
    # 3. Fibonacci sphere
    golden_ratio = (1 + np.sqrt(5)) / 2
    for _ in range(5):
        pts = np.zeros((n, d))
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            pts[i, 0] = r * np.cos(theta)
            pts[i, 1] = r * np.sin(theta)
            pts[i, 2] = z
        _try_candidate(pts)
    
    # 4. Random perturbations of icosahedron
    for trial in range(20):
        pts = ico.copy()
        noise = np.random.randn(2, d) * 0.3
        pts = np.vstack([pts, noise])
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        _try_candidate(pts)

    # --- Optimization Phase ---
    # Use gradient-free optimization with multiple strategies
    
    from scipy.optimize import minimize
    
    if best_points is not None:
        best_pts_copy = best_points.copy()
        
        # Strategy 1: Nelder-Mead on the full parameter space
        # Center the points to reduce degrees of freedom
        centered = best_pts_copy - np.mean(best_pts_copy, axis=0)
        x0 = centered.flatten()
        
        def objective_nm(x):
            pts = x.reshape(n, d)
            pts = pts - np.mean(pts, axis=0)
            norm = np.linalg.norm(pts)
            if norm < 1e-10:
                return 1e10
            return -_fast_metric(pts)
        
        # Run Nelder-Mead with time check
        elapsed = time.time() - start_time
        if elapsed < deadline - 60:
            try:
                res = minimize(objective_nm, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'maxfev': 15000, 'xatol': 1e-8, 'fatol': 1e-10})
                candidate = res.x.reshape(n, d)
                candidate = candidate - np.mean(candidate, axis=0)
                _try_candidate(candidate)
            except Exception:
                pass
        
        # Strategy 2: Coordinate descent / local search
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            current_pts = best_points.copy()
            current_score = best_score
            step_size = 0.1
            for iter in range(20000):
                if time.time() > deadline - 10:
                    break
                # Random coordinate perturbation
                dim_idx = np.random.randint(n * d)
                point_idx = dim_idx // d
                coord_idx = dim_idx % d
                old_val = current_pts[point_idx, coord_idx]
                current_pts[point_idx, coord_idx] = old_val + np.random.randn() * step_size
                new_score = _fast_metric(current_pts)
                if new_score > current_score:
                    current_score = new_score
                    step_size *= 0.95
                else:
                    current_pts[point_idx, coord_idx] = old_val
                    step_size = min(step_size * 1.05, 0.5)
            _try_candidate(current_pts)
        
        # Strategy 3: Simulated Annealing
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            sa_pts = best_points.copy()
            sa_score = best_score
            T = 1.0
            cooling = 0.99995
            for iter in range(50000):
                if time.time() > deadline - 5:
                    break
                # Perturb a random point
                pt_idx = np.random.randint(n)
                old_val = sa_pts[pt_idx].copy()
                sa_pts[pt_idx] = old_val + np.random.randn(d) * 0.1
                new_score = _fast_metric(sa_pts)
                delta = new_score - sa_score
                if delta > 0 or np.random.random() < np.exp(delta / T):
                    sa_pts[pt_idx] = sa_pts[pt_idx]  # keep
                    sa_score = new_score
                else:
                    sa_pts[pt_idx] = old_val
                T *= cooling
            _try_candidate(sa_pts)

    # If best_points is still None, use fallback
    if best_points is None:
        # Return a reasonable default
        phi = (1 + 5**0.5) / 2
        ico = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
        ico = ico / ico_norms
        best_points = np.vstack([ico, np.array([[0, 0, 1]], dtype=np.float64), np.array([[0, 0, -1]], dtype=np.float64)])
    
    # Final check: ensure shape and finite
    best_points = best_points.reshape(n, d)
    if not np.all(np.isfinite(best_points)):
        phi = (1 + 5**0.5) / 2
        ico = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
        ico = ico / ico_norms
        best_points = np.vstack([ico, np.array([[0, 0, 1]], dtype=np.float64), np.array([[0, 0, -1]], dtype=np.float64)])
    
    return best_points

# EVOLVE-BLOCK-END
