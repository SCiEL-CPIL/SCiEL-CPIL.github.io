import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute all C(14,3)=364 triangle index triples
_idx = np.arange(N)
_ii, _jj, _kk = np.meshgrid(_idx, _idx, _idx, indexing='ij')
_mask = (_ii < _jj) & (_jj < _kk)
_I, _J, _K = _ii[_mask].flatten(), _jj[_mask].flatten(), _kk[_mask].flatten()


def _all_areas(pts):
    """Vectorized area computation for all 364 triangles."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)


def _hull_area(pts):
    """Area of convex hull; returns 0 if degenerate."""
    try:
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
        h = ConvexHull(pts)
        return h.volume
    except Exception:
        return 0.0


def _score(pts):
    """min triangle area / convex hull area."""
    h = _hull_area(pts)
    if h < 1e-12:
        return 0.0
    return float(np.min(_all_areas(pts))) / h


def _pts_from_params(params):
    """Convert (r_0..r_13, theta_0..theta_13) to point coordinates.
    
    params shape: (28,) -> r[0:14], theta[14:28]
    Points are in polar coordinates, then sorted by angle for convexity.
    """
    r = np.clip(params[:N], 0.05, 3.0)
    theta = params[N:2*N]
    # Sort by angle to maintain convex ordering
    order = np.argsort(theta)
    r_sorted = r[order]
    theta_sorted = theta[order]
    # Make angles span exactly [0, 2*pi) by normalizing
    theta_sorted = theta_sorted - theta_sorted[0]
    theta_sorted = theta_sorted / (2 * np.pi) * 2 * np.pi  # ensure full circle
    x = r_sorted * np.cos(theta_sorted)
    y = r_sorted * np.sin(theta_sorted)
    return np.column_stack((x, y))


def _sa_objective(params):
    """Objective for SA: minimize hull_area / min_triangle_area (lower = better)."""
    pts = _pts_from_params(params)
    h = _hull_area(pts)
    if h < 1e-12:
        return 1e6
    min_a = np.min(_all_areas(pts))
    if min_a < 1e-12:
        return 1e6
    return h / min_a


def _directed_refine(pts, max_iter=2000, tol=1e-10):
    """Refine points directly using Nelder-Mead on -score."""
    def obj(flat):
        p = flat.reshape(N, 2)
        s = _score(p)
        return -s if s > 1e-12 else 1.0
    
    try:
        res = minimize(obj, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
        cand = res.x.reshape(N, 2)
        return cand, _score(cand)
    except Exception:
        return pts, _score(pts)


def _sa_search(deadline, initial_params, T0=1.0, T_end=0.001, max_iter=50000):
    """Simulated annealing on the 28D parameter space."""
    best_params = initial_params.copy()
    best_obj = _sa_objective(best_params)
    cur_params = initial_params.copy()
    cur_obj = _sa_objective(cur_params)
    
    n_iter = 0
    while n_iter < max_iter and time.time() < deadline:
        # Temperature schedule (geometric cooling)
        frac = n_iter / max_iter
        T = T0 * (T_end / T0) ** frac
        
        # Perturbation: random walk in parameter space
        step_size = 0.1 * (1.0 + 0.5 * np.random.randn(2 * N))
        new_params = cur_params + step_size
        # Keep radii positive
        new_params[:N] = np.clip(new_params[:N], 0.05, 3.0)
        
        new_obj = _sa_objective(new_params)
        
        # Accept or reject
        if new_obj < cur_obj:
            cur_params = new_params
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_params = new_params.copy()
        else:
            delta = new_obj - cur_obj
            if delta < 1e-10:
                accept = True
            else:
                accept = np.random.random() < np.exp(-delta / max(T, 1e-10))
            if accept:
                cur_params = new_params
                cur_obj = new_obj
        
        n_iter += 1
        # Check time every 1000 iterations
        if n_iter % 1000 == 0 and time.time() > deadline:
            break
    
    return best_params, best_obj


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # --- Phase 1: Multi-start Simulated Annealing on convex parameter space ---
    # Initialize with regular 14-gon in polar form
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_radii = np.ones(N)
    
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        # Different initial perturbations
        np.random.seed(42 + s * 7)
        if s == 0:
            # Regular 14-gon
            init_r = np.ones(N)
            init_theta = base_angles.copy()
        elif s == 1:
            # Slightly perturbed regular
            init_r = 1.0 + 0.1 * np.random.randn(N)
            init_theta = base_angles + 0.05 * np.random.randn(N)
        elif s == 2:
            # Two concentric rings (7 inner, 7 outer)
            init_r = np.concatenate([0.6 * np.ones(N_HALF), 1.4 * np.ones(N_HALF)])
            init_theta = base_angles.copy()
        elif s == 3:
            # Random convex-ish start
            init_r = 0.7 + 0.5 * np.random.rand(N)
            init_theta = np.sort(np.random.uniform(0, 2 * np.pi, N))
        elif s == 4:
            # Alternating radii
            init_r = np.array([1.0, 0.7] * 7)
            init_theta = base_angles.copy()
        elif s == 5:
            # Near-square with extra points
            init_r = np.ones(N)
            init_theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.1 * np.sin(np.linspace(0, 4*np.pi, N))
        elif s == 6:
            # Hexagonal-ish with extras
            init_r = np.ones(N)
            init_theta = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.15 * np.cos(np.linspace(0, 6*np.pi, N))
        else:
            # Very random
            init_r = 0.5 + np.random.rand(N)
            init_theta = np.sort(np.random.uniform(0, 2 * np.pi, N))
        
        init_params = np.concatenate([init_r, init_theta])
        
        # Run SA
        sa_params, sa_obj = _sa_search(deadline - 25.0, init_params, 
                                        T0=0.5, T_end=0.0005, 
                                        max_iter=20000)
        
        # Convert to points and score
        pts = _pts_from_params(sa_params)
        sc = _score(pts)
        if sc > 1e-12:
            update_best(pts, sc)
    
    # --- Phase 2: Directed refinement of best SA result ---
    if best_points is not None and time.time() < deadline - 15.0:
        cand_pts, cand_score = _directed_refine(best_points, max_iter=3000, tol=1e-11)
        update_best(cand_pts, cand_score)
    
    # --- Phase 3: Powell refinement ---
    if best_points is not None and time.time() < deadline - 10.0:
        def obj_powell(flat):
            p = flat.reshape(N, 2)
            s = _score(p)
            return -s if s > 1e-12 else 1.0
        try:
            res = minimize(obj_powell, best_points.flatten(), method='Powell',
                           options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-14})
            cand = res.x.reshape(N, 2)
            sc = _score(cand)
            if sc > 1e-12:
                update_best(cand, sc)
        except Exception:
            pass
    
    # --- Phase 4: Second SA round from refined point ---
    if best_points is not None and time.time() < deadline - 8.0:
        # Convert refined points back to polar parameters
        cx, cy = np.mean(best_points[:, 0]), np.mean(best_points[:, 1])
        dx = best_points[:, 0] - cx
        dy = best_points[:, 1] - cy
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        # Sort by angle
        order = np.argsort(theta)
        r_sorted = r[order]
        theta_sorted = theta[order]
        # Normalize theta to [0, 2*pi)
        theta_sorted = (theta_sorted - theta_sorted[0]) % (2 * np.pi)
        # Ensure sorted
        theta_sorted = np.sort(theta_sorted)
        
        init_params2 = np.concatenate([r_sorted, theta_sorted])
        np.random.seed(999)
        sa_params2, sa_obj2 = _sa_search(deadline - 5.0, init_params2,
                                           T0=0.2, T_end=0.0002,
                                           max_iter=15000)
        pts2 = _pts_from_params(sa_params2)
        sc2 = _score(pts2)
        if sc2 > 1e-12:
            update_best(pts2, sc2)
    
    # --- Phase 5: Final high-precision Nelder-Mead polish ---
    if best_points is not None and time.time() < deadline - 2.0:
        cand_pts, cand_score = _directed_refine(best_points, max_iter=5000, tol=1e-13)
        update_best(cand_pts, cand_score)
    
    # Fallback: regular 14-gon
    if best_points is None or best_score < 1e-12:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
