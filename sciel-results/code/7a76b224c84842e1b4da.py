"""Combined program for Heilbronn-11: barycentric representation + Cartesian refinement."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 142  # safety margin from 150s limit

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _bary_to_cart(uv_flat):
    """Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2).
    w = 1 - u - v.
    P = u*V0 + v*V1 + w*V2.
    Since V0=(0,0), P = v*(1,0) + w*(0.5, S3H) = (v + 0.5w, w*S3H).
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))


def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v) where w = 1-u-v."""
    uv = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        x, y = pts[i]
        w = y / S3H if S3H != 0 else 0
        v = x - 0.5 * w
        u = 1.0 - v - w
        uv[i] = [u, v]
    return uv


def _vectorized_min_area_cart(pts):
    """Vectorized computation of min triangle area from Cartesian points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area_cart(pts, beta):
    """Soft minimum of triangle areas."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_cartesian(pts):
    """Project points into the triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _project_to_simplex(uv_flat):
    """Project barycentric coords to the simplex u>=0, v>=0, u+v<=1."""
    uv = uv_flat.reshape(-1, 2).copy()
    for i in range(len(uv)):
        u, v = uv[i]
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        if u + v > 1:
            scale = 1.0 / (u + v)
            u *= scale
            v *= scale
        uv[i] = [u, v]
    return uv.flatten()


def _is_valid_cart(pts, tol=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True


def _is_valid_bary(uv_flat, tol=1e-6):
    """Check if all points are inside the triangle in barycentric coords."""
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    if np.any(w < -tol):
        return False
    if np.any(u < -tol) or np.any(v < -tol):
        return False
    return True


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area = _vectorized_min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Initial configuration: corners, mid-edges, centroid, and some interior points
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [1/3, S3H/3],        # centroid
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)
    
    # Convert initial to barycentric
    best_uv = _cart_to_bary(best_pts) if best_pts is not None else None
    
    def _start_from_base(base, scale):
        pts = base + rng.normal(0, scale, size=(N, 2))
        pts = _project_cartesian(pts)
        return pts
    
    def _optimize_bary(uv_start, beta):
        """Optimize in barycentric coordinates with soft-min objective."""
        def objective(x):
            pts = _bary_to_cart(x)
            # Penalty for invalid barycentric coords
            uv = x.reshape(-1, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1.0 - u - v
            viol_w = np.maximum(0, -w)
            penalty = 1000.0 * np.sum(viol_w ** 2)
            
            ma = _soft_min_area_cart(pts, beta)
            return -ma + penalty
        
        bounds = [(0, 1), (0, 1)] * N
        
        try:
            result = minimize(
                objective,
                uv_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            return result.x
        except Exception:
            return uv_start
    
    def _optimize_cart(pts_start, beta):
        """Optimize in Cartesian coordinates with soft-min objective."""
        def objective(x):
            pts = x.reshape(-1, 2)
            pts = _project_cartesian(pts)
            ma = _soft_min_area_cart(pts, beta)
            return -ma
        
        try:
            result = minimize(
                objective,
                pts_start.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts = result.x.reshape(-1, 2)
            pts = _project_cartesian(pts)
            return pts
        except Exception:
            return pts_start
    
    # Phase A: Multi-start optimization in both barycentric and Cartesian
    n_starts = 30
    bases = [init]
    for _ in range(15):
        bases.append(_start_from_base(init, 0.25))
    
    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.06)
        _update_best(pts)
        
        # Alternate between barycentric and Cartesian optimization
        if s % 2 == 0:
            # Barycentric optimization
            uv = _cart_to_bary(pts)
            for beta in [10.0, 30.0, 80.0, 200.0]:
                if not _elapsed_ok(40):
                    break
                uv = _optimize_bary(uv, beta)
                uv = _project_to_simplex(uv)
                pts = _bary_to_cart(uv)
                _update_best(pts)
        else:
            # Cartesian optimization
            for beta in [10.0, 30.0, 80.0, 200.0]:
                if not _elapsed_ok(40):
                    break
                pts = _optimize_cart(pts, beta)
                _update_best(pts)
    
    # Phase B: Nelder-Mead local polish on the best configuration
    if best_pts is not None and _elapsed_ok(15):
        for scale in [0.05, 0.02, 0.01]:
            if not _elapsed_ok(10):
                break
            for _ in range(8):
                if not _elapsed_ok(8):
                    break
                cand = best_pts + rng.normal(0, scale, size=(N, 2))
                cand = _project_cartesian(cand)
                
                def nm_obj(x):
                    p = x.reshape(-1, 2)
                    p = _project_cartesian(p)
                    return -_vectorized_min_area_cart(p)
                
                try:
                    res = minimize(nm_obj, cand.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
                    p = res.x.reshape(-1, 2)
                    p = _project_cartesian(p)
                    _update_best(p)
                except Exception:
                    pass
    
    # Phase C: Final fine polish on the absolute best
    if best_pts is not None and _elapsed_ok(5):
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_cartesian(p)
            return -_vectorized_min_area_cart(p)
        
        try:
            res = minimize(nm_obj, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-11, 'fatol': 1e-13})
            p = res.x.reshape(-1, 2)
            p = _project_cartesian(p)
            _update_best(p)
        except Exception:
            pass
    
    # Final check and return
    if best_pts is None:
        # Fallback: random feasible configuration
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = v + 0.5 * (1 - u - v)
            y = (1 - u - v) * S3H
            pts[i] = [x, y]
        return pts
    
    # Ensure the final result is valid
    best_pts = _project_cartesian(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
