import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _facet_circumradii(pts, simplices):
    """Vectorized computation of circumradii for all facets.
    Returns array of radii (radians) and array of circumcenter unit vectors."""
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    n = np.cross(a - b, a - c)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm > 1e-10
    n = n / np.maximum(n_norm[:, None], 1e-10)
    centroids = (a + b + c) / 3.0
    signs = np.where(np.einsum('ij,ij->i', n, centroids) < 0, -1.0, 1.0)
    n = n * signs[:, None]
    dots = np.clip(np.einsum('ij,ij->i', n, a), -1.0, 1.0)
    radii = np.arccos(dots)
    radii[~mask] = 0.0
    return radii, n, mask

def _covering_radius_deg(pts):
    """Exact covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.degrees(np.max(radii)))
    except Exception:
        return 180.0

def _covering_radius_rad(pts):
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.max(radii))
    except Exception:
        return np.pi

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _electrostatic_refine(pts, n_iters=500, lr=0.01):
    """Repulsion-based refinement: each point is pushed away by all others via Coulomb-like force."""
    pts = pts.copy()
    for it in range(n_iters):
        D = pts @ pts.T
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        forces /= (np.linalg.norm(forces, axis=1, keepdims=True) + 1e-12)
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts

def _smooth_objective_and_grad(x_flat, beta=20.0):
    """Compute smooth surrogate objective and gradient for L-BFGS-B.
    x_flat: flat array of 3N values (unnormalized points).
    Returns (objective, gradient_flat)."""
    pts = x_flat.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1e6, np.zeros_like(x_flat)
    
    radii, cc, mask = _facet_circumradii(pts, hull.simplices)
    if not np.any(mask):
        return 1e6, np.zeros_like(x_flat)
    
    valid = mask
    r_valid = radii[valid]
    cc_valid = cc[valid]
    simp_valid = hull.simplices[valid]
    
    r2 = r_valid * r_valid
    m = np.max(r2)
    exps = np.exp((r2 - m) * beta)
    s = np.sum(exps)
    if s < 1e-300:
        weights = np.ones_like(exps) / len(exps)
        surrogate = m
    else:
        weights = exps / s
        surrogate = m + np.log(s) / beta
    
    # Gradient computation
    grad = np.zeros_like(pts)
    for k in range(len(simp_valid)):
        w = weights[k]
        if w < 1e-10:
            continue
        c = cc_valid[k]
        r_k = r_valid[k]
        if r_k < 1e-10:
            continue
        sin_r = np.sin(r_k)
        if sin_r < 1e-10:
            sin_r = 1e-10
        # d(r^2)/d(v) tangent direction: push v toward c
        # Magnitude: 2 * r / sin(r) * w * (1/beta) for the log-sum-exp scaling
        mag = 2.0 * r_k / sin_r * w / beta
        for j in range(3):
            idx = simp_valid[k, j]
            v = pts[idx]
            d = c - v
            d_t = d - np.dot(d, v) * v
            d_t_norm = np.linalg.norm(d_t)
            if d_t_norm < 1e-10:
                continue
            d_t = d_t / d_t_norm
            grad[idx] -= mag * d_t
    
    # Project gradient to tangent space
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    
    return surrogate, grad.flatten()

def _lgbfgs_optimize(pts, max_time_remaining, beta=20.0, max_iter=200):
    """Run L-BFGS-B on the smooth surrogate. Returns optimized points."""
    start_t = time.time()
    x0 = pts.flatten()
    
    def obj_and_grad(x_flat):
        elapsed = time.time() - start_t
        if elapsed > max_time_remaining - 2.0:
            raise RuntimeError("timeout")
        val, grad = _smooth_objective_and_grad(x_flat, beta=beta)
        return val, grad
    
    try:
        result = minimize(
            obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-6, 'maxfun': max_iter * 10}
        )
        pts_opt = result.x.reshape(N, 3)
        pts_opt = _normalize(pts_opt)
        return pts_opt
    except Exception:
        return pts

def _polish_worst_facets(pts, max_iters=100, tol=1e-8):
    """Directly polish the worst facets by moving their vertices toward circumcenters."""
    pts = pts.copy()
    for it in range(max_iters):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        radii, cc, mask = _facet_circumradii(pts, hull.simplices)
        if not np.any(mask):
            break
        valid = mask
        r_valid = radii[valid]
        cc_valid = cc[valid]
        simp_valid = hull.simplices[valid]
        
        # Find the worst facet
        worst_idx = np.argmax(r_valid)
        worst_r = r_valid[worst_idx]
        if worst_r < tol:
            break
        
        # Move vertices of worst facet toward circumcenter
        c = cc_valid[worst_idx]
        step = min(0.1, worst_r / 10.0)
        for j in range(3):
            idx = simp_valid[worst_idx, j]
            pts[idx] = pts[idx] + step * (c - pts[idx])
        pts = _normalize(pts)
    
    return pts

def _nelder_mead_polish(pts, max_time_remaining, max_iter=200):
    """Nelder-Mead polish on the covering radius (radians)."""
    start_t = time.time()
    
    def obj(x_flat):
        elapsed = time.time() - start_t
        if elapsed > max_time_remaining - 2.0:
            raise RuntimeError("timeout")
        p = x_flat.reshape(N, 3)
        p = _normalize(p)
        return _covering_radius_rad(p)
    
    try:
        result = minimize(
            obj,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        pts_opt = result.x.reshape(N, 3)
        pts_opt = _normalize(pts_opt)
        return pts_opt
    except Exception:
        return pts

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin before 150s kill

    def _time_left():
        return max_time - (time.time() - start_time)

    best_points = _fibonacci_spiral()
    best_rad = _covering_radius_deg(best_points)

    # Phase 0: Multiple initializations
    inits = []
    inits.append(_fibonacci_spiral(0.0))
    inits.append(_fibonacci_spiral(0.3))
    inits.append(_fibonacci_spiral(0.7))
    rng = np.random.RandomState(42)
    for _ in range(3):
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)
    inits.append(_electrostatic_refine(_fibonacci_spiral(0.0), n_iters=300, lr=0.005))

    # Evaluate all initializations
    init_list = []
    for init in inits:
        if _time_left() < 5.0:
            break
        rad = _covering_radius_deg(init)
        init_list.append((rad, init.copy()))
        if rad < best_rad:
            best_rad = rad
            best_points = init.copy()

    # Sort by quality and take top 3
    init_list.sort(key=lambda x: x[0])
    top_inits = [x[1] for x in init_list[:3]]

    # Phase 1: Optimize each top initialization
    for init_idx, init_pts in enumerate(top_inits):
        if _time_left() < 20.0:
            break
        pts = init_pts.copy()
        cur_rad = _covering_radius_deg(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()

        # Phase 1a: L-BFGS-B on smooth surrogate
        if _time_left() > 25.0:
            pts = _lgbfgs_optimize(pts, max_time_remaining=_time_left(), beta=15.0, max_iter=80)
            rad = _covering_radius_deg(pts)
            if rad < best_rad:
                best_rad = rad
                best_points = pts.copy()

        # Phase 1b: Polish worst facets
        if _time_left() > 15.0:
            pts = _polish_worst_facets(pts, max_iters=60, tol=1e-9)
            rad = _covering_radius_deg(pts)
            if rad < best_rad:
                best_rad = rad
                best_points = pts.copy()

        # Phase 1c: Nelder-Mead polish
        if _time_left() > 10.0:
            pts = _nelder_mead_polish(pts, max_time_remaining=_time_left(), max_iter=100)
            rad = _covering_radius_deg(pts)
            if rad < best_rad:
                best_rad = rad
                best_points = pts.copy()

        # Phase 1d: Final high-precision polish
        if _time_left() > 5.0:
            pts = _polish_worst_facets(pts, max_iters=100, tol=1e-10)
            rad = _covering_radius_deg(pts)
            if rad < best_rad:
                best_rad = rad
                best_points = pts.copy()

    # Final verification
    final_rad = _covering_radius_deg(best_points)
    if final_rad < best_rad:
        best_rad = final_rad

    return best_points
