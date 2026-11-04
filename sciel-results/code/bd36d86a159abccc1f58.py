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

def _electrostatic_refine(pts, n_iters=1000, lr=0.01):
    """Repulsion-based refinement: each point is pushed away by all others via Coulomb-like force.
    Uses a simple gradient descent on the electrostatic potential energy."""
    pts = pts.copy()
    for it in range(n_iters):
        # Pairwise dot products
        D = pts @ pts.T
        # Force on i from j: proportional to (pts[i] - pts[j]) / |...|^2 but on sphere we use tangent component
        # Use inverse-square repulsion in chordal distance
        # chord^2 = 2 - 2*D
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        # Repulsion magnitude ~ 1/chord^2
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        # Force = sum_j W_ij * (pts[i] - pts[j]) projected to tangent
        # = pts[i] * sum_j W_ij - sum_j W_ij * pts[j]
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        # Project to tangent plane
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        # Normalize forces to prevent huge steps
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / (force_norms + 1e-12)
        
        # Adaptive learning rate: start higher, decay
        current_lr = lr * (0.99 ** (it / 100))
        pts = pts + current_lr * forces
        pts = _normalize(pts)
    return pts

def _smooth_surrogate_and_grad(pts, beta=20.0):
    """Smooth surrogate: log-sum-exp of squared facet radii (soft-max) with true gradient.
    Returns (surrogate_value, gradient) or (None, None) on failure."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None
    radii, cc, mask = _facet_circumradii(pts, hull.simplices)
    if not np.any(mask):
        return None, None
    valid = mask
    r_valid = radii[valid]
    cc_valid = cc[valid]
    simp_valid = hull.simplices[valid]
    # Soft-max of squared radii: f = (1/beta) * log(sum(exp(beta * r_k^2)))
    r2 = r_valid * r_valid
    m = np.max(r2)
    exps = np.exp((r2 - m) * beta)
    s = np.sum(exps)
    if s < 1e-300:
        weights = np.ones_like(exps) / len(exps)
    else:
        weights = exps / s
    surrogate = m + np.log(s) / beta
    
    # Compute gradient: d(surrogate)/d(pts[i]) = sum_k w_k * d(r_k^2)/d(pts[i])
    # r_k = arccos(c_k . v) for vertex v of facet k
    # d(r_k^2)/d(v) = -2 * r_k / sin(r_k) * c_k (projected to tangent at v)
    grad = np.zeros_like(pts)
    for k in range(len(simp_valid)):
        w = weights[k]
        if w < 1e-10:
            continue
        c_k = cc_valid[k]
        r_k = r_valid[k]
        if r_k < 1e-12:
            continue
        sin_r = np.sin(r_k)
        if sin_r < 1e-12:
            continue
        coeff = -2.0 * r_k / sin_r * w
        for j in range(3):
            idx = simp_valid[k, j]
            v = pts[idx]
            # Project c_k onto tangent plane at v
            c_t = c_k - np.dot(c_k, v) * v
            grad[idx] += coeff * c_t
    # Project gradient to tangent plane for each point
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    return surrogate, grad

def _combined_objective_and_grad(pts_flat, beta=20.0):
    """Combined objective: smooth covering radius + small electrostatic term.
    Returns (value, gradient) for use with L-BFGS-B."""
    pts = pts_flat.reshape(N, 3)
    pts = _normalize(pts)
    
    # Get covering radius gradient
    surrogate, grad_cov = _smooth_surrogate_and_grad(pts, beta=beta)
    if surrogate is None:
        return 1e6, np.zeros_like(pts_flat)
    
    # Add small electrostatic repulsion to help with local minima
    D = pts @ pts.T
    chord2 = 2.0 - 2.0 * D
    np.fill_diagonal(chord2, 1.0)
    W = 1.0 / chord2
    np.fill_diagonal(W, 0.0)
    row_sum = W.sum(axis=1)
    forces = pts * row_sum[:, None] - (W @ pts)
    forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
    grad_elec = forces
    
    # Blend: primarily covering radius, small electrostatic component
    alpha = 0.1
    grad_total = (1 - alpha) * grad_cov + alpha * grad_elec
    
    value = surrogate
    return value, grad_total.flatten()

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin before 150s kill

    def _time_left():
        return max_time - (time.time() - start_time)

    best_points = _fibonacci_spiral()
    best_rad = _covering_radius_deg(best_points)

    # Phase 0: Multiple initializations with strong electrostatic refinement
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Fibonacci + Electrostatic
    fib1 = _fibonacci_spiral(0.0)
    inits.append(_electrostatic_refine(fib1, n_iters=500, lr=0.02))
    
    # 2. Fibonacci (offset) + Electrostatic
    fib2 = _fibonacci_spiral(0.5)
    inits.append(_electrostatic_refine(fib2, n_iters=500, lr=0.02))
    
    # 3. Random + Electrostatic
    rand1 = _normalize(rng.randn(N, 3))
    inits.append(_electrostatic_refine(rand1, n_iters=800, lr=0.02))
    
    # 4. Raw Fibonacci (fallback)
    inits.append(fib1)

    # Evaluate all initializations
    for init in inits:
        if _time_left() < 5.0:
            break
        rad = _covering_radius_deg(init)
        if rad < best_rad:
            best_rad = rad
            best_points = init.copy()

    # Phase 1: L-BFGS-B optimization with smooth surrogate
    for beta_val in [30.0, 50.0, 100.0]:
        if _time_left() < 20.0:
            break
        pts = best_points.copy()
        cur_rad = _covering_radius_deg(pts)
        
        # Use L-BFGS-B on the flattened parameter space
        x0 = pts.flatten()
        res = minimize(
            _combined_objective_and_grad,
            x0,
            args=(beta_val,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if res.success or res.fun < 1e5:
            new_pts = _normalize(res.x.reshape(N, 3))
            new_rad = _covering_radius_deg(new_pts)
            if new_rad < best_rad:
                best_rad = new_rad
                best_points = new_pts.copy()

    # Phase 2: Greedy Lloyd + Polish on the best points
    pts = best_points.copy()
    cur_rad = _covering_radius_deg(pts)
    
    # Process top-3 worst facets simultaneously
    step = 0.015
    for it in range(150):
        if _time_left() < 8.0:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        radii, cc, mask = _facet_circumradii(pts, hull.simplices)
        if not np.any(mask):
            break
        valid_idx = np.where(mask)[0]
        # Get top 3 worst facets
        top_k = min(3, len(valid_idx))
        worst_locals = valid_idx[np.argsort(-radii[valid_idx])][:top_k]
        worst_rad = radii[worst_locals[0]]
        if worst_rad < 0.14:  # ~8.0 degrees
            break
        
        # Move vertices of all top facets toward their circumcenters
        moved = set()
        for worst_local in worst_locals:
            simp = hull.simplices[worst_local]
            c = cc[worst_local]
            for j in range(3):
                idx = simp[j]
                if idx in moved:
                    continue
                v = pts[idx]
                new_v = v + step * (c - v)
                new_v = new_v / np.linalg.norm(new_v)
                pts[idx] = new_v
                moved.add(idx)
        
        # Small random perturbation to escape local minima
        if it % 15 == 0:
            for _ in range(3):
                idx = rng.randint(N)
                tangent = rng.randn(3)
                tangent -= np.dot(tangent, pts[idx]) * pts[idx]
                tn = np.linalg.norm(tangent)
                if tn > 1e-10:
                    tangent /= tn
                    new_v = pts[idx] + 0.003 * tangent
                    pts[idx] = new_v / np.linalg.norm(new_v)
                    
        cur_rad = _covering_radius_deg(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()
        step *= 0.985

    # Phase 3: Final L-BFGS-B polish with high beta for precision
    if _time_left() > 15.0:
        pts = best_points.copy()
        for beta_val in [200.0, 500.0]:
            if _time_left() < 10.0:
                break
            x0 = pts.flatten()
            res = minimize(
                _combined_objective_and_grad,
                x0,
                args=(beta_val,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            if res.success or res.fun < 1e5:
                new_pts = _normalize(res.x.reshape(N, 3))
                new_rad = _covering_radius_deg(new_pts)
                if new_rad < best_rad:
                    best_rad = new_rad
                    best_points = new_pts.copy()
                    pts = new_pts.copy()

    return _normalize(best_points)
