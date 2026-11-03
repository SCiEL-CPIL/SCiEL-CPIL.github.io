import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _tammes_soft_min(pts, beta=50.0):
    """Compute soft-min of pairwise angular distances using log-sum-exp.
    Returns (soft_min_value, gradient) where gradient points in direction to increase min distance."""
    D = pts @ pts.T
    dots = np.clip(D, -1.0, 1.0)
    ang = np.arccos(dots)
    # Zero out diagonal
    np.fill_diagonal(ang, 0.0)
    # Soft-min: -log-sum-exp(-beta * ang) / beta
    neg_beta_ang = -beta * ang
    max_val = np.max(neg_beta_ang, axis=1)
    log_sum = np.log(np.exp(neg_beta_ang - max_val[:, None]).sum(axis=1)) + max_val
    soft_min_vals = -log_sum / beta
    # Gradient of soft_min w.r.t. pts[i]
    # d(soft_min_i)/d(pts[i]) = -1/beta * d(log_sum)/d(pts[i])
    # log_sum = log(sum_j exp(-beta*ang_ij)) + max_val
    # exps = exp(-beta*ang_ij - max_val_i)
    exps = np.exp(neg_beta_ang - max_val[:, None])
    # d ang_ij / d pts[i] = -1/sin(ang_ij) * (pts[i] - pts[j]) projected to tangent at pts[i]
    # For soft-min, we want to increase the minimum, so gradient direction is toward increasing all distances
    # But the soft-min gradient: d(-logsumexp(-beta*ang))/d pts[i]
    # = (1/beta) * sum_j exp(-beta*ang_ij - max) * (-beta) * d(ang_ij)/d pts[i]
    # = -sum_j exps_ij * d(ang_ij)/d pts[i]
    # d(ang_ij)/d pts[i] = -(pts[i]-pts[j])/sin(ang_ij) projected to tangent
    # So gradient = sum_j exps_ij * (pts[i]-pts[j])/sin(ang_ij) projected
    sin_ang = np.sin(ang)
    sin_ang = np.maximum(sin_ang, 1e-10)
    np.fill_diagonal(sin_ang, 1.0)
    # Weight matrix: exps_ij / sin_ang_ij
    W = exps / sin_ang
    np.fill_diagonal(W, 0.0)
    # grad_i = sum_j W_ij * (pts[i] - pts[j]) = pts[i]*sum_j W_ij - W @ pts
    row_sum = W.sum(axis=1)
    grad = pts * row_sum[:, None] - (W @ pts)
    # Project to tangent
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    return soft_min_vals, grad

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

def _tammes_refine(pts, n_iters=200, beta_start=50.0, beta_end=200.0, lr=0.005):
    """Refine points to maximize minimum pairwise distance using soft-min gradient."""
    pts = pts.copy()
    for it in range(n_iters):
        beta = beta_start + (beta_end - beta_start) * (it / max(n_iters - 1, 1))
        _, grad = _tammes_soft_min(pts, beta=beta)
        pts = pts + lr * grad
        pts = _normalize(pts)
    return pts

def _electrostatic_refine(pts, n_iters=500, lr=0.01):
    """Repulsion-based refinement: each point is pushed away by all others via Coulomb-like force."""
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
        forces /= (np.linalg.norm(forces, axis=1, keepdims=True) + 1e-12)
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts

def _smooth_surrogate(pts, beta=8.0):
    """Smooth surrogate: log-sum-exp of squared facet radii (soft-max).
    Returns (surrogate_value, softmax_weights, radii, simplices, circumcenters, mask)."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    radii, cc, mask = _facet_circumradii(pts, hull.simplices)
    if not np.any(mask):
        return None
    # Use only valid facets
    valid = mask
    r_valid = radii[valid]
    cc_valid = cc[valid]
    simp_valid = hull.simplices[valid]
    # Soft-max of squared radii
    r2 = r_valid * r_valid
    m = np.max(r2)
    exps = np.exp((r2 - m) * beta)
    s = np.sum(exps)
    if s < 1e-300:
        weights = np.ones_like(exps) / len(exps)
    else:
        weights = exps / s
    surrogate = m + np.log(s) / beta
    return surrogate, weights, r_valid, cc_valid, simp_valid

def _surrogate_gradient(pts, weights, simplices, circumcenters, beta=8.0):
    """Gradient of smooth surrogate w.r.t. each point (in R^3, then projected to tangent).
    For a facet with circumcenter c and radius r, d(r^2)/d(vertex v) direction:
    moving v toward c decreases r. Gradient of r^2 w.r.t. v (tangent) ~ -2 * (c - v)_tangent / r ... 
    Simpler: use finite-difference-free analytic: the circumradius r satisfies cos(r) = c . a.
    d r / d a = -1/sin(r) * (c - (c.a) a) projected... 
    We use a simpler gradient: for each facet, push its 3 vertices toward the circumcenter,
    weighted by the softmax weight and by 1/sin(r) (steeper when r is small)."""
    grad = np.zeros_like(pts)
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-8:
            continue
        c = circumcenters[k]
        # radius r_k: cos(r) = c . a
        a = pts[simplices[k, 0]]
        cos_r = np.clip(np.dot(c, a), -1.0, 1.0)
        sin_r = np.sqrt(max(1.0 - cos_r * cos_r, 1e-12))
        # Gradient magnitude factor: dr/da direction is -(c - cos_r * a)/sin_r (tangent direction)
        # For each of the 3 vertices, the contribution to d(r^2)/d v = 2 r * dr/dv
        # dr/dv along tangent = -(c - cos_r * v) / sin_r  (unit-ish direction)
        # We'll just push toward c with weight w * r / sin_r
        scale = w * (2.0 * cos_r) / sin_r  # d(r^2)/d(cos_r) = -2r, but we use direction to c
        # Actually simpler: gradient of r^2 w.r.t. v (tangent) is proportional to (v - c) projected,
        # with magnitude 2r/sin(r). We want to DECREASE r^2, so move v toward c.
        for j in range(3):
            idx = simplices[k, j]
            v = pts[idx]
            # Direction from v to c, projected to tangent at v
            d = c - v
            d_t = d - np.dot(d, v) * v
            d_t_norm = np.linalg.norm(d_t)
            if d_t_norm < 1e-10:
                continue
            d_t = d_t / d_t_norm
            # Magnitude: 2 * r / sin(r) * w
            r_k = np.arccos(cos_r)
            mag = 2.0 * r_k / sin_r * w
            grad[idx] -= mag * d_t  # negative because we want to decrease r^2
    # Project gradient to tangent plane
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    return grad

def construct():
    start_time = time.time()
    max_time = 140.0  # Safety margin before 150s kill

    def _time_left():
        return max_time - (time.time() - start_time)

    rng = np.random.RandomState(42)
    
    # Phase 0: Generate high-quality initializations using Tammes refinement
    inits = []
    
    # Fibonacci spirals with Tammes refinement
    for offset in [0.0, 0.3, 0.7, 1.1]:
        if _time_left() < 10.0:
            break
        fib = _fibonacci_spiral(offset)
        refined = _tammes_refine(fib, n_iters=150, beta_start=30.0, beta_end=150.0, lr=0.003)
        inits.append(refined)
    
    # Random initializations with Tammes refinement
    for seed in [42, 123, 456]:
        if _time_left() < 10.0:
            break
        rng_local = np.random.RandomState(seed)
        pts = rng_local.randn(N, 3)
        pts = _normalize(pts)
        refined = _tammes_refine(pts, n_iters=150, beta_start=30.0, beta_end=150.0, lr=0.003)
        inits.append(refined)
    
    # Electrostatic-refined from Fibonacci (keep as backup)
    if _time_left() > 10.0:
        inits.append(_electrostatic_refine(_fibonacci_spiral(0.0), n_iters=200, lr=0.005))
    
    # Evaluate all initializations
    best_points = inits[0] if inits else _fibonacci_spiral()
    best_rad = _covering_radius_deg(best_points)
    
    for init in inits[1:]:
        if _time_left() < 5.0:
            break
        rad = _covering_radius_deg(init)
        if rad < best_rad:
            best_rad = rad
            best_points = init.copy()
    
    # Phase 1: For each good initialization, do greedy Lloyd + gradient + polish
    init_list = []
    for init in inits:
        rad = _covering_radius_deg(init)
        init_list.append((rad, init.copy()))
    init_list.sort(key=lambda x: x[0])
    # Take top 3 initializations
    top_inits = [x[1] for x in init_list[:3]]
    
    for init_idx, init_pts in enumerate(top_inits):
        if _time_left() < 10.0:
            break
        pts = init_pts.copy()
        cur_rad = _covering_radius_deg(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()
        
        # Phase 1a: Greedy Lloyd (move worst facet vertices toward circumcenter)
        step = 0.02
        for it in range(400):
            if _time_left() < 5.0:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            radii, cc, mask = _facet_circumradii(pts, hull.simplices)
            if not np.any(mask):
                break
            # Find worst facet
            valid_idx = np.where(mask)[0]
            worst_local = valid_idx[np.argmax(radii[valid_idx])]
            worst_rad = radii[worst_local]
            if worst_rad < 0.1:  # ~5.7 degrees, good enough
                break
            simp = hull.simplices[worst_local]
            c = cc[worst_local]
            # Move vertices toward circumcenter
            for j in range(3):
                idx = simp[j]
                v = pts[idx]
                new_v = v + step * (c - v)
                new_v = new_v / np.linalg.norm(new_v)
                pts[idx] = new_v
            # Small random perturbation every 30 iters to escape local minima
            if it % 30 == 0:
                for _ in range(2):
                    idx = rng.randint(N)
                    tangent = rng.randn(3)
                    tangent -= np.dot(tangent, pts[idx]) * pts[idx]
                    tn = np.linalg.norm(tangent)
                    if tn > 1e-10:
                        tangent /= tn
                        new_v = pts[idx] + 0.01 * tangent
                        pts[idx] = new_v / np.linalg.norm(new_v)
            cur_rad = _covering_radius_deg(pts)
            if cur_rad < best_rad:
                best_rad = cur_rad
                best_points = pts.copy()
            # Decay step
            step *= 0.995
            if step < 0.001:
                step = 0.001
        
        # Phase 1b: Gradient descent on smooth surrogate
        if _time_left() < 5.0:
            continue
        lr = 0.01
        momentum = 0.5
        vel = np.zeros_like(pts)
        for it in range(300):
            if _time_left() < 5.0:
                break
            result = _smooth_surrogate(pts, beta=8.0)
            if result is None:
                break
            surrogate, weights, r_valid, cc_valid, simp_valid = result
            grad = _surrogate_gradient(pts, weights, simp_valid, cc_valid, beta=8.0)
            vel = momentum * vel - lr * grad
            pts = pts + vel
            pts = _normalize(pts)
            cur_rad = _covering_radius_deg(pts)
            if cur_rad < best_rad:
                best_rad = cur_rad
                best_points = pts.copy()
            # Decay learning rate
            lr *= 0.995
            if lr < 0.0005:
                lr = 0.0005
        
        # Phase 1c: Nelder-Mead style local polish on top-k worst facets
        if _time_left() < 5.0:
            continue
        for polish_iter in range(80):
            if _time_left() < 3.0:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            radii, cc, mask = _facet_circumradii(pts, hull.simplices)
            if not np.any(mask):
                break
            valid_idx = np.where(mask)[0]
            # Sort by radius descending, take top 3
            order = valid_idx[np.argsort(-radii[valid_idx])][:3]
            improved = False
            for worst_local in order:
                if _time_left() < 2.0:
                    break
                simp = hull.simplices[worst_local]
                c = cc[worst_local]
                for j in range(3):
                    idx = simp[j]
                    v = pts[idx]
                    # Try moving toward circumcenter with decreasing steps
                    for s in [0.02, 0.01, 0.005, 0.002]:
                        new_v = v + s * (c - v)
                        new_v = new_v / np.linalg.norm(new_v)
                        old_v = pts[idx].copy()
                        pts[idx] = new_v
                        new_rad = _covering_radius_deg(pts)
                        if new_rad < best_rad:
                            best_rad = new_rad
                            best_points = pts.copy()
                            improved = True
                        else:
                            pts[idx] = old_v
                    # Try random tangent perturbations
                    for _ in range(4):
                        tangent = rng.randn(3)
                        tangent -= np.dot(tangent, v) * v
                        tn = np.linalg.norm(tangent)
                        if tn < 1e-10:
                            continue
                        tangent /= tn
                        for s in [0.01, 0.005]:
                            new_v = v + s * tangent
                            new_v = new_v / np.linalg.norm(new_v)
                            old_v = pts[idx].copy()
                            pts[idx] = new_v
                            new_rad = _covering_radius_deg(pts)
                            if new_rad < best_rad:
                                best_rad = new_rad
                                best_points = pts.copy()
                                improved = True
                            else:
                                pts[idx] = old_v
            if not improved and polish_iter > 40:
                break
    
    # Final high-precision polish on best_points
    if _time_left() > 3.0:
        pts = best_points.copy()
        for it in range(100):
            if _time_left() < 2.0:
                break
            try:
                hull = ConvexHull(pts)
            except Exception:
                break
            radii, cc, mask = _facet_circumradii(pts, hull.simplices)
            if not np.any(mask):
                break
            valid_idx = np.where(mask)[0]
            worst_local = valid_idx[np.argmax(radii[valid_idx])]
            worst_rad = radii[worst_local]
            if worst_rad < 0.19:  # ~10.9 degrees
                break
            simp = hull.simplices[worst_local]
            c = cc[worst_local]
            for j in range(3):
                idx = simp[j]
                v = pts[idx]
                for s in [0.005, 0.002, 0.001]:
                    new_v = v + s * (c - v)
                    new_v = new_v / np.linalg.norm(new_v)
                    old_v = pts[idx].copy()
                    pts[idx] = new_v
                    new_rad = _covering_radius_deg(pts)
                    if new_rad < best_rad:
                        best_rad = new_rad
                        best_points = pts.copy()
                    else:
                        pts[idx] = old_v
    
    return _normalize(best_points)
