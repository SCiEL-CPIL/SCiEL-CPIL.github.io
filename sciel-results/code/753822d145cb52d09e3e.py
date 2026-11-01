import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14
DEADLINE_FRAC = 145.0  # wall-clock guard (evaluator kills at 150s)

def _triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=np.int32)

_TRI = _triangle_indices(N)
_I = _TRI[:, 0]
_J = _TRI[:, 1]
_K = _TRI[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        return float(scipy.spatial.ConvexHull(points).volume)
    except Exception:
        return 0.0

def _hull_area_grad(points):
    """Gradient of convex hull area w.r.t. all points.
    Only hull vertices contribute to the gradient."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        n = len(points)
        grad = np.zeros_like(points)
        # Hull area = 0.5 * sum of cross products of consecutive hull vertices
        # For a convex polygon with vertices v_0, v_1, ..., v_{m-1} in order:
        # Area = 0.5 * sum_i (x_i * y_{i+1} - x_{i+1} * y_i)
        # d(Area)/d(x_i) = 0.5 * (y_{i-1} - y_{i+1})
        # d(Area)/d(y_i) = 0.5 * (x_{i+1} - x_{i-1})
        hv = hull.vertices
        m = len(hv)
        for i in range(m):
            idx = hv[i]
            prev_idx = hv[(i - 1) % m]
            next_idx = hv[(i + 1) % m]
            grad[idx, 0] += 0.5 * (points[prev_idx, 1] - points[next_idx, 1])
            grad[idx, 1] += 0.5 * (points[next_idx, 0] - points[prev_idx, 0])
        return grad
    except Exception:
        return np.zeros_like(points)

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        return _min_triangle_area(points) / ha
    except Exception:
        return 0.0

def _neg_score(flat):
    return -_score(flat.reshape(-1, 2))

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])

def _two_ring_config(n, frac=0.5, phase=0.5):
    n_outer = n - 7
    n_inner = 7
    outer = _regular_polygon(n_outer, 1.0, 0.0)
    inner = _regular_polygon(n_inner, frac, phase * 2 * np.pi / n_inner)
    return np.vstack([outer, inner])

def _ellipse_config(n, a=1.0, b=0.8, rotation=0.0):
    """Points on an ellipse."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    pts = np.column_stack([x, y])
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T

def _two_ring_heptagon(r, phi):
    """
    Construct 14 points as two nested regular heptagons.
    Outer radius 1.0, inner radius r, relative rotation phi.
    """
    outer = _regular_polygon(7, 1.0, 0.0)
    inner = _regular_polygon(7, r, phi)
    return np.vstack([outer, inner])

def _build_structured_starts():
    starts = []
    # Regular polygon baseline (and a few rotations)
    for rot in [0.0, np.pi / N, np.pi / 14, np.pi / 3]:
        starts.append(_regular_polygon(N, 1.0, rot))
    # Two-ring "near-optimal" structure with varied inner radius and phase
    for frac in [0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for phase in [0.0, 0.5, 0.25, 0.75, 1.0/7]:
            starts.append(_two_ring_config(N, frac, phase))
    # Ellipse configurations with varied aspect ratios and rotations
    for a, b in [(1.0, 0.9), (1.0, 0.85), (1.0, 0.8), (1.0, 0.75), (1.05, 0.95), (1.0, 0.7)]:
        for rot in [0.0, np.pi / 6, np.pi / 4, np.pi / 3]:
            starts.append(_ellipse_config(N, a, b, rot))
    # Three-ring configuration
    for r1, r2 in [(0.3, 0.6), (0.35, 0.65), (0.4, 0.7)]:
        outer = _regular_polygon(5, 1.0, 0.0)
        mid = _regular_polygon(4, r2, np.pi/4)
        inner = _regular_polygon(5, r1, 0.0)
        starts.append(np.vstack([outer, mid, inner]))
    return starts

def _lse_min_area_and_grad(points, tau):
    """
    Computes the LSE approximation of min triangle area and its gradient.
    Returns (lse_value, grad_flat)
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product: v1 x v2 = v1_x * v2_y - v1_y * v2_x
    # Area = 0.5 * |cross|
    # To handle the absolute value smoothly, we assume we are in a region 
    # where the orientation is consistent or we use a soft-min on signed areas?
    # Standard LSE on absolute values is tricky for gradients.
    # However, for a convex polygon, all triangles formed by vertices have the same sign orientation 
    # if we order them correctly? No, any 3 points on a convex hull have the same orientation 
    # relative to the center, but the cross product sign depends on the order i,j,k.
    # We defined i<j<k. For points on a convex hull, the orientation of (i,j,k) is consistent 
    # if the points are ordered angularly. If the points are not in angular order, signs might flip.
    # But _min_triangle_area uses abs().
    
    # A better smooth approximation for min(|x_i|) is -tau * log(sum(exp(-|x_i|/tau))).
    # Gradient of |x| is sign(x).
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # LSE calculation
    max_a = np.max(areas)
    exp_terms = np.exp(-(areas - max_a) / tau)
    sum_exp = np.sum(exp_terms)
    lse_val = max_a + tau * np.log(sum_exp)
    
    # Gradient calculation
    # d/dx [-tau * log(sum(exp(-A_i/tau)))] = sum(exp(-A_i/tau) * (-1/tau) * dA_i/dx) / sum(exp(-A_i/tau))
    # Wait, the objective to minimize is -min_area.
    # min_area ~ -tau * log(sum(exp(-A_i/tau)))
    # So we minimize f(x) = -min_area ~ tau * log(sum(exp(-A_i/tau)))
    # grad f = sum( exp(-A_i/tau) * (-1/tau) * grad A_i ) / sum(exp(-A_i/tau))
    # grad f = - sum( w_i * grad A_i ) where w_i = exp(-A_i/tau) / sum_exp
    
    weights = exp_terms / sum_exp
    
    # grad A_i w.r.t. points
    # A_i = 0.5 * |cross_i|
    # dA_i/dp = 0.5 * sign(cross_i) * d(cross_i)/dp
    
    # d(cross)/dp_k (for point k in triangle i,j,k):
    # cross = (p_j - p_i) x (p_k - p_i)
    # d(cross)/dp_i = (p_j - p_i) x (p_k - p_i) ... wait.
    # Let u = p_j - p_i, v = p_k - p_i. cross = u_x v_y - u_y v_x.
    # d(cross)/dp_i = - (d(cross)/dp_j + d(cross)/dp_k)? No.
    # d(cross)/dp_j = (1, 0) x v - ... ?
    # Simpler:
    # d(cross)/dp_i = - (v1 x v2 derivative w.r.t p_i)
    # Actually, the gradient of the cross product w.r.t. the three vertices is:
    # grad_i = (v2_y, -v2_x)  ? No.
    # Let's derive:
    # C = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
    # dC/dx_i = -(y_k-y_i) + (y_j-y_i)(0) ... no.
    # dC/dx_i = -(y_k-y_i) - (y_j-y_i)*0 + ... 
    # dC/dx_i = -(y_k-y_i) + (y_j-y_i) * (d(x_k-x_i)/dx_i? no)
    # Let's use the property that grad C = (v2_y, -v2_x) for p_i?
    # Check: p_i = (0,0), p_j = (1,0), p_k = (0,1). C = 1.
    # Move p_i to (eps, 0). p_j-p_i = (1-eps, 0). p_k-p_i = (-eps, 1).
    # C = (1-eps)*1 - 0*(-eps) = 1-eps. dC/dx_i = -1.
    # Formula v2_y, -v2_x: v2 = p_k-p_i = (0,1). v2_y=1, -v2_x=0. (1,0). Mismatch.
    
    # Correct gradient of Cross Product C = u x v where u=p_j-p_i, v=p_k-p_i:
    # dC/dp_i = - (u x v)' w.r.t p_i?
    # dC/dp_i = (v_y, -v_x) ? No.
    # dC/dp_i = (-(y_k-y_i), (x_k-x_i))? 
    # Let's check p_i=(0,0), p_j=(1,0), p_k=(0,1).
    # v = (0,1). -v_y = -1, v_x = 0. So (-1, 0). Matches dC/dx_i = -1.
    # So grad_i = (-v2_y, v2_x) ? No, (-v_y, v_x).
    # Let's check p_j. p_j=(1,0) -> (1+eps, 0). u=(1+eps,0). v=(0,1). C=1+eps. dC/dx_j = 1.
    # Formula for p_j: (v_y, -v_x) = (1, 0). Matches.
    # Let's check p_k. p_k=(0,1) -> (0, 1+eps). v=(0,1+eps). u=(1,0). C=1+eps. dC/dy_k = 1.
    # Formula for p_k: (-u_y, u_x) = (0, 1). Matches.
    
    # So:
    # grad C w.r.t p_i = (-v2_y, v2_x)
    # grad C w.r.t p_j = (v2_y, -v2_x)
    # grad C w.r.t p_k = (-v1_y, v1_x)
    
    # A = 0.5 * |C|. grad A = 0.5 * sign(C) * grad C.
    
    grad_flat = np.zeros_like(points.flatten())
    
    # Vectorized gradient accumulation
    signs = np.sign(cross)
    if np.any(signs == 0):
        signs[signs == 0] = 1.0 # Avoid zero gradient, arbitrary direction
        
    coeff = 0.5 * signs * weights
    
    # Accumulate for p_i
    gi = coeff[:, np.newaxis] * np.column_stack([-v2[:, 1], v2[:, 0]])
    np.add.at(grad_flat, _I * 2 + 0, gi[:, 0])
    np.add.at(grad_flat, _I * 2 + 1, gi[:, 1])
    
    # Accumulate for p_j
    gj = coeff[:, np.newaxis] * np.column_stack([v2[:, 1], -v2[:, 0]])
    np.add.at(grad_flat, _J * 2 + 0, gj[:, 0])
    np.add.at(grad_flat, _J * 2 + 1, gj[:, 1])
    
    # Accumulate for p_k
    gk = coeff[:, np.newaxis] * np.column_stack([-v1[:, 1], v1[:, 0]])
    np.add.at(grad_flat, _K * 2 + 0, gk[:, 0])
    np.add.at(grad_flat, _K * 2 + 1, gk[:, 1])
    
    return lse_val, grad_flat

def _smooth_neg_score(flat, tau):
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    lse_val, _ = _lse_min_area_and_grad(pts, tau)
    return -lse_val / ha

def _smooth_neg_score_grad(flat, tau):
    pts = flat.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10, np.zeros_like(flat)
    lse_val, grad_lse = _lse_min_area_and_grad(pts, tau)
    # f = -lse_val / ha
    # grad f = -(grad_lse * ha - lse_val * grad_ha) / ha^2
    #       = -grad_lse / ha + lse_val * grad_ha / ha^2
    grad_ha = _hull_area_grad(pts)
    grad_f = -grad_lse / ha + (lse_val / (ha * ha)) * grad_ha
    return -lse_val / ha, grad_f.flatten()

def construct():
    t0 = time.time()
    def left():
        return DEADLINE_FRAC - (time.time() - t0)

    best_score = 0.0
    best_pts = _normalize(_regular_polygon(N, 1.0, 0.0))

    # 1. High-resolution search over the symmetric 2-ring manifold
    # This is cheap and likely to find a very good local optimum
    if left() > 10.0:
        r_grid = np.linspace(0.3, 0.8, 50)
        phi_grid = np.linspace(0, np.pi/7, 50)
        
        best_sym_score = 0.0
        best_r = 0.5
        best_phi = 0.0
        
        for r in r_grid:
            for phi in phi_grid:
                pts = _two_ring_heptagon(r, phi)
                sc = _score(pts)
                if sc > best_sym_score:
                    best_sym_score = sc
                    best_r = r
                    best_phi = phi
        
        # Refine (r, phi) using Nelder-Mead on the 2D space
        if best_sym_score > 0:
            def obj_sym(params):
                r, phi = params
                if r < 0.1 or r > 1.5:
                    return 1e10
                pts = _two_ring_heptagon(r, phi)
                return -_score(pts)
            
            res_sym = scipy.optimize.minimize(obj_sym, np.array([best_r, best_phi]), method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000})
            if res_sym.success and res_sym.fun < -best_sym_score:
                r_opt, phi_opt = res_sym.x
                best_sym_pts = _two_ring_heptagon(r_opt, phi_opt)
                best_sym_score = _score(best_sym_pts)
            
            if best_sym_score > best_score:
                best_score = best_sym_score
                best_pts = best_sym_pts.copy()

    # 2. General optimization starting from the best symmetric point
    # This allows breaking symmetry if it helps
    def try_opt(x0, method, maxiter, fatol, xatol):
        nonlocal best_score, best_pts
        if left() <= 5.0:
            return
        try:
            opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                    'adaptive': True, 'maxfev': 5000}
            res = scipy.optimize.minimize(_neg_score, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score = sc
                best_pts = opt.copy()
        except Exception:
            pass

    # Run general optimization on the best symmetric start
    try_opt(best_pts, 'Nelder-Mead', 10000, 1e-12, 1e-10)
    
    if left() > 10.0:
        try_opt(best_pts, 'Powell', 5000, 1e-13, 1e-11)
        
    # Final polish
    if left() > 5.0:
        try_opt(best_pts, 'Nelder-Mead', 5000, 1e-14, 1e-12)

    return _normalize(best_pts)
