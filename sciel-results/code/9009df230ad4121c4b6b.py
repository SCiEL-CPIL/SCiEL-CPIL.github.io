import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

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
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _hull_area_and_grad(points):
    """
    Computes the area of the convex hull and its gradient with respect to points.
    Returns (area, grad) where grad is (N, 2).
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
        area = hull.volume
        hull_idx = hull.vertices
        n_hull = len(hull_idx)
        
        grad = np.zeros_like(points)
        
        if n_hull < 3:
            return area, grad
            
        # For each hull vertex, find its neighbors in the hull cycle
        # hull_idx is in CCW order
        for i in range(n_hull):
            idx_curr = hull_idx[i]
            idx_prev = hull_idx[(i - 1) % n_hull]
            idx_next = hull_idx[(i + 1) % n_hull]
            
            p_curr = points[idx_curr]
            p_prev = points[idx_prev]
            p_next = points[idx_next]
            
            # Contribution to area from edges (prev, curr) and (curr, next)
            # Area = 0.5 * sum (x_i y_{i+1} - x_{i+1} y_i)
            # Terms involving p_curr:
            # 0.5 * (x_prev * y_curr - y_prev * x_curr) + 0.5 * (x_curr * y_next - y_curr * x_next)
            # d/dx_curr: 0.5 * (-y_prev + y_next)
            # d/dy_curr: 0.5 * (x_prev - x_next)
            
            grad[idx_curr, 0] = 0.5 * (p_next[1] - p_prev[1])
            grad[idx_curr, 1] = 0.5 * (p_prev[0] - p_next[0])
            
        return area, grad
    except Exception:
        return 0.0, np.zeros_like(points)

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2):
    """
    Generate points on two concentric circles.
    n1, n2: number of points on outer and inner ring.
    r1, r2: radii.
    rot1, rot2: rotations.
    """
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    return np.vstack([pts1, pts2])

def _smooth_min_triangle_area_and_grad(points, beta=100.0):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    Returns (value, gradient).
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # For numerical stability, shift by max
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points)
    
    # Stable log-sum-exp: log(sum(exp(-beta * areas))) = -beta*max_area + log(sum(exp(-beta*(areas - max_area))))
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    log_sum = -beta * max_area + np.log(np.sum(exp_vals))
    smooth_min = -log_sum / beta
    
    # Gradient of smooth_min
    # S = -1/beta * log(sum(exp(-beta * a_i)))
    # dS/da_i = (exp(-beta * a_i) / sum(exp(-beta * a_j)))
    # Let w_i = exp(-beta * (a_i - max_area)) / sum(exp(-beta * (a_j - max_area)))
    # Then dS/da_i = w_i
    # Note: The shift by max_area cancels out in the ratio.
    
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Now we need d(areas)/d(points)
    # areas = 0.5 * |cross|
    # cross = (pj - pi) x (pk - pi)
    # Let's compute gradient of cross first, then chain rule with abs.
    # Actually, since we are using smooth min, we assume areas > 0.
    # d(0.5 * |cross|) = 0.5 * sign(cross) * d(cross)
    
    # d(cross) / d(pi) = - ( (pk-pi) x 1 + 1 x (pj-pi) )? No.
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # Let v1 = pj - pi, v2 = pk - pi. cross = v1_x v2_y - v1_y v2_x.
    # d(cross)/d(pi) = -d(v1)/d(pi) x v2 - v1 x d(v2)/d(pi)
    # d(v1)/d(pi) = -I, d(v2)/d(pi) = -I.
    # d(cross)/d(pi) = -(-I) x v2 - v1 x (-I) = I x v2 + v1 x I?
    # Gradient of scalar f(v1, v2) w.r.t pi:
    # df/dpi = -df/dv1 - df/dv2
    # df/dv1 = (v2_y, -v2_x)
    # df/dv2 = (-v1_y, v1_x)
    # So df/dpi = -(v2_y, -v2_x) - (-v1_y, v1_x) = (-v2_y + v1_y, v2_x - v1_x)
    
    # df/dpj = df/dv1 = (v2_y, -v2_x)
    # df/dpk = df/dv2 = (-v1_y, v1_x)
    
    grad_cross_pi = np.column_stack([v2[:, 1] - v1[:, 1], v1[:, 0] - v2[:, 0]]) # Wait, check signs.
    # Let's re-verify.
    # f = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    # d/dxi = -(yk-yi) + (yj-yi) = yj - yk
    # d/dyi = -(xk-xi) + (xj-xi) = xj - xk
    # So grad_pi = (yj - yk, xj - xk)
    # My previous derivation: (-v2_y + v1_y, v2_x - v1_x)
    # v1_y = yj-yi, v2_y = yk-yi. -v2_y + v1_y = -(yk-yi) + (yj-yi) = yj - yk. Correct.
    # v1_x = xj-xi, v2_x = xk-xi. v2_x - v1_x = (xk-xi) - (xj-xi) = xk - xj.
    # Wait, d/dyi = xj - xk. My formula gave xk - xj.
    # Let's re-evaluate d/dyi.
    # f = ... - (yj-yi)(xk-xi)
    # d/dyi [ - (yj-yi)(xk-xi) ] = - (xk-xi)
    # f = (xj-xi)(yk-yi) ...
    # d/dyi [ (xj-xi)(yk-yi) ] = (xj-xi)
    # Sum: (xj-xi) - (xk-xi) = xj - xk.
    # Formula: v2_x - v1_x = (xk-xi) - (xj-xi) = xk - xj.
    # Sign error in formula for y component.
    # Let's stick to explicit calculation.
    
    # grad_pi = (yj - yk, xj - xk)
    # grad_pj = (yk - yi, -(xk - xi)) = (v2_y, -v2_x)
    # grad_pk = (-(yj - yi), xj - xi) = (-v1_y, v1_x)
    
    g_pi = np.column_stack([points[_J, 1] - points[_K, 1], points[_J, 0] - points[_K, 0]])
    g_pj = np.column_stack([v2[:, 1], -v2[:, 0]])
    g_pk = np.column_stack([-v1[:, 1], v1[:, 0]])
    
    # Now, areas = 0.5 * |cross|
    # d(area_i)/d(points) = 0.5 * sign(cross_i) * grad(cross_i)
    
    signs = np.sign(cross)
    signs[signs == 0] = 1.0 # Avoid division by zero or undefined sign, though area is 0
    
    # Accumulate gradient
    grad_points = np.zeros_like(points)
    
    # We need to sum weights_i * grad(area_i)
    # grad(area_i) is a (3, 2) matrix? No, it's distributed to pi, pj, pk.
    # For each triangle t, we add:
    # weights[t] * 0.5 * signs[t] * g_pi[t] to grad_points[pi_idx]
    # weights[t] * 0.5 * signs[t] * g_pj[t] to grad_points[pj_idx]
    # weights[t] * 0.5 * signs[t] * g_pk[t] to grad_points[pk_idx]
    
    coeff = 0.5 * weights * signs
    
    np.add.at(grad_points, _I, coeff[:, np.newaxis] * g_pi)
    np.add.at(grad_points, _J, coeff[:, np.newaxis] * g_pj)
    np.add.at(grad_points, _K, coeff[:, np.newaxis] * g_pk)
    
    return smooth_min, grad_points

def _smooth_score_and_grad(points, beta=100.0):
    """
    Smooth version of the score: smooth_min_triangle_area / hull_area
    Returns (score, gradient).
    """
    try:
        if len(points) < 3:
            return 0.0, np.zeros_like(points)
        
        ha, grad_ha = _hull_area_and_grad(points)
        if ha < 1e-12:
            return 0.0, np.zeros_like(points)
            
        smta, grad_smta = _smooth_min_triangle_area_and_grad(points, beta)
        
        score = smta / ha
        
        # d(S/A) = (dS*A - S*dA) / A^2
        grad_score = (grad_smta * ha - score * grad_ha) / (ha * ha)
        
        return score, grad_score
    except Exception:
        return 0.0, np.zeros_like(points)

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Returns (-smooth_score, gradient) for L-BFGS-B.
    """
    points = points_flat.reshape(-1, 2)
    score, grad = _smooth_score_and_grad(points, beta)
    return -score, -grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    def _try_optimize_smooth(init_points, beta_start=50.0, beta_end=500.0, maxiter=2000):
        """
        Optimize using smooth surrogate with annealed beta.
        Start with low beta (smoother) and increase to high beta (sharper).
        """
        if time.time() > deadline:
            return
        init_norm = _normalize(init_points)
        x0 = init_norm.flatten()
        
        # Anneal beta from beta_start to beta_end
        n_stages = 5
        betas = np.linspace(beta_start, beta_end, n_stages)
        
        x = x0.copy()
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = scipy.optimize.minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter // n_stages, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final polish with exact (non-smooth) score using Nelder-Mead
        if time.time() < deadline - 3:
            try:
                def _neg_exact_score(x_flat):
                    return -_score(x_flat.reshape(-1, 2))
                
                res = scipy.optimize.minimize(
                    _neg_exact_score,
                    x,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    def _try_optimize_points(init_points, maxiter=3000):
        """Optimize the 2N coordinates directly using smooth surrogate."""
        _try_optimize_smooth(init_points, beta_start=50.0, beta_end=500.0, maxiter=maxiter)

    # 1. Explore the two-ring family analytically
    # Possible splits for N=14: (1,13), (2,12), (3,11), (4,10), (5,9), (6,8), (7,7)
    # (7,7) is symmetric and a very strong candidate.
    # (6,8) and (8,6) are also good.
    
    splits = [(7, 7), (6, 8), (8, 6), (5, 9), (9, 5), (4, 10), (10, 4)]
    
    for n1, n2 in splits:
        if time.time() > deadline:
            break
            
        # Optimize the parameters of the two-ring configuration
        # Params: r1 (normalized to 1), r2/r1, rot1, rot2
        # We fix r1=1 for optimization and scale at the end.
        
        def _two_ring_score(params):
            r_ratio = params[0]
            rot1 = params[1]
            rot2 = params[2]
            
            r1 = 1.0
            r2 = r_ratio * r1
            
            # Ensure r2 > 0
            if r2 <= 0.01:
                return -1e6
                
            pts = _generate_two_ring_config(n1, n2, r1, r2, rot1, rot2)
            # Normalize and score
            sc = _score(_normalize(pts))
            return -sc

        # Initial guesses for r_ratio, rot1, rot2
        # r_ratio typically between 0.4 and 0.8
        # rot1 = 0 is WLOG due to rotational symmetry of the whole set? No, relative rotation matters.
        # We can fix rot1=0 and optimize rot2 relative to it? 
        # Actually, the score is invariant under global rotation. So we can fix rot1=0.
        # But the optimizer might struggle if we don't allow it. Let's just optimize all 3.
        
        # Try a few different starting points for the ratio
        for r_init in [0.4, 0.5, 0.6, 0.7]:
            for rot_init in [0.0, np.pi/n1, np.pi/n2]:
                if time.time() > deadline:
                    continue
                x0 = np.array([r_init, 0.0, rot_init])
                try:
                    res = scipy.optimize.minimize(
                        _two_ring_score,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    r_ratio, rot1, rot2 = res.x
                    pts = _generate_two_ring_config(n1, n2, 1.0, r_ratio, rot1, rot2)
                    sc = _update_best(pts)
                    
                    # If this configuration is good, refine it by optimizing all 2N coordinates
                    if sc > best_score * 0.95 and time.time() < deadline - 15:
                        _try_optimize_points(pts, maxiter=2000)
                        
                except Exception:
                    pass

    # 2. Explore regular polygon and slight perturbations
    # Regular 14-gon
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    # Perturb regular polygon and optimize with smooth surrogate
    rng = np.random.RandomState(42)
    for i in range(15):
        if time.time() > deadline - 10:
            break
        noise = rng.uniform(-0.1, 0.1, (N, 2))
        init = _regular_polygon(N, 1.0) + noise
        _try_optimize_points(init, maxiter=1500)

    # 3. Random restarts with optimization
    # Generate random points and optimize
    for i in range(8):
        if time.time() > deadline - 10:
            break
        # Generate points on a disk, then optimize
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        init = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        _try_optimize_points(init, maxiter=1000)

    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
