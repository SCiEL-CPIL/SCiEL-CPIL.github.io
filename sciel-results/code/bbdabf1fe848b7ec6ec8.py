"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

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
_N_TRI = len(_TRI_INDICES)

def _min_triangle_area(points):
    """Compute the exact minimum triangle area."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the exact score: min triangle area / hull area."""
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

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Compute the negative of the smooth score and its gradient.
    Score = SmoothMin(TriangleAreas) / HullArea.
    We approximate grad(Score) ~ grad(SmoothMin) / HullArea.
    """
    points = points_flat.reshape(-1, 2)
    
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # If all areas are 0, return 0
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points_flat)
        
    # Smooth minimum calculation
    # M_beta = -1/beta * log(sum(exp(-beta * areas)))
    # To avoid underflow, shift by max_area
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
        
    weights = exp_vals / sum_exp
    log_sum = -beta * max_area + np.log(sum_exp)
    smooth_min = -log_sum / beta
    
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0, np.zeros_like(points_flat)
        
    score = smooth_min / ha
    
    # Gradient of smooth_min w.r.t points
    # d(smooth_min)/d(points) = sum(weights * signs * 0.5 * d(cross)/d(points))
    # cross = (pj_x - pi_x)(pk_y - pi_y) - (pj_y - pi_y)(pk_x - pi_x)
    
    signs = np.sign(cross)
    
    # Derivatives of cross w.r.t pi, pj, pk
    # cross = pj_x*pk_y - pj_x*pi_y - pi_x*pk_y + pi_x*pi_y - pj_y*pk_x + pj_y*pi_x + pi_y*pk_x - pi_y*pi_x
    # d(cross)/d(pi_x) = -pk_y + pi_y + pj_y - pi_y = pj_y - pk_y = -v2_y
    # d(cross)/d(pi_y) = -pj_x + pi_x + pk_x - pi_x = pk_x - pj_x = -v1_x
    # d(cross)/d(pj_x) = pk_y - pi_y = v2_y
    # d(cross)/d(pj_y) = -pk_x + pi_x = -v1_x
    # d(cross)/d(pk_x) = -pj_y + pi_y = -v1_y
    # d(cross)/d(pk_y) = pj_x - pi_x = v1_x
    
    # Let's re-verify with vector calculus:
    # A = 0.5 |v1 x v2|
    # If cross > 0, dA/dpi = -0.5 (v1_y, -v1_x) ... wait.
    # Let's use the explicit partials derived above.
    
    d_cross_di_x = -(v1[:, 1] + v2[:, 1]) # Wait, v1 = pj-pi, v2 = pk-pi
    # cross = (pj_x - pi_x)(pk_y - pi_y) - (pj_y - pi_y)(pk_x - pi_x)
    # d/dpi_x: -(pk_y - pi_y) - (pj_y - pi_y) = -pk_y + pi_y - pj_y + pi_y = 2*pi_y - pj_y - pk_y
    # This matches -(v1_y + v2_y) because v1_y = pj_y - pi_y, v2_y = pk_y - pi_y.
    # Sum = pj_y + pk_y - 2*pi_y. Negative is 2*pi_y - pj_y - pk_y. Correct.
    
    d_cross_di_y = v1[:, 0] + v2[:, 0] # d/dpi_y: (pj_x - pi_x) + (pk_x - pi_x) = 2*pi_x - pj_x - pk_x? No.
    # Term 1: (pj_x - pi_x)(pk_y - pi_y). d/dpi_y -> (pj_x - pi_x)(-1) = -pj_x + pi_x.
    # Term 2: -(pj_y - pi_y)(pk_x - pi_x). d/dpi_y -> -(-1)(pk_x - pi_x) = pk_x - pi_x.
    # Sum: -pj_x + pi_x + pk_x - pi_x = pk_x - pj_x = v2_x - v1_x? No.
    # Let's re-evaluate:
    # cross = pj_x pk_y - pj_x pi_y - pi_x pk_y + pi_x pi_y - pj_y pk_x + pj_y pi_x + pi_y pk_x - pi_y pi_x
    # d/dpi_x: -pk_y + pj_y = -(pk_y - pj_y) = -(v2_y - v1_y) = v1_y - v2_y.
    # My previous derivation: d_cross_di_x = -(v1_y + v2_y) was wrong.
    
    # Let's do it carefully.
    # v1 = pj - pi, v2 = pk - pi
    # cross = v1_x v2_y - v1_y v2_x
    # d(cross)/d(pi) = d(v1)/d(pi) x v2 + v1 x d(v2)/d(pi)
    # d(v1)/d(pi) = -I, d(v2)/d(pi) = -I
    # = (-I) x v2 + v1 x (-I) = -(-I x v2) - (v1 x I) ... vector cross product derivative is tricky.
    # Scalar:
    # d/dpi_x (v1_x v2_y - v1_y v2_x)
    # v1_x = pj_x - pi_x, v2_y = pk_y - pi_y
    # d/dpi_x (v1_x v2_y) = (-1)v2_y + v1_x(0) = -v2_y
    # d/dpi_x (-v1_y v2_x) = 0 + (-v1_y)(0) = 0? No, v1_y doesn't depend on pi_x.
    # Wait, v1_y = pj_y - pi_y. No x dependence.
    # So d/dpi_x (cross) = -v2_y.
    # d/dpi_y (cross):
    # d/dpi_y (v1_x v2_y) = v1_x(-1) = -v1_x
    # d/dpi_y (-v1_y v2_x) = (-1)v2_x = -v2_x
    # Sum = -(v1_x + v2_x).
    
    # Let's check d/dpj:
    # v1 = pj - pi, v2 = pk - pi
    # d/dpj_x (v1_x v2_y) = 1 * v2_y = v2_y
    # d/dpj_x (-v1_y v2_x) = 0
    # Sum = v2_y.
    # d/dpj_y (v1_x v2_y) = 0
    # d/dpj_y (-v1_y v2_x) = -1 * v2_x = -v2_x
    # Sum = -v2_x.
    
    # Let's check d/dpk:
    # d/dpk_x (v1_x v2_y) = 0
    # d/dpk_x (-v1_y v2_x) = -v1_y * 1 = -v1_y
    # Sum = -v1_y.
    # d/dpk_y (v1_x v2_y) = v1_x * 1 = v1_x
    # d/dpk_y (-v1_y v2_x) = 0
    # Sum = v1_x.

    d_cross_di_x = -v2[:, 1]
    d_cross_di_y = -(v1[:, 0] + v2[:, 0])
    
    d_cross_dj_x = v2[:, 1]
    d_cross_dj_y = -v2[:, 0]
    
    d_cross_dk_x = -v1[:, 1]
    d_cross_dk_y = v1[:, 0]
    
    # Weights for the smooth min
    # d(M_beta)/d(area_k) = w_k
    # d(M_beta)/d(point) = sum_k w_k * d(area_k)/d(point)
    # area_k = 0.5 * |cross_k|
    # d(area_k)/d(point) = 0.5 * sign(cross_k) * d(cross_k)/d(point)
    
    w = weights * signs * 0.5
    
    grad_points = np.zeros_like(points)
    
    # Accumulate gradients for pi, pj, pk
    # dA_di = w * [d_cross_di_x, d_cross_di_y]
    dA_di = w[:, None] * np.column_stack([d_cross_di_x, d_cross_di_y])
    dA_dj = w[:, None] * np.column_stack([d_cross_dj_x, d_cross_dj_y])
    dA_dk = w[:, None] * np.column_stack([d_cross_dk_x, d_cross_dk_y])
    
    np.add.at(grad_points, _I, dA_di)
    np.add.at(grad_points, _J, dA_dj)
    np.add.at(grad_points, _K, dA_dk)
    
    # Gradient of score = grad(smooth_min) / ha
    grad_score = grad_points / ha
    
    return -score, -grad_score.flatten()

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)

    def _update_best(points):
        nonlocal best_score, best_points
        sc = _score(points)
        if sc > best_score:
            best_score = sc
            best_points = points.copy()
        return sc

    def _try_optimize_smooth(init_points, maxiter=500):
        if time.time() > deadline - 5:
            return
        x0 = init_points.flatten()
        
        # Annealing beta from low to high
        betas = [10.0, 50.0, 200.0, 500.0]
        x = x0.copy()
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            def _obj_and_grad(x_flat, beta_val=beta):
                return _neg_smooth_score_and_grad(x_flat, beta_val)
            
            try:
                res = minimize(
                    _obj_and_grad,
                    x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50}
                )
                x = res.x
                _update_best(x.reshape(-1, 2))
            except Exception:
                pass
        
        # Final polish with Nelder-Mead on exact score
        if time.time() < deadline - 2:
            try:
                def _neg_exact_score(x_flat):
                    return -_score(x_flat.reshape(-1, 2))
                
                res = minimize(
                    _neg_exact_score,
                    x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                _update_best(res.x.reshape(-1, 2))
            except Exception:
                pass

    # Generate initial configurations
    # 1. Two concentric heptagons (current best structure)
    # Optimize the specific 2-param family first to get a good start
    def _compute_score_2p(params):
        r_in, dtheta = params
        angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
        angles_inner = angles_outer + dtheta
        inner = r_in * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
        pts = np.vstack([outer, inner])
        return _score(pts)

    # Quick optimization of 2-param family to find good starting point
    best_2p_score = 0.0
    best_2p_params = [0.5, np.pi/7]
    for sp in [[0.3, 0.0], [0.5, np.pi/7], [0.6, 0.0], [0.4, np.pi/14]]:
        try:
            res = minimize(lambda x: -_compute_score_2p(x), x0=sp, method='Nelder-Mead', options={'maxiter': 100})
            if -res.fun > best_2p_score:
                best_2p_score = -res.fun
                best_2p_params = res.x
        except:
            pass
    
    r_in, dtheta = best_2p_params
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    angles_inner = angles_outer + dtheta
    inner = r_in * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    init_7_7 = np.vstack([outer, inner])
    
    # 2. Other ring splits
    init_6_8 = np.vstack([_regular_polygon(6, 1.0, 0.0), _regular_polygon(8, 0.5, np.pi/8)])
    init_5_9 = np.vstack([_regular_polygon(5, 1.0, 0.0), _regular_polygon(9, 0.5, np.pi/9)])
    init_4_10 = np.vstack([_regular_polygon(4, 1.0, 0.0), _regular_polygon(10, 0.5, np.pi/10)])
    
    # Add some random perturbations to the best 7+7 to break symmetry
    inits = [
        init_7_7,
        init_7_7 + 0.01 * rng.randn(14, 2),
        init_6_8,
        init_5_9,
        init_4_10
    ]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
        _try_optimize_smooth(init)

    if best_points is None:
        # Fallback
        best_points = init_7_7
        
    return best_points
# EVOLVE-BLOCK-END
