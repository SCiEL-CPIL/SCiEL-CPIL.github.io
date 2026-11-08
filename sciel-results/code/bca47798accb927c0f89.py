import numpy as np
from itertools import combinations
from scipy.optimize import minimize
import time

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])

# Precompute triangle indices
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])

# Edge normals for penalty
# Edge 1: V0-V1 (y=0), normal (0, -1) for interior
# Edge 2: V1-V2, normal (1, -1/sqrt(3))?
# Line V1-V2: y - 0 = -sqrt(3) (x - 1) => sqrt(3)x + y - sqrt(3) = 0.
# Interior is V0(0,0): -sqrt(3) < 0. So normal (sqrt(3), 1) points out?
# dist = (sqrt(3)x + y - sqrt(3)) / 2.
# Edge 3: V2-V0, line: y = sqrt(3)x => -sqrt(3)x + y = 0.
# Interior V1(1,0): -sqrt(3) < 0. Normal (-sqrt(3), 1).
# dist = (-sqrt(3)x + y) / 2.

def _compute_areas(points):
    """
    points: (N, 2)
    returns: areas (165,), signs (165,)
    """
    pi = points[_idx_i]
    pj = points[_idx_j]
    pk = points[_idx_k]
    
    v1 = pj - pi
    v2 = pk - pi
    S = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(S)
    signs = np.sign(S)
    # Handle S=0
    signs[S == 0] = 1.0
    return areas, signs, pi, pj, pk

def _soft_min_and_grad(points, beta):
    """
    Returns soft_min_value, gradient (N, 2)
    """
    areas, signs, pi, pj, pk = _compute_areas(points)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # For numerical stability, shift areas
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min_val = -1.0 / beta * (log_sum - beta * max_a) # Wait, -1/beta * log(sum exp(-beta A))
    # log(sum exp(-beta A)) = log(sum exp(-beta (A-max))) - beta max
    # = log_sum - beta max_a
    # soft_min = -1/beta * (log_sum - beta max_a) = max_a - 1/beta log_sum
    
    # Weights w_k = exp(-beta A_k) / sum
    # w_k = exp(-beta (A_k - max_a)) / sum_exp
    weights = exp_terms / sum_exp
    
    # Gradient of -soft_min is what we want for minimization?
    # We want to maximize min area. So minimize -min area.
    # f = -soft_min
    # grad f = - grad soft_min
    # grad soft_min = - sum(w_k * grad A_k)
    # So grad f = sum(w_k * grad A_k)
    
    grad = np.zeros((N, 2))
    
    # dA/dPi = 0.5 * sign * (yj - yk, xk - xj)
    # dA/dPj = 0.5 * sign * (yk, -xk)
    # dA/dPk = 0.5 * sign * (-yj, xj)
    
    coeff = 0.5 * signs * weights
    
    # For Pi
    dPi_x = coeff * (pj[:, 1] - pk[:, 1])
    dPi_y = coeff * (pk[:, 0] - pj[:, 0])
    
    # For Pj
    dPj_x = coeff * pk[:, 1]
    dPj_y = coeff * (-pk[:, 0])
    
    # For Pk
    dPk_x = coeff * (-pj[:, 1])
    dPk_y = coeff * (pj[:, 0])
    
    np.add.at(grad, (_idx_i, 0), dPi_x)
    np.add.at(grad, (_idx_i, 1), dPi_y)
    np.add.at(grad, (_idx_j, 0), dPj_x)
    np.add.at(grad, (_idx_j, 1), dPj_y)
    np.add.at(grad, (_idx_k, 0), dPk_x)
    np.add.at(grad, (_idx_k, 1), dPk_y)
    
    return soft_min_val, grad

def _penalty_and_grad(points):
    """
    Returns penalty_value, gradient (N, 2)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Edge 1: y >= 0. dist = y. If y < 0, penalty y^2.
    # Edge 2: sqrt(3)x + y - sqrt(3) <= 0. dist = (sqrt(3)x + y - sqrt(3))/2.
    # Edge 3: -sqrt(3)x + y <= 0. dist = (-sqrt(3)x + y)/2.
    
    d1 = y
    d2 = (np.sqrt(3) * x + y - np.sqrt(3)) / 2.0
    d3 = (-np.sqrt(3) * x + y) / 2.0
    
    # Penalty is sum of squared distances for negative distances
    pen1 = np.where(d1 < 0, d1**2, 0)
    pen2 = np.where(d2 < 0, d2**2, 0)
    pen3 = np.where(d3 < 0, d3**2, 0)
    
    total_pen = np.sum(pen1) + np.sum(pen2) + np.sum(pen3)
    
    grad_pen = np.zeros((N, 2))
    
    # Grad of d1^2 = 2*d1 * (0, 1)
    grad_pen[:, 1] += 2 * np.where(d1 < 0, d1, 0)
    
    # Grad of d2^2 = 2*d2 * (sqrt(3)/2, 1/2)
    mask2 = d2 < 0
    grad_pen[:, 0] += 2 * np.where(mask2, d2, 0) * (np.sqrt(3) / 2.0)
    grad_pen[:, 1] += 2 * np.where(mask2, d2, 0) * 0.5
    
    # Grad of d3^2 = 2*d3 * (-sqrt(3)/2, 1/2)
    mask3 = d3 < 0
    grad_pen[:, 0] += 2 * np.where(mask3, d3, 0) * (-np.sqrt(3) / 2.0)
    grad_pen[:, 1] += 2 * np.where(mask3, d3, 0) * 0.5
    
    return total_pen, grad_pen

def _objective_and_jac(points_flat, beta, lambda_pen):
    points = points_flat.reshape(N, 2)
    
    sm_val, sm_grad = _soft_min_and_grad(points, beta)
    pen_val, pen_grad = _penalty_and_grad(points)
    
    # Minimize -sm_val + lambda * pen_val
    val = -sm_val + lambda_pen * pen_val
    jac = -sm_grad + lambda_pen * pen_grad
    
    return val, jac.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = 0.0
    best_points = None
    
    # Initial points: random
    rng = np.random.RandomState(42)
    
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Random init inside triangle
        u = rng.rand(N)
        v = rng.rand(N)
        # Map to triangle
        # If u+v > 1, reflect? Or just use rejection?
        # Simple: u' = u, v' = v * (1-u) ? No.
        # Barycentric: P = (1-u-v)V0 + u V1 + v V2.
        # Need u>=0, v>=0, u+v<=1.
        # Sample u,v from [0,1]. If u+v>1, set u=1-v, v=v? No.
        # Just sample u,v and if u+v>1, swap/adjust?
        # Easiest: rejection sampling for init.
        pts = []
        while len(pts) < N:
            uu, vv = rng.rand(2)
            if uu + vv < 1.0:
                pts.append([uu + 0.5*vv, (np.sqrt(3)/2)*vv])
        x0 = np.array(pts).flatten()
        
        # Optimize
        try:
            res = minimize(
                _objective_and_jac,
                x0,
                args=(50.0, 10.0), # beta, lambda_pen
                jac=True,
                method='L-BFGS-B',
                bounds=[(0, 1), (0, np.sqrt(3)/2)] * N,
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if res.success or res.status in [1, 2, 3, 4]: # 1: max iter, 2: ftol, 3: gtol, 4: gtol
                final_points = res.x.reshape(N, 2)
                # Calculate true min area
                areas, _, _, _, _ = _compute_areas(final_points)
                min_area = np.min(areas)
                
                if min_area > best_score:
                    best_score = min_area
                    best_points = final_points
                    
        except Exception:
            pass
            
    if best_points is None:
        # Fallback
        return np.array([[0.5, np.sqrt(3)/6] * N])
        
    return best_points
