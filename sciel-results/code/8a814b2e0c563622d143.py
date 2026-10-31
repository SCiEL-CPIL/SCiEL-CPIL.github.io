import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all triangles
TRI_INDICES = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRI_INDICES.append((i, j, k))
NUM_TRI = len(TRI_INDICES)

def calc_areas_and_grads(points):
    """
    Calculate areas and gradients for all triangles.
    Returns:
    areas: array of shape (NUM_TRI,)
    grads: array of shape (NUM_TRI, N, 2)
    """
    areas = np.zeros(NUM_TRI)
    grads = np.zeros((NUM_TRI, N, 2))
    
    for t, (i, j, k) in enumerate(TRI_INDICES):
        p_i = points[i]
        p_j = points[j]
        p_k = points[k]
        
        # Cross product z-component: (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # Expanded: x_j y_k - x_j y_i - x_i y_k + x_i y_i - y_j x_k + y_j x_i + y_i x_k - y_i x_i
        # S = x_j(y_k - y_i) - x_i(y_k - y_i) - y_j(x_k - x_i) + y_i(x_k - x_i)
        # S = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        
        dx_ij = p_j[0] - p_i[0]
        dy_ij = p_j[1] - p_i[1]
        dx_ik = p_k[0] - p_i[0]
        dy_ik = p_k[1] - p_i[1]
        
        S = dx_ij * dy_ik - dy_ij * dx_ik
        
        if S >= 0:
            sign = 1.0
        else:
            sign = -1.0
            
        area_val = 0.5 * abs(S)
        areas[t] = area_val
        
        # Gradients of S w.r.t p_i, p_j, p_k
        # dS/dx_i = -(dy_ik) + (dy_ij) ... wait.
        # S = x_j y_k - x_j y_i - x_i y_k + x_i y_i - y_j x_k + y_j x_i + y_i x_k - y_i x_i
        # dS/dx_i = -y_k + y_i + y_j
        # dS/dy_i = -x_j + x_i + x_k
        
        # dS/dx_j = y_k - y_i
        # dS/dy_j = -x_k + x_i
        
        # dS/dx_k = -y_j + y_i
        # dS/dy_k = x_j - x_i
        
        grad_i = np.array([-(p_k[1] - p_i[1]) + (p_j[1] - p_i[1]), 
                           -(p_j[0] - p_i[0]) + (p_k[0] - p_i[0])])
        # Simplify:
        # dS/dx_i = y_i + y_j - y_k
        # dS/dy_i = x_i + x_k - x_j
        grad_i = np.array([p_i[1] + p_j[1] - p_k[1], 
                           p_i[0] + p_k[0] - p_j[0]])
        
        # dS/dx_j = y_k - y_i
        # dS/dy_j = x_i - x_k
        grad_j = np.array([p_k[1] - p_i[1], 
                           p_i[0] - p_k[0]])
        
        # dS/dx_k = y_i - y_j
        # dS/dy_k = x_j - x_i
        grad_k = np.array([p_i[1] - p_j[1], 
                           p_j[0] - p_i[0]])
        
        # Gradient of Area = 0.5 * sign(S) * Gradient of S
        factor = 0.5 * sign
        grads[t, i, :] = factor * grad_i
        grads[t, j, :] = factor * grad_j
        grads[t, k, :] = factor * grad_k
        
    return areas, grads

def penalty_and_grad(points):
    """
    Calculate boundary penalty and its gradient.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    
    # Penalty is sum of max(0, -d)^2
    # Grad is 2 * max(0, -d) * (-grad_d)
    
    pen = 0.0
    grad_pen = np.zeros_like(points)
    
    # d1 = y
    viol1 = np.maximum(0, -d1)
    pen += np.sum(viol1**2)
    grad_pen[:, 1] += 2 * viol1 * (-1) # d/dy (y) = 1, so -1
    
    # d2 = sqrt(3)x - y
    viol2 = np.maximum(0, -d2)
    pen += np.sum(viol2**2)
    # grad d2 = [sqrt(3), -1]
    grad_pen[:, 0] += 2 * viol2 * (-SQRT3)
    grad_pen[:, 1] += 2 * viol2 * (1)
    
    # d3 = sqrt(3) - sqrt(3)x - y
    viol3 = np.maximum(0, -d3)
    pen += np.sum(viol3**2)
    # grad d3 = [-sqrt(3), -1]
    grad_pen[:, 0] += 2 * viol3 * (SQRT3)
    grad_pen[:, 1] += 2 * viol3 * (1)
    
    return pen, grad_pen

def objective_and_grad(flat_params):
    """
    Objective: -min_area_smooth + penalty
    """
    points = flat_params.reshape(N, 2)
    
    # Smoothed min area using log-sum-exp
    # F = -1/tau * ln(sum(exp(-A_i/tau)))
    # We want to maximize min A, so minimize -min A.
    # Let's use a temperature tau.
    tau = 0.01
    
    areas, grads = calc_areas_and_grads(points)
    
    # To avoid overflow, shift areas
    max_area = np.max(areas)
    exp_terms = np.exp(-(areas - max_area) / tau)
    sum_exp = np.sum(exp_terms)
    
    # Gradient of -1/tau * ln(sum(exp(-A/tau)))
    # d/dp = -1/tau * (1/sum) * sum( exp(-A/tau) * d/dp(-A/tau) )
    # d/dp(-A/tau) = -1/tau * grad_A
    # So grad_obj = -1/tau * (1/sum) * sum( exp * (-1/tau) * grad_A )
    #             = 1/tau^2 * (1/sum) * sum( exp * grad_A )
    
    # Weights
    weights = exp_terms / sum_exp
    
    # Sum of weighted gradients
    # grads is (NUM_TRI, N, 2)
    # weights is (NUM_TRI,)
    # We need sum_t weights[t] * grads[t]
    # This results in (N, 2)
    grad_min_area = np.tensordot(weights, grads, axes=(0, 0))
    
    # The objective is -min_area. So we negate the gradient of min_area.
    # Wait, F = -min_area. Grad F = -Grad(min_area).
    # My derivation above was for F = -1/tau ln(sum exp(-A/tau)) which approximates -min A.
    # So grad_F = 1/tau^2 * weighted_sum(grads).
    # This is the gradient of the objective function we want to MINIMIZE.
    
    obj_val = -1.0 / tau * np.log(sum_exp) + max_area / tau # This is -min_area approx
    # Actually, just return the value and gradient.
    # Value: - (max_area - tau * ln(sum_exp)) ?
    # min_area_approx = max_area - tau * ln(sum_exp)
    # Obj = -min_area_approx = -max_area + tau * ln(sum_exp)
    
    obj_val = -max_area + tau * np.log(sum_exp)
    
    grad_obj = (1.0 / tau) * grad_min_area
    
    # Add penalty
    pen_val, grad_pen = penalty_and_grad(points)
    
    obj_val += 100.0 * pen_val
    grad_obj += 100.0 * grad_pen
    
    return obj_val, grad_obj.flatten()

def safe_objective_and_grad(flat_params):
    points = flat_params.reshape(N, 2)
    if np.any(points[:, 0] < -1) or np.any(points[:, 0] > 2) or np.any(points[:, 1] < -1) or np.any(points[:, 1] > 2):
        return 1e6, np.zeros_like(flat_params)
    return objective_and_grad(flat_params)

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple random restarts with L-BFGS-B and analytical gradients
    for seed in range(30):
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(seed)
        init_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            init_points[i, 0] = v * 1.0 + w * 0.5
            init_points[i, 1] = w * SQRT3 / 2.0
        
        flat_init = init_points.flatten()
        
        try:
            result = minimize(
                lambda x: safe_objective_and_grad(x)[0], 
                flat_init, 
                jac=lambda x: safe_objective_and_grad(x)[1],
                method='L-BFGS-B', 
                options={'maxiter': 200, 'maxfun': 10000, 'disp': False}
            )
            
            opt_points = result.x.reshape(N, 2)
            areas, _ = calc_areas_and_grads(opt_points)
            min_area = np.min(areas)
            pen_val, _ = penalty_and_grad(opt_points)
            
            if pen_val < 1e-4:
                score = min_area
            else:
                score = min_area - 100 * pen_val
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            pass

    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            # Refine with basinhopping using the same gradient info
            # Note: basinhopping expects a function returning value, and jac for minimizer
            bh_result = basinhopping(
                lambda x: safe_objective_and_grad(x)[0],
                best_points.flatten(),
                minimizer_kwargs={'method': 'L-BFGS-B', 'jac': lambda x: safe_objective_and_grad(x)[1], 'maxiter': 100, 'maxfun': 5000},
                niter=5,
                seed=123
            )
            bh_points = bh_result.x.reshape(N, 2)
            areas, _ = calc_areas_and_grads(bh_points)
            bh_min_area = np.min(areas)
            bh_pen, _ = penalty_and_grad(bh_points)
            
            if bh_pen < 1e-4:
                bh_score = bh_min_area
            else:
                bh_score = bh_min_area - 100 * bh_pen
                
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_points.copy()
        except Exception:
            pass

    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
