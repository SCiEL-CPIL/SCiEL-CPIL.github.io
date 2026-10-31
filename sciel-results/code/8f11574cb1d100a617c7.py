import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 11
SQRT3 = np.sqrt(3)

def get_triangle_indices(n):
    """
    Generate indices for all unique triangles (i, j, k) with i < j < k.
    Returns arrays I, J, K of length nC3.
    """
    i = np.repeat(np.arange(n), n * (n - 1) // 2)
    # This is a bit tricky to generate perfectly with simple repeat.
    # Let's use a loop for clarity first, then optimize if needed. 
    # Actually, for N=11, 165 triangles. A loop is fine for index generation, 
    # but the objective/grad must be vectorized.
    idx_i = []
    idx_j = []
    idx_k = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx_i.append(i)
                idx_j.append(j)
                idx_k.append(k)
    return np.array(idx_i), np.array(idx_j), np.array(idx_k)

I_IDX, J_IDX, K_IDX = get_triangle_indices(N)
NUM_TRI = len(I_IDX)

def compute_areas_and_grads(points):
    """
    Compute areas and gradients of the smooth min objective.
    points: (N, 2)
    Returns:
        areas: (NUM_TRI,)
        grad: (N, 2) gradient of the soft-min of areas
    """
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    # Vectors
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Determinants
    D = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(D)
    
    # Signs for gradient
    signs = np.sign(D)
    # If D is 0, sign is 0. We can handle this or assume it's rare.
    # To be safe, replace 0 with 1? No, gradient is 0 if D=0 for abs? 
    # d/dx |x| is sign(x). At 0 it's undefined. 
    # Using soft-min smooths this out effectively.
    
    # We want to minimize -softmin(areas).
    # softmin(A) = -1/tau * log(sum(exp(-tau * A)))
    # Let's use a fixed tau. 
    # Gradient of softmin w.r.t A is weights: w_k = exp(-tau * A_k) / sum(exp(-tau * A_j))
    # Then grad_x softmin = sum_k w_k * grad_x A_k
    
    tau = 100.0 # Temperature for soft min
    
    # Numerical stability for log-sum-exp
    max_a = np.max(areas)
    # We are minimizing -softmin, so we maximize softmin.
    # softmin is approx min.
    # Let's compute weights for the soft-min.
    # w_k = exp(-tau * (A_k - max_a)) / sum(exp(-tau * (A_j - max_a)))
    exp_vals = np.exp(-tau * (areas - max_a))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient of each area w.r.t points
    # dA/dx_i = 0.5 * sign(D) * (2*y_i - y_j - y_k)
    # dA/dy_i = 0.5 * sign(D) * (2*x_i - x_j - x_k)
    
    grad_x = np.zeros((N, 2))
    
    # Accumulate gradients for i, j, k
    # dD/dx_i = -(y_k - y_i) - (y_j - y_i) = 2*y_i - y_j - y_k
    # dD/dy_i = -(x_k - x_i) - (x_j - x_i) = 2*x_i - x_j - x_k
    
    # dD/dx_j = y_k - y_i
    # dD/dy_j = -(x_k - x_i) = x_i - x_k
    
    # dD/dx_k = -(y_j - y_i) = y_i - y_j
    # dD/dy_k = x_j - x_i
    
    # Weighted sum
    w = weights * signs * 0.5
    
    # For i
    grad_x[I_IDX, 0] += w * (2 * points[I_IDX, 0] - points[J_IDX, 0] - points[K_IDX, 0]) # Wait, formula for dD/dx_i uses y's?
    # Let's re-verify:
    # D = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
    # dD/dx_i = -(y_k-y_i) - (y_j-y_i) = -y_k + y_i - y_j + y_i = 2y_i - y_j - y_k. Correct.
    # So dA/dx_i = 0.5 * sign * (2y_i - y_j - y_k)
    # My code above used x's for grad_x. That was wrong.
    
    # Let's redo properly.
    grad_x = np.zeros((N, 2))
    grad_y = np.zeros((N, 2))
    
    # dA/dx_i = 0.5 * sign * (2*y_i - y_j - y_k)
    # dA/dy_i = 0.5 * sign * (2*x_i - x_j - x_k)
    
    # dA/dx_j = 0.5 * sign * (y_k - y_i)
    # dA/dy_j = 0.5 * sign * (x_i - x_k)
    
    # dA/dx_k = 0.5 * sign * (y_i - y_j)
    # dA/dy_k = 0.5 * sign * (x_j - x_i)
    
    # Using np.add.at for accumulation
    np.add.at(grad_x, I_IDX, w * (2 * points[I_IDX, 1] - points[J_IDX, 1] - points[K_IDX, 1]))
    np.add.at(grad_y, I_IDX, w * (2 * points[I_IDX, 0] - points[J_IDX, 0] - points[K_IDX, 0]))
    
    np.add.at(grad_x, J_IDX, w * (points[K_IDX, 1] - points[I_IDX, 1]))
    np.add.at(grad_y, J_IDX, w * (points[I_IDX, 0] - points[K_IDX, 0]))
    
    np.add.at(grad_x, K_IDX, w * (points[I_IDX, 1] - points[J_IDX, 1]))
    np.add.at(grad_y, K_IDX, w * (points[J_IDX, 0] - points[I_IDX, 0]))
    
    # The objective is -softmin(areas) + penalty
    # grad(-softmin) = -grad(softmin)
    # grad(softmin) = sum w_k grad(A_k)
    # So grad_x_obj = -grad_x_accumulated
    # grad_y_obj = -grad_y_accumulated
    
    # Wait, the function returned is the gradient of the objective.
    # Objective = -softmin + penalty
    # Grad = - (grad_x, grad_y) + grad_penalty
    
    return areas, -grad_x, -grad_y

def penalty_and_grad(points):
    x = points[:, 0]
    y = points[:, 1]
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 - SQRT3 * x - y
    
    # Penalty is sum of max(0, -d)^2
    # Grad is 2 * max(0, -d) * (-grad_d)
    # grad_d1/dy = 1 -> grad_pen1 = 2*max(0,-y)*(-1) = -2*max(0,-y)
    # grad_d2/dx = SQRT3, d/dy = -1
    # grad_d3/dx = -SQRT3, d/dy = -1
    
    pen = 0
    gx = np.zeros(N)
    gy = np.zeros(N)
    
    m1 = np.maximum(0, -d1)
    pen += np.sum(m1**2)
    gy += -2 * m1
    
    m2 = np.maximum(0, -d2)
    pen += np.sum(m2**2)
    gx += -2 * m2 * SQRT3
    gy += 2 * m2
    
    m3 = np.maximum(0, -d3)
    pen += np.sum(m3**2)
    gx += 2 * m3 * SQRT3
    gy += 2 * m3
    
    return pen, gx, gy

def objective_grad(flat_params):
    points = flat_params.reshape(N, 2)
    
    # Check bounds
    if np.any(points[:, 0] < -1) or np.any(points[:, 0] > 2) or np.any(points[:, 1] < -1) or np.any(points[:, 1] > 2):
        return 1e6, np.zeros(2*N)
    
    areas, g_area_x, g_area_y = compute_areas_and_grads(points)
    pen, g_pen_x, g_pen_y = penalty_and_grad(points)
    
    # We want to maximize min area. 
    # Objective = -softmin(areas) + 1000 * penalty
    # Value = -softmin + 1000*pen
    # Grad = -grad_softmin + 1000*grad_pen
    
    # g_area_x is already -grad_softmin_x (from compute_areas_and_grads return)
    # Wait, compute_areas_and_grads returned -grad_x.
    # Let's check: 
    # "The objective is -softmin(areas) + penalty"
    # "Grad = - (grad_x, grad_y) + grad_penalty"
    # In compute_areas_and_grads, I returned -grad_x, -grad_y.
    # So g_area_x is the gradient contribution from -softmin.
    
    val = -(-1/tau * np.log(np.sum(np.exp(-tau * areas)))) + 1000.0 * pen
    # Actually, we can just compute the value directly for the return
    # But we need the value to be consistent with the gradient.
    # Let's recompute value using the softmin formula to be safe.
    max_a = np.max(areas)
    softmin_val = -1/tau * (np.log(np.sum(np.exp(-tau * (areas - max_a)))) + tau * max_a)
    val = -softmin_val + 1000.0 * pen
    
    grad = np.concatenate([g_area_x + 1000.0 * g_pen_x, g_area_y + 1000.0 * g_pen_y])
    
    return val, grad

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Initializations
    # 1. Random
    # 2. Grid-like
    
    for seed in range(100):
        if time.time() - start_time > time_limit:
            break
        
        rng = np.random.RandomState(seed)
        
        if seed % 2 == 0:
            # Random
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
        else:
            # Heuristic: Place points on a grid and jitter
            # 4x3 grid inside triangle?
            init_points = np.zeros((N, 2))
            # Vertices
            init_points[0] = [0, 0]
            init_points[1] = [1, 0]
            init_points[2] = [0.5, SQRT3/2]
            # Centroid
            init_points[3] = [0.5, SQRT3/6]
            # Midpoints
            init_points[4] = [0.5, 0]
            init_points[5] = [0.25, SQRT3/4]
            init_points[6] = [0.75, SQRT3/4]
            # Inner points
            # Add noise to a regular hexagon pattern?
            for i in range(7, N):
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
            # Use L-BFGS-B
            result = minimize(
                objective_grad, 
                flat_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'maxls': 50, 'disp': False}
            )
            
            opt_points = result.x.reshape(N, 2)
            
            # Refine with Powell (derivative free) to handle non-smoothness
            if time.time() - start_time < time_limit - 5:
                result2 = minimize(
                    lambda p: -np.min(compute_areas_and_grads(p.reshape(N,2))[0]) + 1000*penalty_and_grad(p.reshape(N,2))[0], 
                    result.x, 
                    method='Powell', 
                    options={'maxiter': 50, 'maxfev': 2000, 'disp': False}
                )
                opt_points = result2.x.reshape(N, 2)
            
            min_area = np.min(compute_areas_and_grads(opt_points)[0])
            pen = penalty_and_grad(opt_points)[0]
            
            if pen < 1e-4:
                score = min_area
            else:
                score = min_area - 1000 * pen
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception as e:
            pass

    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            # Final polish with basinhopping
            bh_result = basinhopping(
                lambda p: -np.min(compute_areas_and_grads(p.reshape(N,2))[0]) + 1000*penalty_and_grad(p.reshape(N,2))[0],
                best_points.flatten(),
                minimizer_kwargs={'method': 'Powell', 'maxiter': 50, 'maxfev': 1000},
                niter=5,
                seed=123
            )
            bh_points = bh_result.x.reshape(N, 2)
            bh_min_area = np.min(compute_areas_and_grads(bh_points)[0])
            bh_pen = penalty_and_grad(bh_points)[0]
            
            if bh_pen < 1e-4:
                bh_score = bh_min_area
            else:
                bh_score = bh_min_area - 1000 * bh_pen
                
            if bh_score > best_score:
                best_score = bh_score
                best_points = bh_points.copy()
        except Exception:
            pass

    if best_points is None:
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
