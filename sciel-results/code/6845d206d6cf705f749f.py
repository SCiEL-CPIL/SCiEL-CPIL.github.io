"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _eval_packing(centers, radii):
    """
    Evaluate the packing. Returns (sum_radii, is_valid).
    """
    n = len(radii)
    total_r = np.sum(radii)
    
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r or x > 1.0 - r or y < r or y > 1.0 - r:
            return -np.inf, False
            
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist_sq = dx*dx + dy*dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                return -np.inf, False
                
    return total_r, True

def _grid_init(n):
    """
    Initialize with a grid layout.
    """
    side = int(np.ceil(np.sqrt(n)))
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        gx = i % side
        gy = i // side
        # Center of cell
        centers[i][0] = (gx + 0.5) / side
        centers[i][1] = (gy + 0.5) / side
        # Max radius is half the cell size (distance to edge), minus small margin
        radii[i] = 0.5 / side - 0.01
    return centers, radii

def _smooth_objective_and_grad(params, n, beta):
    """
    Smooth objective function and gradient for circle packing.
    Uses exponential penalties to enforce constraints.
    """
    # Unpack parameters
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Ensure radii are positive (handled by bounds, but clip for safety in exp)
    radii_safe = np.clip(radii, 1e-6, 1.0)
    
    objective = 0.0
    grad = np.zeros_like(params)
    
    # 1. Maximize sum of radii (minimize negative sum)
    # d/d(r_i) of -r_i is -1
    objective -= np.sum(radii_safe)
    grad[2*n:] -= 1.0
    
    # 2. Boundary Penalties
    # Penalty: exp(beta * (r - x)) + exp(beta * (x - (1-r))) etc.
    # Gradient w.r.t x: -beta * exp(beta * (r - x)) + beta * exp(beta * (x - (1-r)))
    # Gradient w.r.t r: beta * exp(beta * (r - x)) + beta * exp(beta * (r - y)) ...
    
    # Left boundary: x >= r
    # violation = r - x
    viol_l = radii_safe - centers[:, 0]
    pen_l = np.exp(beta * viol_l)
    objective += np.sum(pen_l)
    grad[:n] += -beta * pen_l  # d/dx
    grad[2*n:] += beta * pen_l  # d/dr
    
    # Right boundary: x <= 1 - r  => 1 - r - x >= 0
    viol_r = centers[:, 0] - (1.0 - radii_safe)
    pen_r = np.exp(beta * viol_r)
    objective += np.sum(pen_r)
    grad[:n] += beta * pen_r   # d/dx
    grad[2*n:] += -beta * pen_r # d/dr
    
    # Bottom boundary: y >= r
    viol_b = radii_safe - centers[:, 1]
    pen_b = np.exp(beta * viol_b)
    objective += np.sum(pen_b)
    grad[n:2*n] += -beta * pen_b
    grad[2*n:] += beta * pen_b
    
    # Top boundary: y <= 1 - r
    viol_t = centers[:, 1] - (1.0 - radii_safe)
    pen_t = np.exp(beta * viol_t)
    objective += np.sum(pen_t)
    grad[n:2*n] += beta * pen_t
    grad[2*n:] += -beta * pen_t
    
    # 3. Overlap Penalties
    # For each pair i, j: dist >= r_i + r_j
    # violation = r_i + r_j - dist
    # penalty = exp(beta * violation)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12) # Avoid division by zero
            
            min_dist = radii_safe[i] + radii_safe[j]
            viol = min_dist - dist
            
            # Only compute penalty if violation is significant or close
            # To save time, we can skip if viol is very negative (large gap)
            # But for gradient continuity, we should compute always or use a threshold.
            # Let's compute always for stability, but exp of large negative is ~0.
            
            pen = np.exp(beta * viol)
            objective += pen
            
            # Gradients
            # d(pen)/d(viol) = beta * pen
            # d(viol)/d(x_i) = -dx/dist
            # d(viol)/d(x_j) = dx/dist
            # d(viol)/d(r_i) = 1
            # d(viol)/d(r_j) = 1
            
            factor = beta * pen / dist
            grad[i*2] += -factor * dx
            grad[i*2+1] += -factor * dy
            grad[j*2] += factor * dx
            grad[j*2+1] += factor * dy
            grad[2*n + i] += beta * pen
            grad[2*n + j] += beta * pen
            
    return objective, grad

def _smooth_packing(n, time_limit, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Initial guess
    centers, radii = _grid_init(n)
    params = np.concatenate([centers.flatten(), radii])
    
    # Bounds: centers in [0,1], radii in [0.01, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    for i in range(n):
        bounds.append((0.01, 0.5)) # r
        
    # Multi-stage beta sharpening
    # Start with small beta to allow global movement, increase to enforce constraints strictly
    betas = [10, 50, 200, 1000, 5000]
    
    best_params = params.copy()
    best_score, _ = _eval_packing(centers, radii)
    
    start_time = time.time()
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Run L-BFGS-B
        # We need a wrapper for scipy that returns (f, g)
        def fun_and_grad(p):
            f, g = _smooth_objective_and_grad(p, n, beta)
            return f, g
            
        try:
            res = minimize(fun_and_grad, params, method='L-BFGS-B', 
                           jac=True, bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            params = res.x
        except Exception as e:
            pass # Keep previous params if optimization fails
            
        # Check validity and update best
        c = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        score, valid = _eval_packing(c, r)
        if valid and score > best_score:
            best_score = score
            best_params = params.copy()
            
    return best_params

def _high_precision_polish(params, n, time_limit):
    """
    Final polish with very high beta to ensure strict validity.
    """
    start_time = time.time()
    betas = [5000, 10000]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 2:
            break
        def fun_and_grad(p):
            f, g = _smooth_objective_and_grad(p, n, beta)
            return f, g
            
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.01, 0.5))
            
        try:
            res = minimize(fun_and_grad, params, method='L-BFGS-B', 
                           jac=True, bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            params = res.x
        except:
            pass
            
    return params

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    best_params = None
    best_score = -np.inf
    
    # Try a few different seeds/init strategies
    # 1. Grid init
    # 2. Random init
    # 3. Hex-like init
    
    for trial in range(4):
        if time.time() > deadline:
            break
            
        seed = trial * 100 + 42
        
        if trial == 0:
            # Grid
            c, r = _grid_init(N)
        elif trial == 1:
            # Random
            c = np.random.rand(N, 2) * 0.8 + 0.1
            r = np.full(N, 0.05)
        elif trial == 2:
            # Hexagonal-ish
            c = np.zeros((N, 2))
            r = np.zeros(N)
            side = 5 # 5x5 grid is 25, add 1
            for i in range(N):
                gx = i % side
                gy = i // side
                if gy % 2 == 1:
                    c[i, 0] = (gx + 0.5) / side + 0.5/side # Offset
                else:
                    c[i, 0] = (gx + 0.5) / side
                c[i, 1] = (gy + 0.5) / side
                c[i, 0] = np.clip(c[i, 0], 0.1, 0.9)
                c[i, 1] = np.clip(c[i, 1], 0.1, 0.9)
                r[i] = 0.08
        else:
            # Clustered random
            c = np.random.rand(N, 2)
            c = np.clip(c, 0.1, 0.9)
            r = np.full(N, 0.06)
            
        params = np.concatenate([c.flatten(), r])
        
        # Run smooth packing
        p_res = _smooth_packing(N, time_limit=25.0, seed=seed)
        
        c_res = p_res[:2*N].reshape(N, 2)
        r_res = p_res[2*N:]
        
        score, valid = _eval_packing(c_res, r_res)
        
        if valid and score > best_score:
            best_score = score
            best_params = p_res.copy()
            
        # If we have time, polish the best found so far
        if best_params is not None and time.time() < deadline - 10:
             p_polished = _high_precision_polish(best_params.copy(), N, time.time() - start_time)
             c_pol = p_polished[:2*N].reshape(N, 2)
             r_pol = p_polished[2*N:]
             score_pol, valid_pol = _eval_packing(c_pol, r_pol)
             if valid_pol and score_pol > best_score:
                 best_score = score_pol
                 best_params = p_polished.copy()

    if best_params is None:
        # Fallback to grid
        c, r = _grid_init(N)
        best_params = np.concatenate([c.flatten(), r])
        
    # Final extraction and safety cleanup
    result_c = best_params[:2*N].reshape(N, 2)
    result_r = best_params[2*N:]
    
    result = np.column_stack([result_c, result_r])
    
    # Strict validity check and repair
    for i in range(N):
        x, y, ri = result[i]
        # Clip radius to boundary
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if ri > max_r:
            ri = max(0.01, max_r)
        result[i, 2] = ri
        # Clip center to boundary based on radius
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    # Resolve overlaps by shrinking smaller circles
    for _ in range(20):
        modified = False
        for i in range(N):
            for j in range(i+1, N):
                dx = result[i, 0] - result[j, 0]
                dy = result[i, 1] - result[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = result[i, 2] + result[j, 2]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Shrink the smaller one by half the overlap
                    if result[i, 2] < result[j, 2]:
                        result[i, 2] = max(0.01, result[i, 2] - overlap/2)
                    else:
                        result[j, 2] = max(0.01, result[j, 2] - overlap/2)
                    modified = True
        if not modified:
            break
            
    return result
# EVOLVE-BLOCK-END
