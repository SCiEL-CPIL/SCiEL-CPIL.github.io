import numpy as np
import time
from scipy.optimize import minimize

N = 600
DELTA = 1.0 / 1200.0
TARGET_NORM_SQ = 1.0 / DELTA  # 1200

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 150s limit
    
    # Initial guess: uniform
    h0 = np.ones(N)
    norm = np.sqrt(np.sum(h0**2))
    h0 = h0 * (np.sqrt(TARGET_NORM_SQ) / norm)
    
    def objective(h):
        """Minimize the max of autoconvolution."""
        conv = np.convolve(h, h, mode='full')
        return DELTA * np.max(conv)
    
    def objective_grad(h):
        """Gradient of the max of autoconvolution."""
        conv = np.convolve(h, h, mode='full')
        max_val = np.max(conv)
        # Find where the max occurs (there may be multiple locations)
        max_idx = np.where(conv == max_val)[0]
        
        # Gradient of max(conv) with respect to h
        # conv[k] = sum_i h[i]*h[k-i] (for valid range)
        # d(conv[k])/d(h[j]) = h[k-j] + h[j-k] (with appropriate bounds)
        # For the max, gradient is sum over max locations of d(conv[max_loc])/d(h)
        
        grad = np.zeros(N)
        for idx in max_idx:
            # conv[idx] = sum_{i} h[i] * h[idx-i] where 0<=i<N and 0<=idx-i<N
            # d(conv[idx])/d(h[j]) = h[idx-j] if 0<=idx-j<N else 0
            # Plus the symmetric term from the other side
            for j in range(N):
                # h[j] * h[idx-j] contributes h[idx-j] to d/dh[j]
                # h[idx-j] * h[j] contributes h[j] to d/dh[idx-j]
                # So total contribution to d(conv[idx])/d(h[j]) is:
                # If 0 <= idx-j < N: h[idx-j]
                # If 0 <= j-idx < N: h[j-idx]
                if 0 <= idx - j < N:
                    grad[j] += h[idx - j]
                if 0 <= j - idx < N:
                    grad[j] += h[j - idx]
        
        # Scale by DELTA
        return DELTA * grad
    
    # Constraints: ||h||_2^2 = TARGET_NORM_SQ, h >= 0
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h**2) - TARGET_NORM_SQ,
         'jac': lambda h: 2 * h},
    ]
    
    bounds = [(0, None)] * N
    
    # Try multiple starting points
    best_h = h0.copy()
    best_score = objective(h0)
    
    rng = np.random.default_rng(42)
    
    # Try a few different initializations
    init_list = [h0]
    
    # Random initialization
    for _ in range(3):
        h_rand = rng.random(N)
        norm = np.sqrt(np.sum(h_rand**2))
        if norm > 0:
            h_rand = h_rand * (np.sqrt(TARGET_NORM_SQ) / norm)
            init_list.append(h_rand)
    
    # Structured: concentrated in center
    h_center = np.zeros(N)
    center = N // 2
    width = N // 4
    h_center[center - width:center + width] = 1.0
    norm = np.sqrt(np.sum(h_center**2))
    h_center = h_center * (np.sqrt(TARGET_NORM_SQ) / norm)
    init_list.append(h_center)
    
    # Structured: two peaks
    h_two = np.zeros(N)
    h_two[N//4:N//4+N//8] = 1.0
    h_two[3*N//4:3*N//4+N//8] = 1.0
    norm = np.sqrt(np.sum(h_two**2))
    h_two = h_two * (np.sqrt(TARGET_NORM_SQ) / norm)
    init_list.append(h_two)
    
    for init in init_list:
        if time.time() > deadline - 20:
            break
        
        try:
            result = minimize(
                objective,
                init,
                jac=objective_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success or result.fun < best_score:
                # Project back to feasible set
                h_opt = result.x
                # Ensure non-negativity
                h_opt = np.maximum(h_opt, 0)
                # Normalize
                norm = np.sqrt(np.sum(h_opt**2))
                if norm > 0:
                    h_opt = h_opt * (np.sqrt(TARGET_NORM_SQ) / norm)
                
                score = objective(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt
                    
        except Exception:
            continue
    
    # If we have time, do a local refinement with projected gradient
    if time.time() < deadline - 10:
        h = best_h.copy()
        lr = 0.01
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            grad = objective_grad(h)
            h_new = h - lr * grad
            h_new = np.maximum(h_new, 0)
            norm = np.sqrt(np.sum(h_new**2))
            if norm > 0:
                h_new = h_new * (np.sqrt(TARGET_NORM_SQ) / norm)
            
            score_new = objective(h_new)
            if score_new < objective(h):
                h = h_new
            else:
                lr *= 0.5
        
        score = objective(h)
        if score < best_score:
            best_score = score
            best_h = h
    
    return list(best_h)
