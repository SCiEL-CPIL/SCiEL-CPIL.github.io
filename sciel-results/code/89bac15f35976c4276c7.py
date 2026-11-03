import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 100.0  # 100s limit to be safe
    
    # We optimize the first half of the symmetric function
    # Initial guess: Cosine bump
    x = np.linspace(0, 1, N//2)
    # f(t) = cos(pi * t) mapped to [0, 1]
    # t=0 -> 1, t=1 -> 0
    init_half = np.cos(np.pi * x)
    init_half = np.maximum(init_half, 0.0)
    
    def unpack(half):
        # half is length N//2
        # Symmetric extension
        full = np.concatenate([half, half[::-1]])
        return full
        
    def objective(half):
        if time.time() > deadline:
            return 1e9
        f = unpack(half)
        sum_f = np.sum(f)
        if sum_f < 1e-9:
            return 1e9
        sum_sq = np.sum(f**2)
        return sum_sq / sum_f
        
    def gradient(half):
        # Numerical gradient or analytical?
        # Analytical:
        # J = S2 / S1
        # dJ/dfi = (2fi * S1 - S2) / S1^2
        if time.time() > deadline:
            return np.zeros_like(half)
        f = unpack(half)
        S1 = np.sum(f)
        S2 = np.sum(f**2)
        if S1 < 1e-9:
            return np.zeros_like(half)
        
        # Gradient with respect to full f
        grad_full = (2.0 * f * S1 - S2) / (S1**2)
        
        # Map gradient back to half
        # grad_half[i] = grad_full[i] + grad_full[N-1-i]
        grad_half = grad_full[:N//2] + grad_full[N//2:][::-1]
        return grad_half
        
    # Optimization
    res = None
    try:
        res = minimize(
            objective, 
            init_half, 
            jac=gradient, 
            method='L-BFGS-B', 
            bounds=[(0, None)] * (N//2),
            options={'maxiter': 1000, 'maxfun': 10000}
        )
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        # Return best found or initial
        if res and res.success:
             best_half = res.x
        else:
             best_half = init_half
    else:
        if res and res.success:
             best_half = res.x
        else:
             best_half = init_half

    # Construct final full array
    best_full = unpack(best_half)
    
    # Ensure non-negative
    best_full = np.maximum(best_full, 0.0)
    
    # Convert to list
    return list(best_full)
