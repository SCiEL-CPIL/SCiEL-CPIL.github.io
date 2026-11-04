import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def _compute_objective(coeffs):
    a0, a4, a8 = coeffs
    # Calculate a12 such that P(0) = 0
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-332640
    a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
    
    # Use FFT-based computation for accurate norms
    N = 2**15
    L = 20.0
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    g = P * np.exp(-np.pi * x**2)
    
    # Direct norms of g
    norm2_sq = np.sum(g**2) * dx
    norm4_sq = np.sum(g**4) * dx
    
    # FFT norms of g_hat
    G = np.fft.fft(g)
    norm2_sq_hat = (dx / N) * np.sum(G**2)
    norm4_sq_hat = (dx**3 / N) * np.sum(G**4)
    
    if norm2_sq < 1e-10 or norm2_sq_hat < 1e-10:
        return 1e10
        
    obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
    
    return obj

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin
    
    # Prior state of the art
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_score = _compute_objective(x0)
    
    best_result = x0.copy()
    best_score = seed_score
    
    # Multi-start local optimization with diverse starting points
    starts = [
        x0,
        x0 + np.array([0.05, 0.005, 0.00005]),
        x0 - np.array([0.05, 0.005, 0.00005]),
        x0 + np.array([-0.1, 0.01, 0.0001]),
        x0 + np.array([0.1, -0.01, -0.0001]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-0.9, 0.03, 0.0002]),
        np.array([-1.5, 0.06, 0.0005]),
    ]
    
    for s in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(_compute_objective, s, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 30:
        try:
            res = minimize(_compute_objective, best_result, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
