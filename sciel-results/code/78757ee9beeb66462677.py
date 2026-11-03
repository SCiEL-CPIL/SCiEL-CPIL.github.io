import numpy as np
from scipy.optimize import minimize
import time

# Hermite polynomials H_n(x) (Physicist's convention)
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H_12 = 4096x^12 - 30720x^10 + 72192x^8 - 70512x^6 + 30240x^4 - 5040x^2 + 1680? 
# Let's check H_12(0). 
# Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
# H_0 = 1
# H_1 = 2x
# H_2 = 4x^2 - 2
# H_3 = 8x^3 - 12x
# H_4 = 16x^4 - 48x^2 + 12
# H_5 = 32x^5 - 160x^3 + 120x
# H_6 = 64x^6 - 480x^4 + 720x^2 - 120
# H_7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
# H_8 = 256x^8 - 3584x^6 + 13440x^4 - 13440x^2 + 1680
# H_9 = 512x^9 - 10752x^7 + 60480x^5 - 120960x^3 + 60480x
# H_10 = 1024x^10 - 35840x^8 + 322560x^6 - 1075200x^4 + 1075200x^2 - 120960
# H_11 = 2048x^11 - 114688x^9 + 1433600x^7 - 6839040x^5 + 13516800x^3 - 6839040x
# H_12 = 4096x^12 - 367008x^10 + 6291456x^8 - 39813120x^6 + 118464000x^4 - 153676800x^2 + 39916800? 
# Let's use scipy.special.eval_hermite to be safe.

def hermite(n, x):
    return np.polynomial.hermite.hermval(x, np.polynomial.hermite.herm2phys([0]*n + [1]))

# Actually, simpler to use scipy.special.eval_hermite
from scipy.special import eval_hermite

def compute_objective(coeffs):
    a0, a4, a8 = coeffs
    # Calculate a12 such that P(0) = 0
    # P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-332640
    # a0 - 12a4 + 1680a8 - 332640 a12 = 0
    a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
    
    # Define P(x) and P_minus(x)
    # P(x) = a0 H0(x) + a4 H4(x) + a8 H8(x) + a12 H12(x)
    # P_minus(x) = a0 H0(x) + a4 H4(x) + a8 H8(x) - a12 H12(x) (since i^12 = -1)
    
    # We need to compute:
    # Norm2_sq = integral |g|^2 = integral P(x)^2 e^{-2 pi x^2}
    # Norm4_sq = integral |g|^4 = integral P(x)^4 e^{-4 pi x^2}
    # Same for Fourier side with P_minus
    
    # Use numerical integration
    # Range [-10, 10] should be sufficient for Gaussian decay
    x = np.linspace(-10, 10, 2000)
    dx = x[1] - x[0]
    
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    P_minus = a0*H0 + a4*H4 + a8*H8 - a12*H12
    
    exp_2pi = np.exp(-2 * np.pi * x**2)
    exp_4pi = np.exp(-4 * np.pi * x**2)
    
    norm2_sq = np.sum(P**2 * exp_2pi) * dx
    norm4_sq = np.sum(P**4 * exp_4pi) * dx
    
    norm2_sq_hat = np.sum(P_minus**2 * exp_2pi) * dx
    norm4_sq_hat = np.sum(P_minus**4 * exp_4pi) * dx
    
    if norm2_sq == 0 or norm2_sq_hat == 0:
        return np.inf
        
    obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
    
    return obj

def construct():
    start_time = time.time()
    deadline = 180.0 # Safety margin
    
    # Initial guess from seed
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_result = None
    best_score = np.inf
    
    # Try multiple starting points to avoid local minima
    # The function is likely smooth but non-convex
    starts = [
        x0,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-0.5, 0.05, 0.0005]),
        np.array([-2.0, 0.2, 0.002]),
        np.array([0, 0, 0]) # Will be projected or handled, but a12 handles P(0)=0
    ]
    
    for s in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(compute_objective, s, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    # If optimization failed or didn't improve, return seed
    if best_result is None:
        return x0
        
    # Verify the result is better than seed
    seed_score = compute_objective(x0)
    if best_score < seed_score:
        return best_result
    else:
        return x0

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def _compute_objective(coeffs):
    a0, a4, a8 = coeffs
    # Calculate a12 such that P(0) = 0
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-332640
    a12 = (a0 - 12*a4 + 1680*a8) / 332640.0
    
    # Numerical integration for norms
    # Using a fixed grid for speed and consistency
    x = np.linspace(-8, 8, 1000)
    dx = x[1] - x[0]
    
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    P_minus = a0*H0 + a4*H4 + a8*H8 - a12*H12
    
    exp_2pi = np.exp(-2 * np.pi * x**2)
    exp_4pi = np.exp(-4 * np.pi * x**2)
    
    norm2_sq = np.sum(P**2 * exp_2pi) * dx
    norm4_sq = np.sum(P**4 * exp_4pi) * dx
    
    norm2_sq_hat = np.sum(P_minus**2 * exp_2pi) * dx
    norm4_sq_hat = np.sum(P_minus**4 * exp_4pi) * dx
    
    if norm2_sq < 1e-10 or norm2_sq_hat < 1e-10:
        return 1e10
        
    obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
    
    return obj

def construct():
    start_time = time.time()
    deadline = 180.0
    
    # Prior state of the art
    x0 = np.array([-113 / 100, 1 / 25, 1 / 3240])
    seed_score = _compute_objective(x0)
    
    best_result = x0
    best_score = seed_score
    
    # Multi-start local optimization
    # We use L-BFGS-B which handles bounds and is efficient for smooth functions
    starts = [
        x0,
        x0 + np.array([0.1, 0.01, 0.0001]),
        x0 - np.array([0.1, 0.01, 0.0001]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-1.5, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002])
    ]
    
    for s in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(_compute_objective, s, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    return best_result
# EVOLVE-BLOCK-END
