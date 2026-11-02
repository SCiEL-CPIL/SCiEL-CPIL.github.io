import numpy as np
import time
from scipy.optimize import minimize

# Hermite polynomials coefficients (standard physicists' definition)
# H_n(x) = sum_{k=0}^{n//2} (-1)^k * (2n-1)!! / (k! (n-2k)!) * (2x)^(n-2k)
# Or simply compute them for the specific degrees needed.

def H0(x):
    return np.ones_like(x)

def H4(x):
    return 16*x**4 - 48*x**2 + 12

def H8(x):
    return 256*x**8 - 3072*x**6 + 10752*x**4 - 10752*x**2 + 1680

def H12(x):
    return 4096*x**12 - 73728*x**10 + 483840*x**8 - 1397760*x**6 + 1720320*x**4 - 774144*x**2 + 40824

# Precompute grid for integration
# We need high accuracy. Use a wide range and fine grid.
x_grid = np.linspace(-15, 15, 10000)
dx = x_grid[1] - x_grid[0]

def compute_bound(c0, c4, c8):
    c12 = -(c0 + 12*c4 + 1680*c8) / 40824.0
    
    g = c0*H0(x_grid) + c4*H4(x_grid) + c8*H8(x_grid) + c12*H12(x_grid)
    g = g * np.exp(-x_grid**2 / 2)
    
    norm_sq = np.sum(g**2) * dx
    if norm_sq < 1e-12:
        return 1e100
        
    # A(f) is related to ||x f|| / ||f||
    # A(F) is related to ||f'|| / ||f|| (up to constants)
    # The product A(f)A(F) is minimized when we minimize (||x f|| ||f'||) / ||f||^2
    
    xf = x_grid * g
    norm_xf_sq = np.sum(xf**2) * dx
    
    # Derivative g'
    # g(x) = P(x) e^{-x^2/2}
    # g'(x) = (P'(x) - x P(x)) e^{-x^2/2}
    # We can compute P(x) and P'(x) analytically or numerically.
    # Numerical differentiation is risky near peaks, but with 10000 points it should be okay.
    # Better: Analytical derivative of the polynomial part.
    
    # P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # P'(x) = c4 H4' + c8 H8' + c12 H12' (H0' = 0)
    
    H4_prime = 64*x_grid**3 - 96*x_grid
    H8_prime = 2048*x_grid**7 - 18432*x_grid**5 + 43008*x_grid**3 - 21504*x_grid
    H12_prime = 49152*x_grid**11 - 737280*x_grid**9 + 3870720*x_grid**7 - 8386560*x_grid**5 + 6881280*x_grid**3 - 1548288*x_grid
    
    P_prime = c4*H4_prime + c8*H8_prime + c12*H12_prime
    P = c0*H0(x_grid) + c4*H4(x_grid) + c8*H8(x_grid) + c12*H12(x_grid)
    
    g_prime = (P_prime - x_grid * P) * np.exp(-x_grid**2 / 2)
    
    norm_gp_sq = np.sum(g_prime**2) * dx
    
    # The constant factor for the uncertainty product in this convention:
    # A(f) = ||x f|| / ||f||
    # A(F) = ||F^{-1}(xi \hat{f})|| / ||\hat{f}||
    # By Plancherel, ||\hat{f}|| = ||f|| (depending on normalization).
    # If \hat{f}(xi) = \int f(x) e^{-2\pi i x \xi} dx, then \frac{d}{dx} f(x) \leftrightarrow 2\pi i \xi \hat{f}(\xi).
    # So ||\xi \hat{f}|| = \frac{1}{2\pi} ||f'||.
    # Thus A(F) = \frac{1}{2\pi} \frac{||f'||}{||f||}.
    # Product = \frac{1}{2\pi} \frac{||x f|| ||f'||}{||f||^2}.
    
    bound = (1.0 / (2 * np.pi)) * (np.sqrt(norm_xf_sq * norm_gp_sq) / norm_sq)
    
    return bound

def construct():
    start_time = time.time()
    time_limit = 180.0
    safety_margin = 5.0
    
    best_score = 1e100
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    
    # Initial guesses
    # The seed values are a good starting point.
    # Also try some random perturbations.
    
    # Strategy: Nelder-Mead with multiple restarts.
    # Each restart should be fast enough to allow many tries within the time limit.
    
    def objective(vars):
        return compute_bound(vars[0], vars[1], vars[2])
    
    # Check time before starting heavy loops
    if time.time() - start_time > time_limit - safety_margin:
        return best_coeffs
        
    # Restart centers
    centers = [
        [-1.13, 0.04, 0.0003], # Seed
        [-1.0, 0.0, 0.0],
        [-1.5, 0.1, 0.001],
        [-0.5, -0.1, 0.0],
        [0.0, 0.0, 0.0],
        [-2.0, 0.2, 0.01],
        [-1.13, 0.04, 0.0003] + np.random.normal(0, 0.1, 3),
        [-1.13, 0.04, 0.0003] + np.random.normal(0, 0.5, 3),
    ]
    
    for center in centers:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            res = minimize(objective, center, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            continue
            
    # If we still have time, do a finer polish around the best solution
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res = minimize(objective, best_coeffs, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    return best_coeffs
