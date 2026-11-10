import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import time

def rudin_problem10() -> np.ndarray:
    """
    Finds coefficients for a degree-10 spherical harmonic that minimizes
    the L4/L2 norm ratio.
    """
    k = 10
    n_m = 2 * k + 1
    
    # 1. Setup Quadrature Grid
    # 70-point Gauss-Legendre in cos(theta)
    mu, w_mu = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi
    
    # Create meshgrid
    mu_grid, phi_grid = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_grid.ravel()
    phi_flat = phi_grid.ravel()
    
    # Weights for the discrete integral
    # Integral approx = sum( f^p * w_mu_i * d_phi_j )
    # w_flat has shape (70*130,)
    w_flat = np.repeat(w_mu, n_phi) * d_phi
    
    # 2. Precompute Spherical Harmonics Y_m
    # Y shape: (n_points, n_m)
    n_points = len(mu_flat)
    Y = np.zeros((n_points, n_m), dtype=np.float64)
    
    # Factorials for normalization
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # Use log-gamma for stability if needed, but 10! is small.
    from math import factorial, pi
    
    for idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        # Condon-Shortley phase is included in lpmv
        # lpmv(m, l, x) returns P_l^m(x) with (-1)^m
        P = lpmv(abs_m, k, mu_flat)
        
        # Normalization constant N_m
        # (k-abs_m)! / (k+abs_m)!
        num = factorial(k - abs_m)
        den = factorial(k + abs_m)
        N_m = np.sqrt((2 * k + 1) / (4 * pi) * (num / den))
        
        if m == 0:
            Y[:, idx] = N_m * P
        elif m > 0:
            Y[:, idx] = np.sqrt(2) * N_m * P * np.cos(m * phi_flat)
        else:
            Y[:, idx] = np.sqrt(2) * N_m * P * np.sin(abs_m * phi_flat)

    # 3. Define Objective Function
    # Minimize log(R) = 0.25 * log(||f||_4^4) - 0.5 * log(||f||_2^2)
    # ||f||_4^4 = sum(w * f^4)
    # ||f||_2^2 = sum(w * f^2)
    
    def objective(c):
        f = Y @ c
        # Numerical safety
        # f^2 can be negative due to floating point errors? No, f is real.
        # w is positive.
        sum_f2 = np.sum(w_flat * f**2)
        if sum_f2 < 1e-300:
            return 0.0
        sum_f4 = np.sum(w_flat * f**4)
        if sum_f4 < 1e-300:
            return 0.0
            
        # log(R)
        return 0.25 * np.log(sum_f4) - 0.5 * np.log(sum_f2)

    # 4. Optimization Loop
    best_score = np.inf
    best_c = np.ones(n_m)
    
    start_time = time.time()
    time_limit = 590  # Safety margin
    
    # Initial guesses
    guesses = [np.ones(n_m)]
    rng = np.random.default_rng(42)
    for _ in range(50):
        guesses.append(rng.normal(size=n_m))
        
    # Shuffle guesses to vary the order
    np.random.shuffle(guesses)
    
    for c0 in guesses:
        if time.time() - start_time > time_limit:
            break
            
        # Normalize initial guess to unit norm to avoid extreme scales
        norm_c0 = np.linalg.norm(c0)
        if norm_c0 > 1e-10:
            c0 = c0 / norm_c0
            
        try:
            res = minimize(objective, c0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success or res.fun < best_score:
                if np.isfinite(res.fun):
                    best_score = res.fun
                    best_c = res.x
        except Exception:
            continue

    # 5. Final Processing
    # Ensure validity: max|c| between 1e-3 and 1e5
    # Scale best_c
    max_c = np.max(np.abs(best_c))
    if max_c == 0:
        best_c = np.ones(n_m)
        max_c = 1.0
        
    # Target scale: 1.0 is safe
    if max_c < 1e-3 or max_c > 1e5:
        best_c = best_c / max_c
        # If it's still out of bounds (shouldn't be), adjust
        max_c_new = np.max(np.abs(best_c))
        if max_c_new < 1e-3:
             best_c = best_c * (1e-1 / max_c_new)
        elif max_c_new > 1e5:
             best_c = best_c * (1e4 / max_c_new)

    return best_c
