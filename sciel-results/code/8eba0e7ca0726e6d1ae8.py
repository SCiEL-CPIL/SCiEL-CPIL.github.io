import numpy as np
from scipy.special import eval_hermite
from scipy.integrate import quad
from scipy.optimize import minimize
import time

def construct():
    # Start time to enforce deadline
    start_time = time.time()
    DEADLINE = 180.0  # Safety margin before 210s kill

    # Hermite polynomials at 0 (Physicists' Hermite)
    # H_0(0) = 1
    # H_4(0) = -12
    # H_8(0) = 4032
    # H_12(0) = -1330560
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 4032.0
    H12_0 = -1330560.0

    def c12_from_c0_c4_c8(c0, c4, c8):
        # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
        # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def compute_uncertainty_product(c0, c4, c8):
        c12 = c12_from_c0_c4_c8(c0, c4, c8)
        coeffs = [c0, c4, c8, c12]
        degrees = [0, 4, 8, 12]

        # Define f(x) = exp(-x^2/2) * sum(c_k * H_k(x))
        def f_x(x):
            h_sum = 0.0
            for c, d in zip(coeffs, degrees):
                h_sum += c * eval_hermite(d, x)
            return np.exp(-x**2 / 2.0) * h_sum

        # Define |f(x)| for integration
        def abs_f_x(x):
            return np.abs(f_x(x))

        # Define |f(x)|^2 for L2 norm
        def f_sq_x(x):
            val = f_x(x)
            return val * val

        # Integration limits for f(x)
        # f(x) decays like x^12 * exp(-x^2/2). 
        # exp(-x^2/2) is negligible beyond ~10-15.
        # Let's use [-15, 15] for safety.
        L_F = 15.0
        
        # Calculate L2 norm squared
        norm_sq, _ = quad(f_sq_x, -L_F, L_F, limit=100, epsabs=1e-12, epsrel=1e-12)
        if norm_sq <= 1e-15:
            return 1e10 # Avoid division by zero
        
        norm_l2 = np.sqrt(norm_sq)
        
        # Calculate A(f) = L1 norm of normalized f
        # A(f_norm) = (1/norm_l2) * int |f(x)| dx
        l1_norm_raw, _ = quad(abs_f_x, -L_F, L_F, limit=100, epsabs=1e-12, epsrel=1e-12)
        A_f = l1_norm_raw / norm_l2

        # Calculate Fourier Transform
        # F(xi) = int f(x) exp(-i x xi) dx
        # Using analytic formula:
        # F[exp(-x^2/2) H_n(x)](xi) = sqrt(2*pi) * i^n * exp(-xi^2/2) * H_n(-i*xi)
        # We need int |F(xi)| dxi
        
        def abs_F_xi(xi):
            # Sum of terms c_k * sqrt(2*pi) * i^k * exp(-xi^2/2) * H_k(-i*xi)
            total = 0.0 + 0.0j
            sqrt_2pi = np.sqrt(2.0 * np.pi)
            exp_term = np.exp(-xi**2 / 2.0)
            
            for c, d in zip(coeffs, degrees):
                if c == 0:
                    continue
                # i^d
                i_pow = 1.0j ** d
                # H_d(-i*xi)
                # eval_hermite accepts complex arguments
                h_val = eval_hermite(d, -1.0j * xi)
                total += c * sqrt_2pi * i_pow * exp_term * h_val
            
            return np.abs(total)

        # Integration limits for F(xi)
        # F(xi) decays like xi^12 * exp(-xi^2/2).
        L_FX = 15.0
        
        A_F, _ = quad(abs_F_xi, -L_FX, L_FX, limit=100, epsabs=1e-12, epsrel=1e-12)
        
        # The product A(f) * A(F) for the normalized function
        # Note: The uncertainty product is scale invariant if we normalize L2=1.
        # A(f_norm) * A(F_norm) = A_f * A_F
        
        return A_f * A_F

    # Objective function for minimizer
    def objective(params):
        c0, c4, c8 = params
        try:
            val = compute_uncertainty_product(c0, c4, c8)
            if not np.isfinite(val):
                return 1e10
            return val
        except Exception:
            return 1e10

    # Initial guess from seed
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_val = objective(x0)
    best_params = x0.copy()

    # Optimization
    # Use Nelder-Mead or Powell. Powell is often good for this.
    # Check time periodically.
    
    # Try a few different methods or restarts if time permits
    methods = ['Nelder-Mead', 'Powell']
    
    for method in methods:
        if time.time() > DEADLINE - 30:
            break
            
        try:
            res = minimize(objective, x0, method=method, options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
        except Exception:
            pass

    # If time permits, try a local polish with L-BFGS-B around the best found
    if time.time() < DEADLINE - 10:
        try:
            res2 = minimize(objective, best_params, method='L-BFGS-B', options={'maxiter': 200})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Return the coefficients c0, c4, c8
    return np.array(best_params)
