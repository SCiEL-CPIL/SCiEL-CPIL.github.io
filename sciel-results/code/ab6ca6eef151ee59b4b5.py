import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # 10s safety margin

    # Hermite Polynomials (Physicists')
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 2520x^8 - 30240x^6 + 166320x^4 - 302400x^2 + 10395
    # H12(0) = 10395
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395

    def g(x, c0, c4, c8):
        c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
        x2 = x*x
        H0 = 1.0
        H4 = x2*x2 - 6.0*x2 + 3.0
        H8 = x2*x2*x2*x2 - 28.0*x2*x2*x2 + 210.0*x2*x2 - 420.0*x2 + 105.0
        H12 = x2*x2*x2*x2*x2*x2 - 84.0*x2*x2*x2*x2*x2 + 2520.0*x2*x2*x2*x2 - 30240.0*x2*x2*x2 + 166320.0*x2*x2 - 302400.0*x2 + 10395.0
        P = c0*H0 + c4*H4 + c8*H8 + c12*H12
        return P * np.exp(-x2 / 2.0)

    # Smooth max using log-sum-exp
    def smooth_max(vals, beta=100.0):
        """Approximate max using log-sum-exp with large beta"""
        v = vals * beta
        v_max = np.max(v)
        # Clip to avoid overflow
        v_shifted = v - v_max
        return (v_max + np.log(np.sum(np.exp(v_shifted)))) / beta

    def compute_max_g(c0, c4, c8, x_samples):
        """Compute the maximum of |g(x)| over x_samples"""
        vals = g(x_samples, c0, c4, c8)
        # We need max of |g(x)| <= 1
        return np.max(np.abs(vals))

    # Initial coarse grid
    x_coarse = np.linspace(-8, 8, 200)
    
    def objective(vars):
        c0, c4, c8 = vars
        return -c0

    def constraint(vars):
        c0, c4, c8 = vars
        # Use smooth max for gradient-friendly optimization
        vals = g(x_coarse, c0, c4, c8)
        # Constraint: max(|g(x)|) <= 1, so 1 - max(|g|) >= 0
        # Use smooth approximation
        abs_vals = np.abs(vals)
        smax = smooth_max(abs_vals, beta=50.0)
        return 1.0 - smax

    # Starting points - try various initial guesses
    starts = [
        [0.5, 0.1, 0.0],
        [1.0, 0.0, 0.0],
        [0.8, 0.05, 0.0],
        [1.2, -0.05, 0.0],
        [0.9, 0.0, 0.05],
        [1.1, 0.0, -0.05],
        [0.7, 0.02, 0.01],
        [1.0, 0.01, -0.01],
    ]
    
    best_result = None
    best_score = np.inf
    
    for x0 in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, x0, method='SLSQP', 
                           constraints=[{'type': 'ineq', 'fun': constraint}],
                           options={'maxiter': 200, 'ftol': 1e-10, 'iprint': 0})
            
            if res.success:
                c0_opt, c4_opt, c8_opt = res.x
                # Verify with fine grid
                x_fine = np.linspace(-10, 10, 2000)
                max_val = compute_max_g(c0_opt, c4_opt, c8_opt, x_fine)
                if max_val <= 1.0 + 1e-6:
                    score = 1.0 / c0_opt
                    if score < best_score:
                        best_score = score
                        best_result = res.x.copy()
        except Exception as e:
            pass

    # If we found a good result, do a polish pass with finer grid
    if best_result is not None and time.time() < deadline - 10:
        c0_cur, c4_cur, c8_cur = best_result
        # Use finer grid for polish
        x_polish = np.linspace(-10, 10, 5000)
        
        def constraint_polish(vars):
            c0, c4, c8 = vars
            vals = g(x_polish, c0, c4, c8)
            abs_vals = np.abs(vals)
            smax = smooth_max(abs_vals, beta=100.0)
            return 1.0 - smax

        try:
            res_polish = minimize(objective, best_result, method='SLSQP', 
                                  constraints=[{'type': 'ineq', 'fun': constraint_polish}],
                                  options={'maxiter': 300, 'ftol': 1e-12, 'iprint': 0})
            
            if res_polish.success:
                c0_p, c4_p, c8_p = res_polish.x
                # Final verification with very fine grid
                x_verify = np.linspace(-12, 12, 10000)
                max_val = compute_max_g(c0_p, c4_p, c8_p, x_verify)
                if max_val <= 1.0 + 1e-5:
                    score_p = 1.0 / c0_p
                    if score_p < best_score:
                        best_score = score_p
                        best_result = res_polish.x.copy()
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed if optimization fails
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    return np.array(best_result)
