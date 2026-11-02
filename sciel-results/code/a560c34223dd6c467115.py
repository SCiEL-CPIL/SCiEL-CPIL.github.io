import numpy as np
import scipy.optimize as opt
import scipy.integrate as integrate
import time

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin before 180s limit (210s kill switch)
    
    # Hermite polynomials (physicists' convention)
    def H0(x):
        return np.ones_like(x, dtype=float)
    
    def H4(x):
        return x**4 - 6*x**2 + 3
    
    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def poly(x, c0, c4, c8):
        # Constraint: P(0) = 0
        # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
        # c0*1 + c4*3 + c8*105 + c12*10395 = 0
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    def f_of_x(x, c0, c4, c8):
        return poly(x, c0, c4, c8) * np.exp(-x**2 / 2.0)

    # Precompute Gauss-Hermite quadrature nodes and weights for fast evaluation
    # Integral of |P(x)| exp(-x^2/2) dx
    # = Integral of |P(x)| exp(x^2/2) exp(-x^2) dx
    # Gauss-Hermite approximates Integral f(x) exp(-x^2) dx
    # So f(x) = |P(x)| exp(x^2/2)
    n_gh = 150  # Increase nodes for better accuracy in fast phase
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_gh)
    
    def objective_fast(coeffs):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf
        try:
            p_vals = poly(x_gh, c0, c4, c8)
            abs_f = np.abs(p_vals) * np.exp(x_gh**2 / 2.0)
            integral = np.sum(w_gh * abs_f)
            # The seed minimizes (2 * integral)^2
            return (2 * integral) ** 2
        except Exception:
            return np.inf

    def objective_high_precision(coeffs):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 2:
            return np.inf
        try:
            def abs_f(x):
                return abs(f_of_x(x, c0, c4, c8))
            # Integrate from 0 to inf and double, since f is even
            val, _ = integrate.quad(abs_f, 0, np.inf, limit=500, epsabs=1e-12, epsrel=1e-12)
            return (2 * val) ** 2
        except Exception:
            return np.inf

    # Initial guess from seed
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    best_score = objective_high_precision(x0)
    best_coeffs = x0.copy()

    # Phase 1: Global Search using Differential Evolution
    # This is structurally different from Nelder-Mead restarts.
    # It explores the space more thoroughly.
    if time.time() < deadline - 30:
        try:
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
            # Use a seed for reproducibility
            de_res = opt.differential_evolution(
                objective_fast, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-6,
                popsize=10,
                polish=True  # Polish with L-BFGS-B or Nelder-Mead internally
            )
            if np.isfinite(de_res.fun) and de_res.fun < best_score:
                # Verify with high precision before updating best
                hp_score = objective_high_precision(de_res.x)
                if hp_score < best_score:
                    best_score = hp_score
                    best_coeffs = de_res.x.copy()
        except Exception:
            pass

    # Phase 2: Local Refinement with Nelder-Mead using high precision
    # Multiple restarts around the best found so far
    rng = np.random.RandomState(42)
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x_start = best_coeffs
        else:
            # Random perturbations
            scale = 0.1 * (1.0 / (restart + 1.0))
            x_start = best_coeffs + rng.normal(0, scale, 3)
        
        try:
            res = opt.minimize(objective_high_precision, x_start, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 200,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
            
    return np.array(best_coeffs)
