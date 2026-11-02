import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 175.0  # Safety margin before 180s limit
    
    # Hermite polynomials (physicists' convention)
    def H0(x):
        return np.ones_like(x, dtype=float)
    
    def H4(x):
        return x**4 - 6*x**2 + 3
    
    def H8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def H12(x):
        return x**12 - 66*x**10 + 1260*x**8 - 9240*x**6 + 23100*x**4 - 27720*x**2 + 10395
    
    def H0_prime(x):
        return np.zeros_like(x, dtype=float)
    
    def H4_prime(x):
        return 4*x**3 - 12*x
        
    def H8_prime(x):
        return 8*x**7 - 168*x**5 + 840*x**3 - 840*x
        
    def H12_prime(x):
        return 12*x**11 - 660*x**9 + 10080*x**7 - 55440*x**5 + 92400*x**3 - 55440*x

    def P(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
        
    def P_prime(x, c0, c4, c8):
        c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
        return c0*H0_prime(x) + c4*H4_prime(x) + c8*H8_prime(x) + c12*H12_prime(x)

    # Setup Gauss-Hermite quadrature
    # n=20 is exact for polynomials up to degree 39. P(x)^2 is degree 24. P'(x)^2 is degree 22.
    # x^2 P(x)^2 is degree 26. So n=20 is sufficient and safe.
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(20)
    
    def compute_uncertainty_product(coeffs):
        c0, c4, c8 = coeffs
        
        # Check time
        if time.time() > deadline - 5:
            return np.inf
        
        try:
            # Evaluate P and P' at nodes
            P_vals = P(x_nodes, c0, c4, c8)
            P_prime_vals = P_prime(x_nodes, c0, c4, c8)
            
            # f(x) = P(x) * exp(-x^2/2)
            # |f(x)|^2 = P(x)^2 * exp(-x^2)
            # The quadrature weights for exp(-x^2) are w_nodes.
            # So integral of g(x) exp(-x^2) is sum(w_nodes * g(x_nodes))
            
            # norm_sq = integral |f|^2 dx = sum(w * P^2)
            norm_sq = np.sum(w_nodes * P_vals**2)
            if norm_sq <= 0 or not np.isfinite(norm_sq):
                return np.inf
            
            # x2_norm = integral x^2 |f|^2 dx = sum(w * x^2 * P^2)
            x2_norm = np.sum(w_nodes * (x_nodes**2) * P_vals**2)
            if not np.isfinite(x2_norm) or x2_norm < 0:
                return np.inf
            
            sigma_x2 = x2_norm / norm_sq
            
            # f'(x) = (P'(x) - x P(x)) exp(-x^2/2)
            # |f'(x)|^2 = (P'(x) - x P(x))^2 exp(-x^2)
            f_prime_factor = P_prime_vals - x_nodes * P_vals
            fprime_sq = np.sum(w_nodes * f_prime_factor**2)
            
            if not np.isfinite(fprime_sq) or fprime_sq < 0:
                return np.inf
            
            # sigma_xi^2 = (1/(4*pi^2)) * integral |f'|^2 dx / integral |f|^2 dx
            sigma_xi2 = fprime_sq / (4 * np.pi**2 * norm_sq)
            
            # Uncertainty product: sigma_x * sigma_xi
            product = np.sqrt(sigma_x2 * sigma_xi2)
            
            return product
            
        except Exception:
            return np.inf
    
    # Objective: minimize the uncertainty product sigma_x * sigma_xi
    def objective(coeffs):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 5:
            return np.inf
        return compute_uncertainty_product(coeffs)
    
    # Initial guess - try various starting points
    x0 = np.array([0.0, 0.0, 0.0])
    
    best_score = objective(x0)
    best_coeffs = x0.copy()
    
    # Try multiple initial guesses
    initial_guesses = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, -0.1, 0.01]),
        np.array([-1.0, 0.1, -0.01]),
        np.array([0.5, -0.05, 0.005]),
        np.array([2.0, -0.2, 0.02]),
        np.array([-0.5, 0.05, -0.005]),
        np.array([1.0, 0.1, 0.01]),
        np.array([-1.0, -0.1, -0.01]),
    ]
    
    rng = np.random.RandomState(42)
    
    # Evaluate initial guesses
    for guess in initial_guesses:
        if time.time() > deadline - 15:
            break
        score = objective(guess)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_coeffs = guess.copy()
    
    # Nelder-Mead optimization from multiple restarts
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        if restart < len(initial_guesses):
            x_start = initial_guesses[restart]
        elif restart < len(initial_guesses) + 20:
            # Random restarts around best
            scale = 0.3 * (1 + (restart - len(initial_guesses))/10.0)
            x_start = best_coeffs + rng.normal(0, scale, 3)
        else:
            # Wider random restarts
            x_start = rng.normal(0, 2.0, 3)
        
        try:
            res = opt.minimize(objective, x_start, method='Nelder-Mead', 
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 20:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 5000,
                                     'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
