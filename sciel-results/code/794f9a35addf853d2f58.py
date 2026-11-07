import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        d = self.dim
        budget = self.budget
        calls = [0]
        
        def wrapped(x):
            calls[0] += 1
            return float(func(x))
            
        def alive():
            return calls[0] < budget and time.time() < deadline

        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters for (1+1)
        c_sigma = 1.0 / (d + 3.0)
        c_c = (4.0 + d / 2.0) / (d + 4.0 + 2.0)
        c_1 = 2.0 / ((d + 1.7) ** 2 + 10.0)
        mu_1 = c_1
        chi_N = np.sqrt(d) * (1 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d ** 2))
        
        best_x = None
        best_f = np.inf
        
        # Initial points: Center and a few random ones
        # BBOB f11 is shifted, so center might not be great, but it's a start.
        # We will run CMA-ES from the best of these.
        init_candidates = [np.zeros(d)]
        for _ in range(5):
            init_candidates.append(rng.uniform(-4, 4, d))
            
        best_init_x = None
        best_init_f = np.inf
        for c in init_candidates:
            if not alive(): break
            c = np.clip(c, -5, 5)
            fc = wrapped(c)
            if fc < best_init_f:
                best_init_f = fc
                best_init_x = c.copy()
                
        if best_init_x is not None:
            best_x = best_init_x
            best_f = best_init_f
            
        # Run CMA-ES from best_init_x
        if alive() and best_x is not None:
            m = best_x.copy()
            f_m = best_f
            sigma = 2.0 # Initial step size
            C = np.eye(d)
            p_c = np.zeros(d)
            p_sigma = np.zeros(d)
            
            # Cholesky factor of C
            try:
                B = np.linalg.cholesky(C)
                D = np.ones(d)
            except:
                B = np.eye(d)
                D = np.ones(d)
                
            iter_count = 0
            no_improve_count = 0
            
            while alive():
                iter_count += 1
                z = rng.standard_normal(d)
                y = B @ (D * z)
                x_new = m + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                f_new = wrapped(x_new)
                
                if f_new < f_m:
                    # Success
                    f_m = f_new
                    m = x_new
                    
                    if f_m < best_f:
                        best_f = f_m
                        best_x = m.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        
                    # Update p_sigma
                    y_norm = np.linalg.norm(y)
                    if y_norm > 1e-16:
                        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma)) * (y_norm / sigma - chi_N) * (y / y_norm)
                    else:
                        p_sigma = (1 - c_sigma) * p_sigma
                        
                    # Update sigma
                    p_sigma_norm = np.linalg.norm(p_sigma)
                    sigma = sigma * np.exp(c_sigma / 5.0 * (p_sigma_norm / chi_N - 1.0))
                    sigma = np.clip(sigma, 1e-16, 100.0)
                    
                    # Update p_c
                    # h_sigma is a correction term, simplified to 1.0 for robustness
                    h_sigma = 1.0
                    p_c = (1 - c_c) * p_c + h_sigma * c_c * (y / sigma)
                    
                    # Update C
                    C = (1 - mu_1) * C + mu_1 * np.outer(y / sigma, y / sigma)
                    
                    # Ensure C is positive definite
                    try:
                        B = np.linalg.cholesky(C)
                        D = np.ones(d)
                    except:
                        # If C is not PD, reset to identity or regularize
                        C = np.eye(d)
                        B = np.eye(d)
                        D = np.ones(d)
                        sigma = 2.0
                        p_c = np.zeros(d)
                        p_sigma = np.zeros(d)
                        
                    # Stagnation check: if no global improvement for many iterations, restart
                    if no_improve_count > 100:
                        break
                else:
                    # Failure: no update to parameters in standard (1+1)
                    # But we might want to shrink sigma slightly to explore locally?
                    # Standard CMA-ES doesn't. We just try again.
                    pass
                    
                if not alive():
                    break

        return best_x, best_f
