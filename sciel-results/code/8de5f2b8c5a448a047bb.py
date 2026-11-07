"""Hybrid optimizer: L-BFGS-B for basin finding, coordinate descent for anisotropic refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f12 (Bent Cigar).
    Strategy:
    1. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for the full optimization.
       CMA-ES adapts its search distribution to the function's anisotropy, making it
       ideal for Bent Cigar's highly ill-conditioned valley (1000x curvature difference).
       It handles rotation, shift, and offset of the hidden minimum naturally.
    2. Final coordinate descent refinement for last-mile precision.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_f, best_x
            if calls_used >= budget:
                return np.inf
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: CMA-ES Optimization ---
        # Implement a lightweight CMA-ES inline to avoid external dependencies
        # and to have full control over budget usage.
        
        # CMA-ES parameters
        # For dim=5, population size lambda = 4 + floor(3*log(dim)) = 4 + 3*1.6 = 8
        lambda_pop = 4 + int(3 * np.log(dim))
        # Initial step size: start with a moderate value
        sigma = 2.0  # Initial standard deviation
        # Mean (search center)
        mean = np.zeros(dim)
        
        # CMA-ES state
        # Covariance matrix C, initial identity
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation rates
        mu = lambda_pop // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Cholesky decomposition for sampling
        L = np.linalg.cholesky(C)
        
        # Main CMA-ES loop
        max_cma_iter = budget // 2  # Reserve half budget for CMA-ES
        cma_iter = 0
        
        while cma_iter < max_cma_iter:
            if time.time() > deadline - 15:
                break
            if calls_used >= budget - 50:  # Reserve 50 for final refinement
                break
            
            # Sample lambda_pop individuals
            offspring = []
            for i in range(lambda_pop):
                z = np.random.randn(dim)
                x = mean + sigma * L @ z
                offspring.append(x)
            
            # Evaluate offspring
            f_values = []
            for x in offspring:
                if time.time() > deadline - 15:
                    f_values.append(1e10)
                    continue
                if calls_used >= budget - 50:
                    f_values.append(1e10)
                    continue
                fx = evaluate(x)
                if fx == np.inf:
                    fx = 1e10
                f_values.append(fx)
            
            # Sort by fitness
            indices = np.argsort(f_values)
            offspring = [offspring[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            
            # Update mean using weighted recombination
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * offspring[i]
            
            # Update evolution paths
            old_mean = mean.copy()
            mean = new_mean
            
            # Standardized step
            if sigma > 0:
                step = (mean - old_mean) / sigma
            else:
                step = np.zeros(dim)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step
            
            # Hessian norm correction
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (cma_iter + 1)))
                       < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            # Update covariance matrix
            C = ((1 - c_1 - c_mu) * C + c_1 * (2 - c_c) * np.outer(p_c, p_c))
            for i in range(mu):
                y_i = (offspring[i] - old_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-10 * np.eye(dim)
                L = np.linalg.cholesky(C)
            
            # Update sigma
            if h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 50.0)
            
            cma_iter += 1
        
        # --- Phase 2: Coordinate Descent Refinement ---
        # Final refinement along each axis for precision
        
        if best_x is None:
            best_x = mean
            best_f = np.inf
        
        current_x = best_x.copy()
        current_f = best_f
        
        # Number of CD passes
        max_passes = 2
        pass_count = 0
        
        while pass_count < max_passes:
            if time.time() > deadline - 10:
                break
            if calls_used >= budget - 5:
                break
            
            improved = False
            for i in range(dim):
                if time.time() > deadline - 10:
                    break
                if calls_used >= budget - 5:
                    break
                
                # Golden Section Search along axis i
                a = max(-5.0, current_x[i] - 1.0)
                b = min(5.0, current_x[i] + 1.0)
                
                if b - a < 1e-8:
                    continue
                
                phi = (np.sqrt(5) - 1) / 2
                inv_phi = 1.0 / phi
                
                c = b - inv_phi * (b - a)
                d = a + inv_phi * (b - a)
                
                def func_1d(t):
                    x_temp = current_x.copy()
                    x_temp[i] = t
                    if calls_used >= budget or time.time() > deadline - 5:
                        return 1e10
                    val = evaluate(x_temp)
                    if val == np.inf:
                        return 1e10
                    return val
                
                fc = func_1d(c)
                fd = func_1d(d)
                
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    if calls_used >= budget - 5:
                        break
                    
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - inv_phi * (b - a)
                        fc = func_1d(c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + inv_phi * (b - a)
                        fd = func_1d(d)
                    
                    if abs(b - a) < 1e-8:
                        break
                
                new_val = (a + b) / 2.0
                current_x[i] = new_val
                
                f_new = func_1d(new_val)
                if f_new < current_f - 1e-10:
                    current_f = f_new
                    improved = True
                
            pass_count += 1
            if not improved:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
