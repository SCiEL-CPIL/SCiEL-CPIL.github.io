import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            x_arr = np.asarray(x, dtype=float)
            val = float(func(x_arr))
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = np.array(x_arr, dtype=float)
            return val

        # Phase 1: Minimal CMA-ES
        # Parameters
        lambda_pop = 4 * (np.log(self.dim) + 2)
        lambda_pop = int(lambda_pop)
        if lambda_pop < 10: lambda_pop = 10
        if lambda_pop > 20: lambda_pop = 20 # Keep small for budget
        
        mean = rng.uniform(-4, 4, self.dim)
        sigma = 2.0
        B = np.eye(self.dim)
        D = np.ones(self.dim)
        
        # CMA-ES update parameters
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1) - 1) - 1) + c_sigma
        
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        generations = 0
        max_generations = 50 # Safety
        
        while call_count < self.budget and time.time() < deadline - 20:
            if sigma < 1e-10: break
            
            # Sample
            population = []
            for i in range(lambda_pop):
                z = rng.standard_normal(self.dim)
                x = mean + sigma * (B @ (D * z))
                population.append(x)
            
            # Evaluate
            values = []
            for x in population:
                try:
                    f_val = safe_func(x)
                    values.append(f_val)
                except ValueError:
                    break
                except Exception:
                    values.append(np.inf)
            
            if len(values) < lambda_pop:
                break
                
            # Sort
            indices = np.argsort(values)
            sorted_pop = np.array([population[i] for i in indices])
            sorted_vals = np.array([values[i] for i in indices])
            
            # Update best
            if sorted_vals[0] < best_f:
                best_f = sorted_vals[0]
                best_x = sorted_pop[0].copy()
                
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
            
            # Update sigma
            y = (new_mean - mean) / (sigma * D)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (generations + 1)) / np.sqrt(1 - (1 - c_sigma) ** (generations + 1))) < (1.4 + 2 / (self.dim + 1)) * np.sqrt((mu_eff + 1) / (self.dim + 1))
            
            if h_sigma:
                p_c = (1 - c_c) * p_c + c_c * y
            
            # Update B and D (Eigen decomposition)
            # Simplified: Just update covariance matrix C
            # Actually, standard CMA-ES updates C.
            # C = (1 - c1 - cmu) * C + c1 * (p_c p_c^T + (h_sigma - 1 + c_c) p_c p_c^T) + cmu * sum(w_i (x_i - mean_old)(x_i - mean_old)^T)
            
            # For simplicity and speed, let's just do a simple adaptation of sigma and mean, 
            # or switch to L-BFGS if we've done enough global search.
            # A full CMA-ES implementation is ~50 lines.
            
            mean = new_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            generations += 1
            
            # If we have used 50% budget, break and go to local
            if call_count > self.budget // 2:
                break

        # Phase 2: L-BFGS-B
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 10:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': remaining, 'gtol': 1e-12, 'ftol': 1e-15}
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        # Phase 3: Polish with Nelder-Mead
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                if res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.array(res_nm.x, dtype=float)
            except Exception:
                pass

        if best_x is None:
            return np.zeros(self.dim), 0.0
        return best_x, best_f
