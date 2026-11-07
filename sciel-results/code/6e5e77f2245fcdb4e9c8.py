"""CMA-ES with Nelder-Mead polish for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    CMA-ES optimizer with a final Nelder-Mead polish phase.
    Designed for BBOB f20 (Schwefel) on [-5,5]^5 with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # CMA-ES Parameters
        # For dim=5, lambda is typically 4 + floor(3 * ln(5)) = 10
        lam = 10
        mu = lam // 2  # 5
        
        # Initial mean and sigma
        # Start at origin, but Schwefel global min is at ~4.2096
        # A slightly larger initial sigma helps explore the domain
        mean = np.zeros(self.dim)
        sigma = 2.0
        
        # CMA-ES state
        C = np.eye(self.dim)
        C_inv = np.eye(self.dim)
        Cs = np.eye(self.dim)
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # Standard CMA-ES parameters
        cmu = 2 / ((self.dim + 1) ** 2 + 10)
        c_sigma = (self.dim + 1.3) / (self.dim + 3 + 2 * self.dim / (self.dim + 1))
        c_c = (4 + self.dim / self.dim) / (4 + self.dim + 2 * self.dim / (self.dim + 1))
        d_sigma = 1 + 2 * max(0, np.sqrt((self.dim - 1) / (self.dim + 1)) - 1) + cmu
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        
        # Precompute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)
        
        # CMA-ES Loop
        # Reserve ~50 evaluations for Nelder-Mead polish
        cma_budget = self.budget - 50
        
        while call_count < cma_budget:
            if time.time() > deadline:
                break
            
            # Generate lambda offspring
            y = []
            for i in range(lam):
                if call_count >= self.budget:
                    break
                # Cholesky decomposition of C
                L = np.linalg.cholesky(C)
                z = rng.standard_normal(self.dim)
                # x = mean + sigma * L @ z
                x = mean + sigma * (L @ z)
                # Clip to domain [-5, 5]
                x = np.clip(x, -5, 5)
                
                fx = func(x)
                call_count += 1
                y.append((fx, x))
                
            if len(y) < lam:
                break
                
            # Sort by fitness
            y.sort(key=lambda t: t[0])
            
            # Update best
            if y[0][0] < best_f:
                update_best(y[0][1], y[0][0])
            
            # Selection
            y_selected = [y[i][1] for i in range(mu)]
            f_selected = [y[i][0] for i in range(mu)]
            
            # Check for stagnation (all fitnesses identical)
            if len(set(f_selected)) == 1:
                # Restart with larger sigma or different mean if stuck
                # For simplicity, just break and let NM polish handle it, 
                # or nudge sigma. Given budget, breaking is safer to avoid waste.
                # However, a simple restart logic:
                if sigma > 0.1:
                    sigma *= 0.5
                else:
                    break
                continue

            # Weighted mean
            new_mean = np.sum(np.array(y_selected) * np.array(weights)[:, np.newaxis], axis=0)
            
            # Update path covariance matrix
            v = (new_mean - mean) / sigma
            pc = (1 - c_c) * pc + c_c * v
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (call_count // lam)))) < (1 + 2 / self.dim) * chi_n
            
            if hsig:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (new_mean - mean)
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + cmu * delta_hsig - cmu) * C + cmu * np.sum(
                weights[i] * np.outer(y_selected[i] - mean, y_selected[i] - mean) for i in range(mu)
            ) - cmu * delta_hsig * np.outer(pc, pc)
            
            # Ensure positive definite
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)
                C_inv = np.eye(self.dim)
                break

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5, 5)

        # Phase 2: Nelder-Mead Polish
        # Use the best point found so far
        if best_x is not None:
            x0 = best_x.copy()
            remaining_budget = self.budget - call_count
            
            if remaining_budget > 0:
                # Limit NM iterations to fit within time and budget
                # NM usually converges fast in 5D
                max_iter = min(remaining_budget, 100)
                
                # Use a wrapper to count calls and check time
                nm_call_count = 0
                def nm_func(x):
                    nonlocal nm_call_count
                    if time.time() > deadline:
                        return best_f # Return current best to stop early effectively
                    if nm_call_count >= remaining_budget:
                        return best_f
                    fx = func(x)
                    nm_call_count += 1
                    call_count += 1
                    update_best(x, fx)
                    return fx

                try:
                    res = minimize(nm_func, x0, method='Nelder-Mead', 
                                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
