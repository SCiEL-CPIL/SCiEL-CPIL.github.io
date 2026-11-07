"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts on BBOB f9 (rotated Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        max_calls = self.budget
        dim = self.dim
        
        def run_cma(x0, sigma0, remaining_budget):
            nonlocal total_calls, best_x, best_f
            
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = (2 + 2 * (mu_eff - 2) / (n + 2)) / ((n + 2) * (n + 2) + mu_eff)
            c_mu = (2 * (mu_eff - 2 + 2 / (mu_eff - 1))) / ((n + 2) * (n + 2) + mu_eff)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            
            # Initial eval
            f_mean = func(x_mean)
            total_calls += 1
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()
            
            calls_used = 1
            best_f_local = f_mean
            stagnation = 0
            
            while calls_used < remaining_budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append(y)
                
                if len(offspring) < lam:
                    break
                
                # Evaluate
                offspring_f = []
                for y in offspring:
                    if time.time() > deadline:
                        break
                    fy = func(y)
                    total_calls += 1
                    calls_used += 1
                    offspring_f.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                if len(offspring_f) < lam:
                    break
                
                # Sort
                indices = np.argsort(offspring_f)
                offspring_sorted = [offspring[i] for i in indices]
                offspring_f_sorted = [offspring_f[i] for i in indices]
                
                # Update best local
                if offspring_f_sorted[0] < best_f_local:
                    best_f_local = offspring_f_sorted[0]
                    stagnation = 0
                else:
                    stagnation += 1
                
                # Update mean
                x_mean_new = np.sum([weights[i] * offspring_sorted[i] for i in range(mu)], axis=0)
                
                # Update sigma
                z = np.linalg.norm(x_mean_new - x_mean) / (sigma * chi_n) if sigma > 1e-15 else 0
                sigma *= np.exp(c_sigma / d_sigma * (z - chi_n))
                sigma = np.clip(sigma, 1e-12, 100.0)
                
                # Update evolution paths
                v = (x_mean_new - x_mean) / sigma if sigma > 1e-15 else np.zeros(n)
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * v
                pc_norm = np.linalg.norm(pc)
                delta_h = 1 if pc_norm <= (1.4 + 1 / c_c) * chi_n else 0
                
                ps = (1 - c_sigma) * ps + delta_h * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
                
                # Update covariance
                C = (1 + c_1 * (2 - c_1) - c_mu) * C
                C += c_1 * (pc * pc.T - (1 - c_c) * np.eye(n))
                for i in range(mu):
                    d = (offspring_sorted[i] - x_mean) / sigma
                    C += c_mu * weights[i] * np.outer(d, d)
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Re-derive B, D
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except:
                    B = np.eye(n)
                    D = np.ones(n)
                
                x_mean = x_mean_new
                
                # Stagnation check
                if stagnation > 15 * lam:
                    break
                
                # Convergence check
                if sigma < 1e-10:
                    break
            
            return best_x, best_f
        
        # Multi-restart strategy
        n_restarts = 3
        for r in range(n_restarts):
            if total_calls >= max_calls:
                break
            if time.time() > deadline:
                break
            
            if r == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            elif r == 1:
                x0 = rng.uniform(-3, 3, dim)
                sigma0 = 1.5
            else:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 1.0
            
            remaining = max_calls - total_calls
            if remaining <= 0:
                break
            
            run_cma(x0, sigma0, remaining)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
