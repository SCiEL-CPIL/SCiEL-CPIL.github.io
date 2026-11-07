import numpy as np
import scipy.stats.qmc as qmc
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        used_calls = 0
        max_calls = self.budget
        
        def safe_func(x):
            nonlocal used_calls
            if used_calls >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            used_calls += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)
        
        # Phase 1: Small Sobol exploration to find a reasonable starting point
        n_initial = 16
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sobol.random(n_initial)
        initial_points = sobol_points * 10.0 - 5.0
        
        initial_values = []
        for p in initial_points:
            if used_calls >= max_calls or time.time() > deadline:
                break
            try:
                fp = safe_func(p)
                initial_values.append((fp, p))
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            except RuntimeError:
                break
        
        if best_x is None:
            return None, np.inf
        
        # Phase 2: CMA-ES style population-based optimization
        dim = self.dim
        pop_size = 10
        
        x0 = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        c_sigma = 1.0 / (pop_size + 3)
        c_c = (4 + dim / 3) / (dim + 4 + 2 * pop_size / dim)
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((dim + 2)**2 + mu_eff))
        
        remaining_budget = max_calls - used_calls
        max_generations = remaining_budget // pop_size
        
        for gen in range(max_generations):
            if used_calls >= max_calls or time.time() > deadline:
                break
            
            offspring = []
            offspring_vals = []
            
            for i in range(pop_size):
                if used_calls >= max_calls or time.time() > deadline:
                    break
                try:
                    z = rng.standard_normal(dim)
                    x_new = x0 + sigma * B @ (D * z)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = safe_func(x_new)
                    offspring.append(x_new)
                    offspring_vals.append(f_new)
                except RuntimeError:
                    break
            
            if len(offspring) < 2:
                break
            
            idx = np.argsort(offspring_vals)
            offspring = [offspring[i] for i in idx]
            offspring_vals = [offspring_vals[i] for i in idx]
            
            if offspring_vals[0] < best_f:
                best_f = offspring_vals[0]
                best_x = offspring[0].copy()
            
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (offspring[i] - x0) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
            
            x0 = np.clip(x0 + sigma * y_w, -5, 5)
            
            delta = 1.0 if (p_sigma @ p_sigma) / (1 - (1 - c_sigma)**(2 * (gen + 1))) < (1 + 2 / (dim + 1)) * c_sigma else 0.0
            C = (1 + c_1 * delta * (mu_eff - 1) - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            for i in range(mu):
                y_i = (offspring[i] - x0) / sigma
                C += c_mu * weights[i] * (np.outer(y_i, y_i) - C)
            
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
            
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            sigma *= np.exp(c_sigma / chi_n * (np.sqrt(p_sigma @ p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
        
        return best_x, best_f
