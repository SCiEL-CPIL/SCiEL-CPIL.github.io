"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        
        # Initial population
        pop_size = 4 + int(3 * np.sqrt(dim))  # ~11 for dim=5
        x = np.zeros(dim)
        sigma = 2.5
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Precompute eigen decomposition for C
        eigvals, eigvecs = np.linalg.eigh(C)
        sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        # Population
        z = rng.standard_normal((pop_size, dim))
        y = z @ sqrt_C.T
        x_pop = np.array([np.clip(x + sigma * y[i], -5, 5) for i in range(pop_size)])
        f_pop = np.array([func(x_pop[i]) for i in range(pop_size)])
        
        best_idx = np.argmin(f_pop)
        best_x = x_pop[best_idx].copy()
        best_f = f_pop[best_idx]
        
        calls_made = pop_size
        updates = 0
        no_improve = 0
        stagnation_threshold = 15
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Sort by fitness
            order = np.argsort(f_pop)
            m = pop_size // 2  # number of parents
            parents = order[:m]
            
            # Weighted recombination
            w = np.log(m + 0.5) - np.log(np.arange(1, m + 1))
            w /= w.sum()
            w = w * (pop_size / (m + 1))  # scaling for Rastrigin-like problems
            
            # Mean update
            y_parents = y[parents]
            mu_new = x + sigma * (w @ (y_parents @ inv_sqrt_C.T))
            
            # Covariance matrix adaptation
            v = (y_parents.T @ w).T  # (m, dim)
            delta = (mu_new - x) / sigma
            C = (1 - 2 * (1 - np.prod(eigvals ** 0.5)) / pop_size) * C + \
                (2 * (1 - np.prod(eigvals ** 0.5)) / pop_size) * C + \
                np.outer(delta, delta) - C * (1 - 2 * (1 - np.prod(eigvals ** 0.5)) / pop_size)
            
            # Simplified: just use rank-one update
            C = (1 - 0.1) * C + 0.1 * np.outer(delta, delta)
            C = (C + C.T) / 2  # ensure symmetry
            
            # Recompute eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, 1e10)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Step size adaptation (1/5 rule style)
            success = np.sum(f_pop[parents] < f_pop[parents[-1:]])
            if success >= m * 0.5:
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = np.clip(sigma, 0.01, 10.0)
            
            x = mu_new
            
            # Generate new population
            z = rng.standard_normal((pop_size, dim))
            y = z @ sqrt_C.T
            x_pop = np.array([np.clip(x + sigma * y[i], -5, 5) for i in range(pop_size)])
            f_pop = np.array([func(x_pop[i]) for i in range(pop_size)])
            calls_made += pop_size
            
            # Track best
            cur_best_idx = np.argmin(f_pop)
            if f_pop[cur_best_idx] < best_f:
                best_x = x_pop[cur_best_idx].copy()
                best_f = f_pop[cur_best_idx]
                no_improve = 0
            else:
                no_improve += 1
            
            # Restart if stagnating
            if no_improve >= stagnation_threshold:
                # Perturb from best
                x = best_x + sigma * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                sigma = max(sigma * 1.5, 0.5)
                no_improve = 0
                updates += 1
                
                # Regenerate population around new x
                z = rng.standard_normal((pop_size, dim))
                y = z @ sqrt_C.T
                x_pop = np.array([np.clip(x + sigma * y[i], -5, 5) for i in range(pop_size)])
                f_pop = np.array([func(x_pop[i]) for i in range(pop_size)])
                calls_made += pop_size
        
        return best_x, best_f
# EVOLVE-BLOCK-END
