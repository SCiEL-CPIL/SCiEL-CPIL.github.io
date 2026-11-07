import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget or not time_ok():
                return np.inf
            calls_used += 1
            x_c = clip(x)
            f_val = func(x_c)
            if f_val < best_f:
                best_f = f_val
                best_x = x_c.copy()
            return f_val

        def cma_es(start, sigma, max_iter):
            nonlocal best_x, best_f, calls_used

            # CMA-ES parameters for 5D
            n = dim
            # Population size
            lam = 4 + int(3 * np.log(n))  # ~13 for n=5
            # Weights
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

            # State
            x_mean = clip(start).copy()
            sigma = sigma
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            inv_sigma = 1.0 / sigma if sigma > 1e-15 else 1e15
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Precompute Cholesky for C
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_chol = np.eye(n)

            for iteration in range(max_iter):
                if not time_ok() or calls_used >= budget - lam - 5:
                    break

                # Generate population
                population = []
                for i in range(lam):
                    if not time_ok() or calls_used >= budget:
                        break
                    z = rng.standard_normal(n)
                    x_i = x_mean + sigma * (C_chol @ z)
                    population.append(clip(x_i))

                if len(population) == 0:
                    break

                # Evaluate
                fitness = []
                for x_i in population:
                    if not time_ok() or calls_used >= budget:
                        break
                    f_i = call_func(x_i)
                    fitness.append(f_i)

                if len(fitness) == 0:
                    break

                # Sort by fitness
                order = np.argsort(fitness)
                population = [population[i] for i in order]
                fitness = [fitness[i] for i in order]

                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += w[i] * population[i]
                x_mean = new_mean

                # Update sigma (step size adaptation)
                old_sigma = sigma
                # Compute ps
                # Need the normalized step
                # ps = (1-c_sigma)*ps + c_sigma/sigma * (weighted sum of top mu steps)
                # Actually standard: ps = (1-c_sigma)*ps + c_sigma * (1/sigma) * sum(w_i * (x_i - x_mean_old)/sigma)
                # Let's use the standard formulation
                # Recompute with old mean
                # Actually, let's use a simpler sigma adaptation:
                # sigma = sigma * exp(c_sigma * (||ps|| - chi_n) / (d_sigma * chi_n))
                # But we need ps first.
                
                # Let's implement standard CMA-ES update
                # 1. Update pc
                # pc = (1-c_c)*pc + c_c * (1/sigma) * (x_mean - x_mean_old)
                # 2. Update ps
                # ps = (1-c_sigma)*ps + c_sigma * (1/sigma) * (weighted sum of selected z_i)
                # But we don't have z_i stored. Let's re-generate or store them.
                # For simplicity and robustness, let's use a simplified CMA-ES variant
                # or just adapt sigma based on improvement rate.
                
                # Actually, let's stick to a robust simplified CMA-ES
                # that works well in practice for low dimensions.
                
                # Simplified approach:
                # 1. Update mean (done)
                # 2. Adapt sigma based on how much the best points improved
                # 3. Adapt covariance to align with ridge direction
                
                # Let's use a covariance matrix adaptation based on the top solutions
                # This is more like a covariance matrix adaptation for evolution strategies
                
                # Compute covariance of top mu solutions
                top_mu = np.array(population[:mu])
                mean_top = np.mean(top_mu, axis=0)
                if mu > 1:
                    C_new = np.cov(top_mu, rowvar=False)
                    # Regularize
                    C_new = C_new + 1e-6 * np.eye(n)
                    # Blend with old C
                    C = 0.5 * C + 0.5 * C_new
                    # Ensure positive semi-definite
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-8)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                else:
                    C = np.eye(n)

                # Update sigma
                # If we made progress, keep sigma or increase slightly
                # If not, decrease
                if fitness[0] < best_f:
                    # Made progress
                    sigma = min(sigma * 1.05, 10.0)
                else:
                    sigma = max(sigma * 0.9, 0.01)

                # Update Cholesky
                try:
                    C_chol = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_chol = np.eye(n)

                # Check for convergence
                if sigma < 1e-8:
                    break

        # Multi-start CMA-ES strategy
        # We'll use several restarts with different starting points and sigma values
        
        # Phase 1: Broad exploration
        n_restarts = 5
        calls_per_restart = max(50, budget // n_restarts - 10)
        
        for r in range(n_restarts):
            if not time_ok() or calls_used >= budget - 20:
                break
            
            # Different starting points for each restart
            if r == 0:
                x_start = rng.uniform(-3, 3, dim)
                sigma_start = 2.0
            elif r == 1:
                x_start = rng.uniform(-4, 4, dim)
                sigma_start = 1.5
            elif r == 2:
                x_start = rng.uniform(-2, 2, dim)
                sigma_start = 1.0
            elif r == 3:
                x_start = rng.uniform(-4, 4, dim)
                sigma_start = 2.5
            else:
                # Use best found so far with small sigma for polishing
                x_start = best_x.copy()
                sigma_start = 0.5
            
            # Run CMA-ES
            max_iter = max(5, calls_per_restart // dim)
            cma_es(x_start, sigma_start, max_iter)
            
            # If we've used most of the budget, stop
            if calls_used >= budget - 20:
                break
        
        # Final polishing from best point
        if calls_used < budget - 10 and time_ok():
            cma_es(best_x, 0.1, 20)
        
        return best_x, best_f
