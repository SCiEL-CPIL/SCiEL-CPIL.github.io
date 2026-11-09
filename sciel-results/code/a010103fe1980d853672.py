"""Optimizer for BBOB-style f13 (Sharp ridge) - CMA-ES with structured restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f13 with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget

        def elapsed():
            return time.time() - t0

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or elapsed() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial diverse sampling with Sobol sequence
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
            sobol.random(1)
            n_init = min(50, budget // 10)
            pts = sobol.random(n_init) * 10.0 - 5.0
            for i in range(n_init):
                if calls >= budget or elapsed() > deadline:
                    break
                try_func(pts[i])
        except Exception:
            # Fallback: random sampling
            rng = np.random.default_rng(42)
            n_init = min(50, budget // 10)
            for i in range(n_init):
                if calls >= budget or elapsed() > deadline:
                    break
                try_func(rng.uniform(-5, 5, n))

        # Phase 2: CMA-ES implementation
        def cma_es(x0, sigma0, pop_size, max_iter):
            nonlocal calls, best_f, best_x, elapsed, deadline, budget, n
            
            # CMA-ES parameters
            lam = pop_size
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Learning rates
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Initialize
            mean = x0.copy().astype(float)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            pop_history = []
            
            for iteration in range(max_iter):
                if calls >= budget or elapsed() > deadline:
                    break
                
                # Generate population
                population = []
                for i in range(lam):
                    if calls >= budget or elapsed() > deadline:
                        break
                    z = np.random.standard_normal(n)
                    x = mean + sigma * (B @ (D * z))
                    population.append(x)
                
                if len(population) < 2:
                    break
                
                # Evaluate population
                fitness = []
                for x in population:
                    f = try_func(x)
                    fitness.append(f)
                
                if len(fitness) < 2:
                    break
                
                # Sort by fitness
                idx = np.argsort(fitness)
                population = [population[i] for i in idx]
                fitness = [fitness[i] for i in idx]
                
                # Update best
                if fitness[0] < best_f:
                    best_f = fitness[0]
                    best_x = population[0].copy()
                
                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * population[i]
                
                # Update step size control
                y_mean = (new_mean - mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B.T @ (D * (D.T @ y_mean)))
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1 + 2 / (n + 1)) * np.sqrt(n + 1 - 1) else 0
                
                # Update covariance matrix
                pc = (1 - cs) * pc + hsig * cs * y_mean
                delta_h = (1 - hsig) * cs * (2 - cs)
                C = (1 + c1 * delta_h - c1 - cmu) * np.eye(n) + c1 * np.outer(pc, pc)
                for i in range(mu):
                    zi = B.T @ (D * (population[i] - mean) / sigma)
                    C += cmu * weights[i] * np.outer(zi, zi)
                
                # Eigendecomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
                
                # Update step size
                sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                mean = new_mean
                pop_history.append((mean.copy(), sigma))
            
            return best_f, best_x

        # Run CMA-ES from best point
        if best_f < np.inf and calls < budget and elapsed() < deadline:
            pop_size = 2 * (n + 1)
            max_iter = min(200, (budget - calls) // pop_size)
            if max_iter > 5:
                try:
                    _, _ = cma_es(best_x, 1.0, pop_size, max_iter)
                except Exception:
                    pass

        # Phase 3: Multi-start CMA-ES from diverse points
        if calls < budget - 50 and elapsed() < deadline:
            rng = np.random.default_rng(789)
            n_starts = min(8, (budget - calls) // 60)
            
            for s in range(n_starts):
                if calls >= budget - 30 or elapsed() > deadline:
                    break
                
                # Different starting points
                if s == 0:
                    x0 = best_x.copy()
                    sigma0 = 0.5
                elif s == 1:
                    x0 = best_x + rng.standard_normal(n) * 2.0
                    sigma0 = 1.5
                elif s == 2:
                    x0 = rng.uniform(-3, 3, n)
                    sigma0 = 2.0
                elif s == 3:
                    x0 = best_x + rng.standard_normal(n) * 0.5
                    sigma0 = 0.3
                elif s == 4:
                    x0 = rng.uniform(-4, 4, n)
                    sigma0 = 3.0
                elif s == 5:
                    x0 = best_x + rng.standard_normal(n) * 3.0
                    sigma0 = 2.5
                elif s == 6:
                    x0 = np.zeros(n) + rng.standard_normal(n) * 0.2
                    sigma0 = 0.4
                else:
                    x0 = best_x + rng.standard_normal(n) * 1.0
                    sigma0 = 0.8
                
                x0 = np.clip(x0, -5.0, 5.0)
                
                pop_size = max(10, 2 * (n + 1))
                max_iter = min(50, (budget - calls) // (pop_size + 5))
                if max_iter > 3:
                    try:
                        _, _ = cma_es(x0, sigma0, pop_size, max_iter)
                    except Exception:
                        pass

        # Phase 4: Fine-tuning with smaller CMA-ES
        if calls < budget - 20 and elapsed() < deadline:
            pop_size = max(8, n + 3)
            max_iter = min(30, (budget - calls) // (pop_size + 2))
            if max_iter > 3:
                try:
                    _, _ = cma_es(best_x, 0.1, pop_size, max_iter)
                except Exception:
                    pass

        # Phase 5: Final coordinate-wise polishing
        if calls < budget - 10 and elapsed() < deadline:
            rng = np.random.default_rng(123)
            for dim_idx in range(n):
                if calls >= budget or elapsed() > deadline:
                    break
                
                # Adaptive step sizes
                step_sizes = [0.01, -0.01, 0.005, -0.005, 0.001, -0.001, 0.0005, -0.0005]
                for delta in step_sizes:
                    if calls >= budget or elapsed() > deadline:
                        break
                    
                    x_new = best_x.copy()
                    x_new[dim_idx] += delta
                    x_new = np.clip(x_new, -5.0, 5.0)
                    
                    f_new = try_func(x_new)
                    if f_new < best_f:
                        continue
                    else:
                        break

        # Phase 6: Final CMA-ES polish with very small sigma
        if calls < budget - 15 and elapsed() < deadline:
            pop_size = max(6, n + 2)
            max_iter = min(20, (budget - calls) // (pop_size + 1))
            if max_iter > 2:
                try:
                    _, _ = cma_es(best_x, 0.01, pop_size, max_iter)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
