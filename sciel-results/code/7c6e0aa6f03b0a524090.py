import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)
        
        def check_time():
            return (time.time() - start_time) < deadline

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if not check_time():
                return None
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
            return f_val

        def cma_es_run(x0, sigma0, max_evals):
            nonlocal calls_made, best_x, best_f
            
            if max_evals < 10 or calls_made >= budget:
                return

            lam = 5  # Smaller population for faster generations
            mu = 3
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)

            # CMA-ES parameters
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 + 1 / (4 * dim) - 1 / (21 * dim ** 2))
            
            evals_in_run = 0
            no_improve_count = 0
            best_f_local = best_f

            while evals_in_run < max_evals and calls_made < budget and check_time():
                # Check if we have enough budget for a full generation
                if budget - calls_made < lam:
                    break

                # Generate population
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * L @ z
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                offspring_f = []
                for y in offspring:
                    if calls_made >= budget or not check_time():
                        break
                    fy = evaluate(y)
                    if fy is None:
                        break
                    evals_in_run += 1
                    offspring_f.append(fy)

                if len(offspring_f) < lam:
                    break

                # Sort by fitness
                idx = np.argsort(offspring_f)
                y_sorted = [offspring[i] for i in idx]
                f_sorted = [offspring_f[i] for i in idx]

                # Update mean
                x_new_mean = np.sum([weights[i] * y_sorted[i] for i in range(mu)], axis=0)

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (L @ (x_new_mean - x_mean))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 0.005, 5.0)

                # Update covariance matrix
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new_mean - x_mean) / sigma

                # Standard CMA-ES covariance update
                C = ((1 + c_1 * (c_mu - 1) - c_mu) * C
                     + c_1 * (np.outer(p_c, p_c) - C)
                     + c_mu / c_1 * np.sum([
                         weights[i] * np.outer(y_sorted[i] - x_mean, y_sorted[i] - x_mean)
                         for i in range(mu)
                     ], axis=0))

                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C = C + 1e-10 * np.eye(dim)

                x_mean = x_new_mean

                # Check for stagnation
                if f_sorted[0] < best_f_local - 1e-8:
                    best_f_local = f_sorted[0]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 15:
                        break

            return x_mean, sigma

        # Phase 1: Global Exploration with diverse starts
        # Rastrigin is multimodal. We need to find good basins early.
        n_starts = 30
        for i in range(n_starts):
            if calls_made >= budget or not check_time():
                break
            if i == 0:
                x = np.zeros(dim)
            elif i < 15:
                x = rng.uniform(-4.5, 4.5, dim)
            else:
                # Try points near integers, as Rastrigin minima are near integers
                x = rng.uniform(-4.5, 4.5, dim)
                for d in range(dim):
                    if rng.random() < 0.6:
                        x[d] = round(x[d])
            evaluate(x)

        # Phase 2: Run CMA-ES from best point
        remaining = budget - calls_made
        if remaining > 20 and check_time() and best_f < float('inf'):
            x0 = best_x.copy()
            sigma0 = 2.0
            cma_es_run(x0, sigma0, remaining)

        # Phase 3: Restart with new random starts if budget remains
        remaining = budget - calls_made
        n_restarts = 0
        while remaining > 30 and check_time() and n_restarts < 5:
            # Try a few random points
            best_new_x = None
            best_new_f = float('inf')
            for _ in range(5):
                if calls_made >= budget or not check_time():
                    break
                x_new = rng.uniform(-4.5, 4.5, dim)
                f_new = evaluate(x_new)
                if f_new is not None and f_new < best_new_f:
                    best_new_f = f_new
                    best_new_x = x_new.copy()
            
            if best_new_x is not None and best_new_f < best_f:
                # Found a better point, refine it
                remaining = budget - calls_made
                if remaining > 20:
                    cma_es_run(best_new_x, 1.5, remaining)
                n_restarts += 1
            else:
                # No improvement, try another restart
                remaining = budget - calls_made
                if remaining < 30:
                    break
                n_restarts += 1

        return best_x, best_f
