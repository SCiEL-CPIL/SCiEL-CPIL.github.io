"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer with CMA-ES and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        # Track best solution
        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f
                return True
            return False

        def time_left():
            return deadline - time.time() > 1.0

        def calls_left():
            return budget - calls

        rng = np.random.default_rng(42)

        # --- Strategy 1: CMA-ES with population-based search ---
        # CMA-ES adapts covariance to landscape shape, crucial for non-separable functions
        def run_cmaes(x0, sigma0, max_calls):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_left():
                return

            # CMA-ES parameters
            lambda_pop = max(4, int(np.ceil(4 + 3 * np.log(dim))))
            mu = lambda_pop // 2
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            # State
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)

            def sample_population():
                nonlocal B, D, sigma
                zs = rng.standard_normal((lambda_pop, dim))
                # Transform: z -> B * D * z
                ys = zs @ np.diag(D) @ B.T
                return x_mean + sigma * ys

            for gen in range(max_calls // lambda_pop + 1):
                if calls >= budget or not time_left():
                    break

                # Sample population
                pop = sample_population()
                pop = np.clip(pop, -5, 5)

                # Evaluate
                fvals = np.array([func(x) for x in pop])
                calls += lambda_pop
                for x, f in zip(pop, fvals):
                    update_best(x, f)

                # Sort by fitness
                order = np.argsort(fvals)
                pop_sorted = pop[order]
                fvals_sorted = fvals[order]

                # Update mean
                x_new_mean = sum(w * x for w, x in zip(weights, pop_sorted[:mu]))

                # Update step size (CSA)
                y_mean = (x_new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update correlation matrix (CMSA)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                C = (1 - c_c) * C + c_c * (np.outer(y_mean, y_mean) + (np.linalg.norm(p_c) ** 2 / (dim + 4) - 1) * C)
                # Ensure symmetry and positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.diag(D))

                x_mean = x_new_mean
                x_mean = np.clip(x_mean, -5, 5)

            # Final evaluation of mean
            if calls < budget and time_left():
                fm = func(x_mean)
                calls += 1
                update_best(x_mean, fm)

        # Initial CMA-ES from center
        run_cmaes(np.zeros(dim), 2.0, 200)

        # --- Strategy 2: (1+1)-ES from best found ---
        if time_left() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            steps = min(50, budget - calls)
            for _ in range(steps):
                if not time_left() or calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                update_best(y, fy)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        # --- Strategy 3: Coordinate-wise line search refinement ---
        if time_left() and calls < budget:
            x4 = best_x.copy()
            fx4 = best_f
            for coord in range(dim):
                if not time_left() or calls >= budget:
                    break
                for direction in [1, -1]:
                    if not time_left() or calls >= budget:
                        break
                    step = 0.5
                    improved = True
                    while improved and time_left() and calls < budget:
                        improved = False
                        for trial in [step, step * 0.5, step * 2.0]:
                            if calls >= budget or not time_left():
                                break
                            trial_x = x4.copy()
                            trial_x[coord] = np.clip(x4[coord] + direction * trial, -5, 5)
                            trial_f = func(trial_x)
                            calls += 1
                            if trial_f < fx4:
                                x4, fx4 = trial_x, trial_f
                                update_best(x4, fx4)
                                improved = True
                                break
                        if improved:
                            step *= 1.5
                        else:
                            break

        # --- Strategy 4: Nelder-Mead simplex refinement ---
        if time_left() and calls < budget:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    func, best_x, method='Nelder-Mead',
                    options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-5}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # --- Strategy 5: Final high-precision polish ---
        if time_left() and calls < budget:
            x5 = best_x.copy()
            fx5 = best_f
            sigma5 = 0.1
            steps5 = min(20, budget - calls)
            for _ in range(steps5):
                if not time_left() or calls >= budget:
                    break
                y = np.clip(x5 + sigma5 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                update_best(y, fy)
                if fy < fx5:
                    x5, fx5 = y, fy
                    sigma5 *= 1.1
                else:
                    sigma5 *= 0.9

        return best_x, best_f

# EVOLVE-BLOCK-END
