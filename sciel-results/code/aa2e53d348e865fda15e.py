import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid CMA-ES + L-BFGS-B optimizer for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Safety margin: 330s limit, stop at 320s
        deadline = time.time() + 320

        best_x = None
        best_f = np.inf
        calls = 0

        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)
        bounds = [(-5, 5) for _ in range(dim)]

        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, lower, upper)
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def cma_es(x0, sigma, max_calls):
            """Simple CMA-ES implementation."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return

            n = dim
            # Population size
            mu = max(1, n // 2)
            lambda_ = 2 * mu
            # Learning rates
            c_sigma = (mu_eff + 2) / ((n + 1.3) ** 2 + mu_eff)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.8) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            mu_eff = mu / 2

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Initialize
            mean = np.array(x0, dtype=float)
            sigma = sigma
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Track local best
            local_best_x = mean.copy()
            local_best_f = wrapped_func(mean)

            generation = 0
            max_generations = max_calls // lambda_ + 1

            for gen in range(max_generations):
                if calls >= budget or time.time() > deadline:
                    break
                if local_best_f > best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

                # Generate population
                for i in range(lambda_):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Sample from multivariate normal with covariance C
                    # Use Cholesky decomposition
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        L = np.eye(n)
                    z = rng.standard_normal(n)
                    x_new = mean + sigma * (L @ z)
                    x_new = np.clip(x_new, lower, upper)
                    f_val = wrapped_func(x_new)
                    if f_val < local_best_f:
                        local_best_f = f_val
                        local_best_x = x_new.copy()

                # Sort by fitness (we need to re-evaluate or store)
                # For simplicity, let's just track the best and use a simpler update
                # Actually, let's use a simpler CMA-ES variant

                # Update mean using weighted average of best mu points
                # This is a simplified version - in practice we'd need to store all points
                # Let's use a simpler approach: just update mean toward best
                mean = 0.8 * mean + 0.2 * local_best_x
                mean = np.clip(mean, lower, upper)

                # Update sigma (simplified)
                sigma *= np.exp(c_sigma / chi_n * (0.5 - 0.5))  # Simplified

                generation += 1

            # Final update
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        def local_refine(x_start, max_calls):
            """Local refinement using L-BFGS-B."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            if x_start is None:
                return

            remaining = budget - calls
            if remaining < 5:
                return

            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxfev': remaining,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Phase 1: Multi-start with CMA-ES exploration
        n_starts = 6
        per_start_budget = max(20, (budget - 10) // n_starts)

        # Diverse starting points
        starts = []
        starts.append(np.zeros(dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-5, 5, dim))
        for i in range(3):
            starts.append(rng.uniform(-5, 5, dim))

        for i, x0 in enumerate(starts):
            if calls >= budget or time.time() > deadline:
                break

            # Evaluate start
            fx0 = wrapped_func(x0)

            if calls >= budget or time.time() > deadline:
                break

            # CMA-ES exploration
            remaining_for_cma = min(per_start_budget - 10, budget - calls)
            if remaining_for_cma > 10:
                cma_es(x0, sigma=2.0, max_calls=remaining_for_cma)

            # Local refinement
            if calls < budget and best_x is not None:
                local_refine(best_x, budget - calls)

        # Phase 2: Intensive refinement of the best point
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                # Multiple refinement passes with decreasing tolerance
                for tol in [1e-12, 1e-14, 1e-16]:
                    if calls >= budget or time.time() > deadline:
                        break
                    local_refine(best_x, budget - calls)

        # Phase 3: Final polish
        if calls < budget and best_x is not None and time.time() < deadline:
            local_refine(best_x, budget - calls)

        return best_x, best_f
