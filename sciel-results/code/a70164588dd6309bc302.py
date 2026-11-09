# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES-like optimizer with adaptive step size,
    restarts, and a strict wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 320.0  # 320s safety margin before 330s kill

        # We'll track the best point and value found
        best_x = None
        best_f = np.inf

        # Wrap func to count calls and enforce budget
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-restart strategy with CMA-ES-like adaptation
        # Use a fixed set of random seeds for reproducibility within an instance
        rng = np.random.default_rng(42)

        # Number of restarts: we'll do several short runs
        # Each run gets a portion of the budget
        # Let's do up to 10 restarts, but stop when budget or time is exhausted

        max_restarts = 10
        calls_per_restart = budget // max_restarts
        if calls_per_restart < 50:
            calls_per_restart = 50
            max_restarts = budget // calls_per_restart

        for restart in range(max_restarts):
            # Check time and budget
            if time.time() > deadline - 5.0:
                break
            if call_count >= budget:
                break

            # Initialize: first restart at origin, others random
            if restart == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            fx0 = eval_func(x0)
            if call_count >= budget:
                break

            # CMA-ES-like parameters
            sigma = 2.0
            # Learning rates
            c_sigma = 1.0 / (dim + 4)
            c_c = (4 + dim / dim) / (dim + 4 + 3)
            c_1 = 2 / ((dim + 1.3) ** 2 + dim)
            c_mu = min(1 - c_1, 2 * ((dim / 2 + 1) / (dim + 2)))

            # Population size
            lam = max(4, int(4 + 3 * np.log(dim)))
            mu = lam // 2  # number of parents

            # Weighted recombination weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step-size control
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # State
            mean = x0.copy()
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            # Track best within this restart
            local_best_x = x0.copy()
            local_best_f = fx0

            # Number of generations in this restart
            max_gens = calls_per_restart // lam
            if max_gens < 5:
                max_gens = 5

            for gen in range(max_gens):
                if time.time() > deadline - 5.0:
                    break
                if call_count + lam > budget:
                    break

                # Sample population
                # Use eigendecomposition of C
                try:
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-20))
                except np.linalg.LinAlgError:
                    break

                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5.0, 5.0)
                    fy = eval_func(y)
                    offspring.append((fy, y))

                if time.time() > deadline - 5.0:
                    break

                # Sort offspring
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < local_best_f:
                    local_best_f = offspring[0][0]
                    local_best_x = offspring[0][1].copy()

                # Recombination
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B @ (D @ ((y_w - mean) / sigma)))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - mean) / sigma

                # Update covariance
                delta = hsig * (1 / c_c - 1) + 1
                outer = np.outer(y_w - mean, y_w - mean)
                C = (1 + c_1 * delta - c_1 - c_mu) * C + c_1 * outer + c_mu / sigma ** 2 * sum(
                    weights[i] * np.outer(offspring[i][1] - mean, offspring[i][1] - mean)
                    for i in range(mu)
                )

                # Update mean
                mean = y_w

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Check for convergence
                if sigma < 1e-8:
                    break

            # End of restart; update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            # If we've found a very good solution, we might want to do a local refinement
            # But for now, just proceed to next restart

        # Ensure we return the best found
        if best_x is None:
            # Fallback: evaluate origin
            best_x = np.zeros(dim)
            best_f = float(func(best_x))

        return best_x, best_f

# EVOLVE-BLOCK-END
