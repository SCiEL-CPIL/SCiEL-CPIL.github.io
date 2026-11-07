"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive covariance for Rosenbrock valley."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def cma_es(x0, sigma0, max_calls):
            """CMA-ES with adaptive covariance matrix."""
            nonlocal calls, best_x, best_f

            if calls >= budget or time.time() > deadline:
                return x0, np.inf

            # CMA-ES parameters
            pop_size = 4 + int(np.floor(3 * np.log(n)))  # lambda
            c_sigma = (4 + pop_size / (n + 4)) / ((n + 1.3) ** 2 + pop_size)
            c_c = (4 + n / pop_size) / (n + 4)
            mu = pop_size // 2  # number of recombination weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_c = (4 + n / pop_size) / (n + 4)
            c_sigma = (mu_eff + 2) / (n + mu_eff)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            m = x0.copy()
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt_inv = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sigma = sigma0
            count = 0

            def update_chol(C):
                """Compute inverse square root of C via eigen decomposition."""
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return C_inv_sqrt

            # Initial population
            pop = []
            pop_f = []

            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                xi = m + sigma * (C_sqrt_inv @ z)
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                pop.append(xi)
                pop_f.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                count += 1

            if len(pop) < 2:
                return m, sigma

            # Sort by fitness
            order = np.argsort(pop_f)
            pop = [pop[i] for i in order]
            pop_f = [pop_f[i] for i in order]

            # Main loop
            while calls < budget and time.time() < deadline:
                if count >= max_calls:
                    break

                # Check convergence
                if sigma < 1e-12:
                    break
                f_range = pop_f[-1] - pop_f[0]
                if f_range < 1e-14 * max(1.0, abs(pop_f[0])):
                    break

                # Weighted recombination
                new_m = np.zeros(n)
                for i in range(mu):
                    new_m += weights[i] * pop[i]

                # Update step-size control
                old_m = m
                m = new_m
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C_sqrt_inv @ ((m - old_m) / sigma))
                if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** (count + 1)):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-15, 10.0)

                # Update covariance
                diff = np.array([(pop[i] - old_m) / sigma for i in range(mu)])
                h_sigma = 1.0
                if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** (count + 1)):
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0

                rank_mu = np.outer(diff.T @ weights, diff)
                rank_one = np.outer(ps, ps)
                C = (1 - c_c * c_sigma - c_sigma * (2 - c_sigma) * mu_eff) * C + \
                    c_sigma * (rank_mu + (1 - h_sigma) * (mu_eff + 2) * rank_one)

                # Symmetrize
                C = (C + C.T) / 2

                # Update inverse square root
                C_sqrt_inv = update_chol(C)

                # Generate new population
                new_pop = []
                new_pop_f = []
                for i in range(pop_size):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    xi = m + sigma * (C_sqrt_inv @ z)
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls += 1
                    new_pop.append(xi)
                    new_pop_f.append(fi)
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()
                    count += 1

                if len(new_pop) < 2:
                    break

                order = np.argsort(new_pop_f)
                pop = [new_pop[i] for i in order]
                pop_f = [new_pop_f[i] for i in order]

            return m, sigma

        # Multi-start strategy
        # Start at 0 (good heuristic for rotated Rosenbrock)
        x_start = np.zeros(n)
        f_start = func(x_start)
        calls += 1
        best_x = x_start.copy()
        best_f = f_start

        if calls >= budget:
            return best_x, best_f

        # First CMA-ES run from origin with moderate sigma
        cma_es(x_start, 1.0, budget - calls)

        # Adaptive restarts
        num_restarts = 0
        max_restarts = 8

        for restart_idx in range(max_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # If very good solution, do minimal additional exploration
            if best_f < 1e-4:
                # Try a few random perturbations around best
                for _ in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    x_pert = best_x + rng.normal(0, 0.1, n)
                    x_pert = np.clip(x_pert, -5, 5)
                    f_pert = func(x_pert)
                    calls += 1
                    if f_pert < best_f:
                        best_f = f_pert
                        best_x = x_pert.copy()
                if restart_idx >= 1:
                    break
                continue

            # Generate diverse starting points
            if restart_idx == 0:
                x_new = rng.uniform(-3, 3, n)
            elif restart_idx == 1:
                x_new = rng.uniform(-4, 4, n)
            elif restart_idx == 2:
                x_new = rng.uniform(-5, 5, n)
            else:
                # Mix of random and perturbed best
                if rng.random() < 0.5:
                    x_new = rng.uniform(-5, 5, n)
                else:
                    x_new = best_x + rng.normal(0, 2.0, n)
                    x_new = np.clip(x_new, -5, 5)

            f_new = func(x_new)
            calls += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            if calls >= budget:
                break

            # Run CMA-ES from this point with smaller sigma as we refine
            sigma0 = 0.5 if restart_idx == 0 else 0.3
            cma_es(x_new, sigma0, budget - calls)
            num_restarts += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
