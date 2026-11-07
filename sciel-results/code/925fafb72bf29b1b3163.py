"""CMA-ES with adaptive population sizing for f24 (Lunacek bi-Rastrigin)."""
import numpy as np
import time


class Optimizer:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) with:
    - Adaptive population sizing based on remaining budget
    - Multi-restart strategy for the two-basin structure of Lunacek
    - Coordinate descent polish at the end
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0

        best_x = None
        best_f = np.inf
        calls_made = [0]

        def counted_func(x):
            calls_made[0] += 1
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        def cma_es(x0, f0, budget_limit, sigma0=None):
            """Run CMA-ES from initial point x0 with initial fitness f0."""
            nonlocal best_x, best_f, calls_made

            if sigma0 is None:
                sigma0 = 1.0

            n = dim
            # Population size: lambda = 4 + floor(3 * ln(n))
            lam = max(4, 4 + int(3 * np.log(n + 1)))
            # Limit population to not exceed remaining budget
            max_iter = (budget_limit - calls_made[0]) // lam
            if max_iter < 2:
                return

            # Strategy parameters
            mu = lam // 2  # number of recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State variables
            m = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)

            # Track best
            local_best_f = f0
            local_best_x = x0.copy()

            # Evaluate initial point if not already done
            if calls_made[0] >= budget or time.time() > deadline:
                return

            for gen in range(max_iter):
                if calls_made[0] >= budget or time.time() > deadline:
                    break

                # Generate offspring
                x_offspring = np.zeros((lam, n))
                f_offspring = np.zeros(lam)

                for i in range(lam):
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    x_i = m + sigma * y
                    x_i = np.clip(x_i, -5, 5)
                    x_offspring[i] = x_i
                    f_offspring[i] = counted_func(x_i)
                    if f_offspring[i] < local_best_f:
                        local_best_f = f_offspring[i]
                        local_best_x = x_i.copy()

                # Sort by fitness
                order = np.argsort(f_offspring)
                x_offspring = x_offspring[order]
                f_offspring = f_offspring[order]

                # Update best
                if f_offspring[0] < local_best_f:
                    local_best_f = f_offspring[0]
                    local_best_x = x_offspring[0].copy()

                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

                # Check for convergence
                if np.std(f_offspring[:mu]) < 1e-10:
                    break

                # Update mean
                m_new = np.sum(weights[:, None] * x_offspring[:mu], axis=0)

                # Update step-size control
                y_mean = (m_new - m) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * y_mean)) / chi_n
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Update evolution paths
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (gen + 1))) / chi_n < (1.4 + 2 / (n + 1))

                # Update covariance matrix
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + delta_hsig * C - np.eye(n))
                C += c_mu * np.sum(weights[:, None, None] * np.outer(x_offspring[:mu] - m, x_offspring[:mu] - m) / sigma ** 2, axis=0)

                # Ensure positive definiteness and recompute B, D
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                B = eigvecs
                D = np.sqrt(eigvals)

                m = m_new

                # Update local best tracking
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Phase 1: Initial exploration - sample a few points to find promising regions
        n_explore = min(20, budget // 20)
        candidates = []
        for i in range(n_explore):
            if calls_made[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = counted_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates
        candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from top candidates with restarts
        n_restarts = min(3, len(candidates))
        remaining = budget - calls_made[0]
        budget_per_restart = max(50, remaining // n_restarts)

        for r in range(n_restarts):
            if calls_made[0] >= budget or time.time() > deadline:
                break

            if r < len(candidates):
                x0 = candidates[r][1].copy()
                f0 = candidates[r][0]
            else:
                x0 = best_x.copy()
                f0 = best_f

            # Start CMA-ES with adaptive sigma
            sigma0 = 1.0 if r == 0 else 0.5
            result = cma_es(x0, f0, calls_made[0] + budget_per_restart, sigma0)
            if result is not None:
                rx, rf = result
                if rf < best_f:
                    best_f = rf
                    best_x = rx.copy()

        # Phase 3: Final coordinate descent polish
        if calls_made[0] < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = 0.2
            for _ in range(50):
                if calls_made[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls_made[0] >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls_made[0] >= budget or time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[d] += sign * step
                        x_trial = np.clip(x_trial, -5, 5)
                        fx_trial = counted_func(x_trial)
                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break
                if not improved:
                    step *= 0.7
                    if step < 0.01:
                        break
                else:
                    step = max(0.03, step * 0.9)

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_made[0] < budget:
                best_f = counted_func(best_x)

        return best_x, best_f
