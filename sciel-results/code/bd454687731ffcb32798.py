"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start random sampling for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(20240101)
        best_x = None
        best_f = np.inf
        calls_made = 0
        all_evaluated = []

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = func(x)
            calls_made += 1
            all_evaluated.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cmaes(x_start, sigma0, max_calls):
            nonlocal calls_made, best_x, best_f, all_evaluated

            dim = self.dim
            pop = max(4, int(4 + 3 * np.log(dim)))
            mu = pop // 2

            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            mean = np.asarray(x_start, dtype=np.float64).copy()
            sigma = max(1e-8, float(sigma0))
            local_best_f = best_f
            local_best_x = best_x.copy() if best_x is not None else mean.copy()

            # Generate and evaluate initial population
            pop_xs = []
            pop_fs = []
            for k in range(pop):
                z = rng.standard_normal(dim)
                Bz = B @ z
                step = sigma * (Bz / D)
                xk = mean + step
                fk = evaluate(xk)
                pop_xs.append(xk)
                pop_fs.append(fk)
                if fk < local_best_f:
                    local_best_f = fk
                    local_best_x = xk.copy()
                if check_deadline() or calls_made >= max_calls:
                    return local_best_x, local_best_f

            while not check_deadline() and calls_made < max_calls:
                # Sort
                order = np.argsort(pop_fs)
                pop_xs = [pop_xs[i] for i in order]
                pop_fs = [pop_fs[i] for i in order]

                # Update best
                if pop_fs[0] < local_best_f:
                    local_best_f = pop_fs[0]
                    local_best_x = pop_xs[0].copy()

                # Mean update
                x_w = np.zeros(dim)
                for i in range(mu):
                    x_w += weights[i] * pop_xs[i]
                mean_old = mean.copy()
                mean = x_w

                # Sigma path update
                inv_B = np.linalg.inv(B)
                d_mean = (mean - mean_old) / (sigma * D)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_mean
                norm_ps = np.linalg.norm(ps)

                # C path update
                hsig = (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (calls_made - 1))) < (1.4 + 2 / (dim + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * d_mean

                # Covariance matrix update
                v_delta = hsig * (1 - c_c) - 1
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                B_inv = np.linalg.inv(B)
                for i in range(mu):
                    zi = B_inv @ (pop_xs[i] - mean_old) / (sigma * D)
                    C = B @ B.T
                    C = (1 + c_1 * v_delta * np.outer(pc, pc) + c_1 * delta_hsig) * C
                    C += 2 * c_mu * np.outer(zi, zi) * weights[i]
                    # Eigendecomposition
                    D2, B2 = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-20))
                    B = B2
                    D = D2

                # Sigma update
                sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1))
                sigma = min(sigma, 1e8)

                # Check convergence
                if sigma < 1e-12 or np.max(D) / np.min(D) > 1e14:
                    break

                # Generate new population
                pop_xs = []
                pop_fs = []
                for k in range(pop):
                    if check_deadline() or calls_made >= max_calls:
                        break
                    z = rng.standard_normal(dim)
                    Bz = B @ z
                    step = sigma * (Bz / D)
                    xk = mean + step
                    fk = evaluate(xk)
                    pop_xs.append(xk)
                    pop_fs.append(fk)
                    if fk < local_best_f:
                        local_best_f = fk
                        local_best_x = xk.copy()

                # If population was cut short, break
                if len(pop_xs) < pop:
                    break

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Phase 1: Random sampling to find good starting points
        n_random = min(120, total_budget)
        random_points = []
        for i in range(n_random):
            if check_deadline():
                break
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = rng.uniform(-4, 4, dim)
            else:
                # Mix of uniform and slightly biased toward center
                if rng.random() < 0.3:
                    x = rng.uniform(-2, 2, dim)
                else:
                    x = rng.uniform(-4.5, 4.5, dim)
            x = np.clip(x, -5, 5)
            evaluate(x)
            random_points.append(x.copy())

        # Sort by fitness
        random_points.sort(key=lambda p: best_f if p is None else func(p))

        # Get top diverse starting points
        start_candidates = []
        for x in random_points:
            if len(start_candidates) >= 5:
                break
            if all(np.linalg.norm(x - sc) > 1.5 for sc in start_candidates):
                start_candidates.append(x)

        # If we didn't get enough diverse, fill with best points
        if len(start_candidates) < 3:
            sorted_by_f = sorted(all_evaluated, key=lambda t: t[0])
            for f, x in sorted_by_f[:len(start_candidates) + 3]:
                if all(np.linalg.norm(x - sc) > 1.0 for sc in start_candidates):
                    start_candidates.append(x)
                if len(start_candidates) >= 5:
                    break

        # Ensure at least one start point
        if not start_candidates:
            start_candidates = [best_x.copy() if best_x is not None else np.zeros(dim)]

        # Phase 2: CMA-ES from top diverse starting points
        remaining = total_budget - calls_made
        n_starts = min(len(start_candidates), 4)
        per_start_budget = max(30, remaining // max(n_starts, 1))

        for i, sp in enumerate(start_candidates[:n_starts]):
            if check_deadline() or calls_made >= total_budget:
                break

            if i == 0:
                sigma_init = 1.5
                max_calls = total_budget if i == n_starts - 1 else calls_made + per_start_budget
            elif i == 1:
                sigma_init = 1.0
                max_calls = total_budget if i == n_starts - 1 else calls_made + per_start_budget
            else:
                sigma_init = 0.5
                max_calls = total_budget if i == n_starts - 1 else calls_made + per_start_budget

            cmaes(sp, sigma_init, max_calls)

        # Phase 3: Final refinement around global best
        if not check_deadline() and best_x is not None and calls_made < total_budget:
            x_init = best_x + rng.normal(0, 0.3, dim)
            x_init = np.clip(x_init, -5, 5)
            cmaes(x_init, 0.5, total_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
