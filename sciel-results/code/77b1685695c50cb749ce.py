"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        dim = self.dim
        total_budget = self.budget

        # We'll do multiple restarts of CMA-ES, each with a portion of the budget.
        # Strategy: first restart gets more budget, subsequent ones get less.
        # For 500 budget: restart 1: 300, restart 2: 150, restart 3: 50

        restart_budgets = []
        remaining = total_budget
        # Give decreasing budgets to restarts
        for i in range(3):
            if i == 0:
                b = min(remaining, 300)
            elif i == 1:
                b = min(remaining, 150)
            else:
                b = remaining
            if b > 0:
                restart_budgets.append(b)
            remaining -= b

        if not restart_budgets:
            restart_budgets = [total_budget]

        best_x = None
        best_f = np.inf
        used_calls = 0

        for restart_idx, rb in enumerate(restart_budgets):
            if time.time() > deadline:
                break
            if used_calls >= total_budget:
                break

            # Initialize CMA-ES
            if restart_idx == 0:
                # First restart: sample from center and corners to find a good start
                candidates = [np.zeros(dim)]
                for d in range(dim):
                    for s in [-1, 1]:
                        c = np.zeros(dim)
                        c[d] = s * 4.0
                        candidates.append(c)
                # Also add a few random points
                rng_init = np.random.default_rng(42 + restart_idx)
                for _ in range(4):
                    candidates.append(rng_init.uniform(-4, 4, dim))

                best_cand = None
                best_cand_f = np.inf
                for cand in candidates:
                    if time.time() > deadline:
                        break
                    f_val = func(np.clip(cand, -5, 5))
                    used_calls += 1
                    if f_val < best_cand_f:
                        best_cand_f = f_val
                        best_cand = cand.copy()
                    if used_calls >= total_budget:
                        break
                x = best_cand if best_cand is not None else np.zeros(dim)
                fx = best_cand_f
            else:
                # Subsequent restarts: start near the best found so far with perturbation
                rng_init = np.random.default_rng(42 + restart_idx * 1000)
                if best_x is not None:
                    x = np.clip(best_x + rng_init.uniform(-2, 2, dim), -5, 5)
                else:
                    x = rng_init.uniform(-4, 4, dim)
                fx = func(x)
                used_calls += 1

            if used_calls >= total_budget:
                break

            # CMA-ES parameters
            pop_size = max(4, int(4 + 2 * np.log(dim)))
            # Ensure pop_size * (restarts) doesn't exceed budget
            # For dim=5: pop_size = max(4, int(4+2*1.609)) = max(4, 7) = 7
            # But we need to be careful with budget
            if rb < pop_size + 2:
                # Too few calls left, just do a simple random search
                while used_calls < total_budget and time.time() < deadline:
                    rng_s = np.random.default_rng(used_calls)
                    y = np.clip(x + rng_s.uniform(-1, 1, dim), -5, 5)
                    fy = func(y)
                    used_calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                continue

            # CMA-ES state
            sigma = 1.0
            C = np.eye(dim)
            # Cholesky of C
            C_sqrt = np.eye(dim)

            # Mean
            mean = x.copy()
            f_mean = fx

            # History for evolution path
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Parameters
            c_sigma = 1.0 / (pop_size + 3)
            c_c = (4 + dim / pop_size) / ((dim + 4) * (pop_size + 4))
            mu = pop_size // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((dim + 2) ** 2 + pop_size))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            calls_in_restart = 0
            no_improve = 0
            best_restart_f = fx

            while used_calls < total_budget and calls_in_restart < rb:
                if time.time() > deadline:
                    break

                # Generate population
                z = np.random.default_rng(used_calls * 7919 + restart_idx * 104729).standard_normal((pop_size, dim))
                # Apply C_sqrt: y_i = mean + sigma * C_sqrt @ z_i
                offspring = np.array([mean + sigma * (C_sqrt @ z[i]) for i in range(pop_size)])
                offspring = np.clip(offspring, -5, 5)

                # Evaluate
                fitness = np.empty(pop_size)
                for i in range(pop_size):
                    if used_calls >= total_budget or time.time() > deadline:
                        fitness[i] = np.inf
                        continue
                    fitness[i] = func(offspring[i])
                    used_calls += 1
                    calls_in_restart += 1

                # Sort by fitness
                order = np.argsort(fitness)
                offspring_sorted = offspring[order]
                fitness_sorted = fitness[order]

                # Update best
                if fitness_sorted[0] < best_restart_f:
                    best_restart_f = fitness_sorted[0]
                    no_improve = 0
                else:
                    no_improve += 1

                if fitness_sorted[0] < best_f:
                    best_f = fitness_sorted[0]
                    best_x = offspring_sorted[0].copy()

                # Update mean
                old_mean = mean.copy()
                mean_new = np.sum(weights[:, None] * offspring_sorted[:mu], axis=0)
                mean = mean_new

                # Update sigma
                if np.linalg.norm(mean - old_mean) > 1e-16:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ ((mean - old_mean) / (sigma * np.linalg.norm(C_sqrt @ (mean - old_mean)))) if np.linalg.norm(C_sqrt @ (mean - old_mean)) > 1e-16 else 0)
                else:
                    p_sigma = (1 - c_sigma) * p_sigma

                chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 100)

                # Update C (simplified: use rank-one and rank-mu updates)
                y_mean = (mean - old_mean) / sigma
                v = C_sqrt @ y_mean
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * v

                # Rank-one update
                C += c_1 * (np.outer(v, v) - (1 + (1 - c_sigma) * c_sigma) * C)
                # Rank-mu update
                z_sorted = z[order][:mu]
                y_sorted = (offspring_sorted[:mu] - old_mean) / sigma
                v_sorted = C_sqrt @ y_sorted
                C += c_mu * np.sum(weights[:, None, None] * np.outer(v_sorted, v_sorted), axis=0)

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Update C_sqrt via eigenvalue decomposition (cheaper than Cholesky for small dim)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

                # Check for stagnation
                if no_improve > 2 * pop_size:
                    break

            # End of this CMA-ES restart

        return best_x, best_f
# EVOLVE-BLOCK-END
