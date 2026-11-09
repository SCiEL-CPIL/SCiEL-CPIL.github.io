"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES with smaller population for faster exploration
        def run_cmaes(x_init, sigma_init, max_calls):
            nonlocal calls_made, best_x, best_f

            lambda_pop = 8
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)

            cc = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
            cs = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
            c1 = 2 / ((lambda_pop + 2) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_pop + 2) ** 2 + mu_eff))

            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            local_best_f = evaluate(x_mean)
            local_best_x = x_mean.copy()
            no_improve_gen = 0
            last_improvement = local_best_f

            while not check_deadline() and calls_made < max_calls:
                offspring = []
                for i in range(lambda_pop):
                    if check_deadline() or calls_made >= max_calls:
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = evaluate(x_new)
                    offspring.append((x_new, f_new))

                if len(offspring) == 0:
                    break

                offspring.sort(key=lambda x: x[1])

                x_new_mean = np.sum(
                    [weights[i] * offspring[i][0] for i in range(min(mu, len(offspring)))], axis=0
                )

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (
                    B @ (D * (B.T @ (x_new_mean - x_mean)))
                )

                hsig = np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt(
                    lambda_pop + 1 - mu_eff
                )

                if hsig:
                    sigma *= np.exp(
                        cs / dim * (np.linalg.norm(ps) / np.sqrt(lambda_pop) - 1)
                    )

                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (
                    x_new_mean - x_mean
                ) / sigma

                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(offspring))):
                    y_i = (offspring[i][0] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2
                x_mean = x_new_mean

                if offspring[0][1] < local_best_f:
                    local_best_f = offspring[0][1]
                    local_best_x = offspring[0][0].copy()
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1

                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-12))
                except Exception:
                    D2, B = np.linalg.eigh(C + 1e-10 * np.eye(dim))
                    D = np.sqrt(np.maximum(D2, 1e-12))

                if no_improve_gen >= 15 and calls_made < max_calls - 10:
                    break

                if sigma < 1e-14:
                    break

            return local_best_f, local_best_x, calls_made

        # Strategy: Multiple diverse starts with progressive refinement
        # Phase 1: Initial exploration from center
        if not check_deadline():
            x_init = rng.uniform(-2, 2, dim)
            sigma_init = 3.0
            alloc = int(total_budget * 0.20)
            max_calls = min(calls_made + alloc, total_budget)
            _, _, calls_made = run_cmaes(x_init, sigma_init, max_calls)

        # Phase 2: Explore from opposite corner
        if not check_deadline() and best_x is not None:
            x_init = rng.uniform(-4, 4, dim)
            # Ensure distance from best
            for _ in range(5):
                if np.linalg.norm(x_init - best_x) > 3.0:
                    break
                x_init = rng.uniform(-4, 4, dim)
            sigma_init = 2.5
            alloc = int(total_budget * 0.15)
            max_calls = min(calls_made + alloc, total_budget)
            _, _, calls_made = run_cmaes(x_init, sigma_init, max_calls)

        # Phase 3: Refine around best
        if not check_deadline() and best_x is not None:
            x_init = np.clip(best_x + rng.normal(0, 0.8, dim), -5, 5)
            sigma_init = 1.2
            alloc = int(total_budget * 0.15)
            max_calls = min(calls_made + alloc, total_budget)
            _, _, calls_made = run_cmaes(x_init, sigma_init, max_calls)

        # Phase 4: Another diverse start
        if not check_deadline():
            x_init = rng.uniform(-4, 4, dim)
            for _ in range(5):
                if best_x is not None and np.linalg.norm(x_init - best_x) > 2.5:
                    break
                x_init = rng.uniform(-4, 4, dim)
            sigma_init = 2.0
            alloc = int(total_budget * 0.12)
            max_calls = min(calls_made + alloc, total_budget)
            _, _, calls_made = run_cmaes(x_init, sigma_init, max_calls)

        # Phase 5: Final refinement with remaining budget
        if not check_deadline() and best_x is not None:
            x_init = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)
            sigma_init = 0.8
            max_calls = total_budget
            _, _, calls_made = run_cmaes(x_init, sigma_init, max_calls)

        # Phase 6: If still have budget, one more exploration
        if not check_deadline():
            x_init = rng.uniform(-4, 4, dim)
            sigma_init = 1.8
            max_calls = total_budget
            _, _, calls_made = run_cmaes(x_init, sigma_init, max_calls)

        return best_x, best_f
# EVOLVE-BLOCK-END
