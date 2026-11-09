"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with aggressive exploration for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # CMA-ES Parameters
        lambda_pop = 12
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)

        cc = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
        cs = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
        c1 = 2 / ((lambda_pop + 2) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_pop + 2) ** 2 + mu_eff))

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

        def run_cmaes(x_init, sigma_init, max_calls):
            nonlocal calls_made, best_x, best_f

            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            local_best_f = evaluate(x_mean)
            local_best_x = x_mean.copy()

            gen = 0
            no_improve_gen = 0
            last_improvement = local_best_f

            while not check_deadline() and calls_made < max_calls:
                # Sample lambda points
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

                # Sort by fitness
                offspring.sort(key=lambda x: x[1])

                # Update mean
                x_new_mean = np.sum([weights[i] * offspring[i][0] for i in range(min(mu, len(offspring)))], axis=0)

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (B.T @ (x_new_mean - x_mean))))

                hsig = (np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt(lambda_pop + 1 - mu_eff))

                if hsig:
                    sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / np.sqrt(lambda_pop) - 1))

                # Update covariance
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma

                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(offspring))):
                    y_i = (offspring[i][0] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2

                # Update mean
                x_mean = x_new_mean
                gen += 1

                # Update best
                if offspring[0][1] < local_best_f:
                    local_best_f = offspring[0][1]
                    local_best_x = offspring[0][0].copy()
                    last_improvement = local_best_f
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1

                # Update basis
                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-12))
                except:
                    D2, B = np.linalg.eigh(C + 1e-10 * np.eye(dim))
                    D = np.sqrt(np.maximum(D2, 1e-12))

                # Check for stall - restart if no improvement for 10 generations
                if no_improve_gen >= 10 and calls_made < max_calls - 10:
                    break

                # Check convergence
                if np.linalg.norm(D) < 1e-12 or sigma < 1e-12:
                    break

            return local_best_f, local_best_x, calls_made

        # Phase 1: Aggressive broad exploration with many diverse starts
        # Use 50% of budget for exploration with 5 different starting regions
        exploration_starts = [
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-2, 2, dim),
        ]

        alloc_per_start = int(total_budget * 0.10)  # 10% per start, 5 starts = 50%

        for i in range(5):
            if check_deadline() or calls_made >= total_budget:
                break
            x_init = exploration_starts[i].copy()
            sigma_init = 3.0 + 0.5 * i  # Vary sigma for diversity
            max_calls_i = min(calls_made + alloc_per_start, total_budget)
            run_cmaes(x_init, sigma_init, max_calls_i)

        # Phase 2: Focused refinement around best - 30% budget
        if not check_deadline() and best_x is not None:
            x_init = best_x.copy()
            sigma_init = 1.5
            alloc2 = int(total_budget * 0.30)
            max_calls_2 = min(calls_made + alloc2, total_budget)
            run_cmaes(x_init, sigma_init, max_calls_2)

        # Phase 3: Final aggressive local refinement - remaining budget
        if not check_deadline() and best_x is not None and calls_made < total_budget:
            # Use a very aggressive local search with shrinking steps
            x_cur = best_x.copy()
            f_cur = best_f
            step_size = 1.0

            for _ in range(100):
                if check_deadline() or calls_made >= total_budget:
                    break

                improved = False
                for coord in range(dim):
                    if check_deadline() or calls_made >= total_budget:
                        break
                    for direction in [-1, 1]:
                        if check_deadline() or calls_made >= total_budget:
                            break
                        x_try = x_cur.copy()
                        x_try[coord] += direction * step_size
                        x_try = np.clip(x_try, -5, 5)
                        f_try = evaluate(x_try)
                        if f_try < f_cur:
                            f_cur = f_try
                            x_cur = x_try.copy()
                            improved = True
                            break

                if not improved:
                    step_size *= 0.7
                    if step_size < 0.001:
                        break

            if f_cur < best_f:
                best_f = f_cur
                best_x = x_cur.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
