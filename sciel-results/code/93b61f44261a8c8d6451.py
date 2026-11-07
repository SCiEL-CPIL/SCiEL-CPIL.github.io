"""
CMA-ES optimizer for BBOB f3 (Rastrigin) on [-5, 5]^5.
Clean, correct implementation with proper convergence handling.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin

        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Brief exploration to find a promising basin
        n_explore = 20
        explore_points = [np.zeros(n)]
        for _ in range(n_explore - 1):
            explore_points.append(rng.uniform(-4.5, 4.5, n))

        for x in explore_points:
            if check_deadline():
                break
            x = np.clip(x, -5, 5)
            f = func(x)
            update_best(x, f)

        if best_x is None:
            x = np.zeros(n)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Phase 2: CMA-ES refinement
        def run_cmaes(x_start, f_start, max_calls):
            nonlocal calls_made, best_x, best_f

            lambda_pop = 4 + int(3 * np.log(n))
            lambda_pop = min(lambda_pop, max(2, max_calls // 2))
            if lambda_pop < 2:
                return

            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            sigma = 0.5
            mean = x_start.copy()

            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_sqrt_inv = np.eye(n)

            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (weights[1] - 1) / ((n + 2)**2 + mu_eff))

            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

            no_improve = 0
            last_best = f_start

            while not check_deadline() and calls_made < budget:
                # Check if we can afford a full generation
                if calls_made + lambda_pop > budget:
                    remaining = budget - calls_made
                    if remaining < mu:
                        break
                    # Do a partial generation
                    lambda_pop = remaining
                    mu = max(1, lambda_pop // 2)
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights /= np.sum(weights)
                    mu_eff = 1.0 / np.sum(weights**2)

                offspring = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(n)
                    y = mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    offspring.append((fy, y))
                    if check_deadline():
                        break

                if len(offspring) < mu:
                    break

                offspring.sort(key=lambda t: t[0])

                # Check improvement
                if offspring[0][0] < last_best - 1e-10:
                    last_best = offspring[0][0]
                    no_improve = 0
                else:
                    no_improve += 1

                # Recombination
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]

                old_mean = mean.copy()
                mean = y_w

                # Update paths
                ys = (mean - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n
                        < (1.4 + 2 / (n + 1)) * cc)

                if hsig:
                    pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                else:
                    pc = (1 - cs) * pc

                # Update covariance
                rank_one = np.outer(ys, ys)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i][1] - old_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = ((1 - c1 - cmu) * C
                     + c1 * (np.outer(pc, pc) + (1 - hsig) * C)
                     + cmu * rank_mu)

                C = (C + C.T) / 2.0

                # Ensure positive definiteness
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))

                # Convergence
                if sigma < 1e-14:
                    break
                if no_improve > 30 and sigma < 0.05:
                    break

            return best_x, best_f

        run_cmaes(best_x, best_f, budget - calls_made)

        # Phase 3: Restart if budget remains and CMA-ES converged early
        remaining = budget - calls_made
        if remaining > 30 and not check_deadline():
            # Try a few restarts from perturbed best_x
            for attempt in range(3):
                if check_deadline() or remaining < 30:
                    break
                perturb = 1.0 / (attempt + 1)
                x_restart = np.clip(best_x + perturb * rng.standard_normal(n), -5, 5)
                f_restart = func(x_restart)
                calls_made += 1
                remaining = budget - calls_made
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()

                if remaining > 20:
                    run_cmaes(best_x, best_f, remaining)
                    remaining = budget - calls_made

        return best_x, best_f

# EVOLVE-BLOCK-END
