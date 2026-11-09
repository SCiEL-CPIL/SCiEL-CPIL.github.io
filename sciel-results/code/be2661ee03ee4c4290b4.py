"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Rastrigin on [-5,5]^5 with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_left():
            return deadline - time.time()

        # Phase 1: Random exploration to find good basin (reduced to save budget)
        explore_budget = min(20, int(budget * 0.05))
        best_x = None
        best_f = np.inf
        calls_used = 0

        for i in range(explore_budget):
            if time_left() < 5:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(n)
            best_f = func(x)
            calls_used += 1
            best_x = x.copy()

        # Phase 2: CMA-ES with restarts
        remaining = budget - calls_used

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))  # for n=5: 8
        mu = lam // 2  # 4
        # Use larger population for better exploration in multimodal landscape
        lam = max(lam, 10)
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = 2 * (mueff - 2 + np.sum(w[:mu] ** 2) / np.sum(w) ** 2) / ((n + 2) ** 2 + mueff)
        cmu = min(cmu, 2 * (1 - c1))
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f

        # Run CMA-ES with adaptive restarts
        # Start with a moderate sigma
        max_restarts = 4
        restart_threshold = 1e-6

        for restart in range(max_restarts):
            if calls_used >= budget or time_left() < 10:
                break

            # Initialize CMA-ES state
            mean = global_best_x.copy()
            sigma = 0.8 / (restart + 1)  # Decreasing sigma for restarts
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            gen = 0

            # Track convergence
            stagnant = 0
            best_f_this_restart = global_best_f

            while calls_used < budget and time_left() > 5:
                gen += 1

                # Generate population
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    L = np.eye(n)

                # Generate lam samples vectorized
                Z = rng.standard_normal((lam, n))
                offspring = mean + sigma * (Z @ L.T)
                offspring = np.clip(offspring, -5, 5)

                # Evaluate offspring
                offspring_f = np.empty(lam)
                for i in range(lam):
                    if calls_used >= budget or time_left() < 5:
                        break
                    offspring_f[i] = func(offspring[i])
                    calls_used += 1

                if calls_used >= budget or time_left() < 5:
                    # Evaluate only what we could
                    if i + 1 < lam:
                        offspring_f = offspring_f[:i + 1]
                        offspring = offspring[:i + 1]
                        lam_actual = i + 1
                    else:
                        lam_actual = lam
                else:
                    lam_actual = lam

                # Update global best
                idx_best = np.argmin(offspring_f)
                if offspring_f[idx_best] < global_best_f:
                    global_best_f = offspring_f[idx_best]
                    global_best_x = offspring[idx_best].copy()

                # Sort by fitness
                order = np.argsort(offspring_f)
                selected_y = offspring[order[:mu]]
                selected_f = offspring_f[order[:mu]]

                # Update mean
                m_old = mean.copy()
                mean_new = np.sum([w[i] * selected_y[i] for i in range(mu)], axis=0)

                # Update evolution paths
                try:
                    C_inv_sqrt = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (mean_new - m_old) / sigma

                # Check hsig
                chi_n = np.sqrt(n) * (1 + 1 / (4 * n) - 1 / (21 * n ** 2))
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + 2 / (n + 1)) * chi_n

                if hsig:
                    pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (mean_new - m_old) / sigma

                # Update covariance matrix
                delta_sig = 1.0 if hsig else 0.0
                C = (1 - c1 - cmu) * C
                C += c1 * (1 - delta_sig) * ps @ ps.T + c1 * delta_sig * (ps @ ps.T + (1 - (1 - cs) ** 2) * pc @ pc.T)
                C += cmu * np.sum([w[i] * np.outer(selected_y[i] - m_old, selected_y[i] - m_old) for i in range(mu)], axis=0)

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += 1e-12 * np.eye(n)

                mean = mean_new

                # Adapt sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-10, min(sigma, 10.0))

                # Check convergence
                if np.linalg.norm(mean_new - m_old) < 1e-9 * max(1, np.linalg.norm(mean_new)):
                    stagnant += 1
                else:
                    stagnant = 0

                if stagnant > 15:
                    break

            # If we converged or used up budget, stop
            if calls_used >= budget or time_left() < 10:
                break

            # Otherwise, restart from global best with smaller sigma
            # Only restart if we haven't made significant progress relative to start
            if global_best_f < best_f - 1e-6:
                # Made progress, continue to next restart with tighter sigma
                continue
            else:
                # No progress, break
                break

        # Phase 3: Final polish with coordinate descent / Nelder-Mead style
        if time_left() > 20 and calls_used < budget:
            polish_budget = min(50, budget - calls_used)
            x_cur = global_best_x.copy()
            f_cur = global_best_f
            step = 0.1

            for polish_iter in range(polish_budget):
                if calls_used >= budget or time_left() < 5:
                    break
                improved = False
                for dim_i in range(n):
                    if calls_used >= budget or time_left() < 5:
                        break
                    for direction in [1, -1]:
                        x_trial = x_cur.copy()
                        x_trial[dim_i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls_used += 1
                        if f_trial < f_cur:
                            x_cur = x_trial
                            f_cur = f_trial
                            improved = True
                            if f_trial < global_best_f:
                                global_best_f = f_trial
                                global_best_x = x_cur.copy()
                    if improved:
                        step *= 0.5
                    else:
                        step *= 0.7
                if step < 1e-6:
                    break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
