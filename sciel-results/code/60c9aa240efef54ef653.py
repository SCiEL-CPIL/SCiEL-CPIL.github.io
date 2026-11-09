# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Random-init (1+1)-ES with multi-scale adaptive restarts and CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_cmaes(self, func, x0, f0, max_calls, rng, deadline):
        """Run CMA-ES from x0 with budget max_calls. Returns (best_x, best_f, calls_made)."""
        dim = self.dim
        lam = max(6, 4 * dim)
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()

        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        c1 = (2 + 2 * min(1, 0.4 / (dim + 1))) / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))

        x_mean = x0.copy()
        sigma = 0.3
        C = np.eye(dim)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        local_calls = 0
        local_best_x = x0.copy()
        local_best_f = f0

        while local_calls < max_calls:
            if time.time() > deadline:
                break

            offspring = []
            offspring_f = []
            for i in range(lam):
                if local_calls >= max_calls:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                fy = func(y)
                offspring_f.append(fy)
                local_calls += 1

                if fy < local_best_f:
                    local_best_f = fy
                    local_best_x = y.copy()

            if len(offspring_f) < 2:
                break

            idx = np.argsort(offspring_f)
            sorted_off = [offspring[i] for i in idx]

            old_mean = x_mean.copy()

            x_mean = np.zeros(dim)
            for i in range(mu):
                x_mean += weights[i] * sorted_off[i]

            if sigma > 1e-10:
                v_mean = (x_mean - old_mean) / sigma
            else:
                v_mean = np.zeros(dim)

            h_sigma = min(1.0, (1.0 - np.exp(-1.0 / ds)) / 2.0)

            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * v_mean
            sigma *= np.exp(cs * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)

            h_c = min(1.0, (1.0 - np.exp(-1.0 / ds)) / 2.0)
            p_c = (1 - c1 - cmu) * p_c + h_c * np.sqrt((c1 + cmu) * (2 - c1 - cmu) * mueff) * v_mean

            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                if sigma > 1e-10:
                    vi = (sorted_off[i] - old_mean) / sigma
                else:
                    vi = np.zeros(dim)
                rank_mu += weights[i] * np.outer(vi, vi)

            C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * rank_mu
            C = (C + C.T) / 2

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = max(sigma, 1e-3)

            if np.linalg.norm(x_mean - old_mean) < 1e-10 * sigma:
                break

        return local_best_x, local_best_f, local_calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285

        rng = np.random.default_rng(42)

        # Random initialization
        best_x = rng.uniform(-5, 5, dim)
        best_f = func(best_x)
        calls = 1

        # (1+1)-ES with adaptive sigma and multi-scale restarts
        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        sigma_min = 0.005
        sigma_max = 4.0

        # 1/5 success rule tracking
        success_window = []
        window_size = 15

        stagnation = 0
        no_improve_count = 0

        # Budget allocation: last 15% for CMA-ES polish
        polish_start = int(budget * 0.85)

        while calls < budget:
            if time.time() > deadline:
                break

            in_polish = calls >= polish_start

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            # Track success
            success = fy < fx
            success_window.append(success)
            if len(success_window) > window_size:
                success_window.pop(0)

            # 1/5th success rule for sigma adaptation
            if len(success_window) >= window_size:
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.3:
                    sigma = min(sigma * 1.2, sigma_max)
                elif success_rate < 0.1:
                    sigma = max(sigma * 0.7, sigma_min)

            # Update current solution
            if success:
                x, fx = y, fy
                stagnation = 0
            else:
                stagnation += 1

            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adaptive restart conditions
            progress_ratio = calls / budget
            stagnation_limit = int(40 * (1.0 - 0.5 * progress_ratio))
            no_improve_limit = int(60 * (1.0 - 0.5 * progress_ratio))

            if in_polish:
                stagnation_limit = 15
                no_improve_limit = 25

            # Restart if stagnated
            if stagnation >= stagnation_limit:
                # Multi-scale restart: pick a scale based on progress
                explore_prob = min(0.3 + 0.01 * no_improve_count, 0.8)

                if rng.random() < explore_prob:
                    # Global exploration
                    x = rng.uniform(-5, 5, dim)
                    sigma = 2.0
                else:
                    # Local exploration around best with moderate step
                    local_sigma = max(sigma * 2.0, 0.5)
                    x = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                    sigma = local_sigma

                fx = func(x)
                calls += 1
                stagnation = 0
                no_improve_count = 0
                success_window = []

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                if calls >= budget:
                    break

            # Aggressive restart if no global improvement
            if no_improve_count >= no_improve_limit:
                # Try multiple random restarts and pick the best
                best_restart_x = x
                best_restart_f = fx
                n_restarts = 3
                for _ in range(n_restarts):
                    if calls >= budget:
                        break
                    cand = rng.uniform(-5, 5, dim)
                    cand_f = func(cand)
                    calls += 1
                    if cand_f < best_restart_f:
                        best_restart_f = cand_f
                        best_restart_x = cand
                    if cand_f < best_f:
                        best_f = cand_f
                        best_x = cand.copy()

                x = best_restart_x
                fx = best_restart_f
                sigma = 1.5
                stagnation = 0
                no_improve_count = 0
                success_window = []

                if calls >= budget:
                    break

        # Phase 2: CMA-ES refinement around the best point found
        remaining = budget - calls
        if remaining > 15:
            cma_best_x, cma_best_f, cma_calls = self._run_cmaes(func, best_x, best_f, remaining, rng, deadline)
            calls += cma_calls
            if cma_best_f < best_f:
                best_f = cma_best_f
                best_x = cma_best_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
