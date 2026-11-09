"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio: CMA-ES + (1+1)-ES + restarts with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cmaes(self, func, x0, f0, budget, rng, time_limit=None):
        """Run CMA-ES with proper parameters for dim=5."""
        dim = self.dim
        # Proper CMA-ES population size: n = 4 + floor(3 * ln(dim))
        n_pop = 4 + int(3 * np.log(max(dim, 2)))
        mu = max(1, n_pop // 2)
        
        # Weights
        wi = np.log((mu + 1) / 2) - np.log(np.arange(1, mu + 1))
        wi /= wi.sum()
        mu_eff = 1.0 / (wi ** 2).sum()
        
        # Adaptation rates
        cmu = 2 / ((dim + 1) ** 2 + 10)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + dim / mu_eff) / (dim + 4 + 2 * mu_eff / dim)
        c1 = (2 - cmu - cs) / ((dim + 1.3) ** 2 + 1)
        cmu_ = cmu * 1.5

        sigma = 0.5
        mean = x0.copy()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x0.copy()
        best_f = f0

        start_time = time.time()
        calls_used = 0

        for it in range(budget):
            # Time guard
            if time_limit is not None and (time.time() - start_time) > time_limit * 0.92:
                break

            # Sample
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-8 * np.eye(dim)
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.eye(dim)

            z = rng.standard_normal((n_pop, dim))
            y = mean[None, :] + sigma * (z @ B.T)
            y = np.clip(y, -5, 5)

            # Evaluate
            fy = np.array([func(yi) for yi in y])
            calls_used += n_pop

            # Sort by fitness
            order = np.argsort(fy)
            y = y[order]
            fy = fy[order]

            # Update best
            if fy[0] < best_f:
                best_f = fy[0]
                best_x = y[0].copy()

            # Compute new mean from top mu
            yw = y[:mu]
            mean_old = mean.copy()
            mean = yw.T @ wi

            # Update step size control variables
            pc = (1 - cs) * pc + cs * np.sqrt(mu_eff) * (mean - mean_old) / sigma
            hsig = int(np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * cc)
            ps = (1 - ds) * ps + hsig * ds * (mean - mean_old) / sigma

            # Update covariance
            diff = yw - mean[None, :]
            C = (1 + c1 * (dim + 1) + cmu_ - c1) * C \
                + c1 * (np.outer(mean - mean_old, mean - mean_old) + (pc * pc * (1 - (1 - cs) ** 2) / cs - C) / (1 - cs)) \
                + cmu_ / wi.sum() * (diff.T @ (wi[:, None] * diff) / sigma ** 2)

            # Enforce PSD
            try:
                B = np.linalg.cholesky(C)
                C = B @ B.T
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(dim)
                try:
                    B = np.linalg.cholesky(C)
                    C = B @ B.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)

            # Update sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 20.0)

            if calls_used >= budget:
                break

        return best_x, best_f, calls_used

    def _es11(self, func, x0, f0, budget, rng, time_limit=None):
        """Run (1+1)-ES."""
        dim = self.dim
        x = x0.copy()
        fx = f0
        sigma = 0.5
        best_x = x0.copy()
        best_f = f0

        start_time = time.time()
        calls_used = 0

        for i in range(budget):
            if time_limit is not None and (time.time() - start_time) > time_limit * 0.92:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1
            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.15
            else:
                sigma *= 0.85
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

        return best_x, best_f, calls_used

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        # Overall time guard: 330s kill, use 310s safety
        overall_start = time.time()
        overall_deadline = 310.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def track_call(x, f):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Quick random sampling (8 points) to find a good starting point
        n_rand = min(8, total_budget)
        for i in range(n_rand):
            if time.time() - overall_start > overall_deadline - 5:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            track_call(x, fx)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        # Phase 2: CMA-ES from the best random point
        remaining_budget = total_budget - calls_made
        cma_budget = int(remaining_budget * 0.65)
        if cma_budget > 10:
            x0 = best_x.copy()
            f0 = best_f
            time_left = overall_deadline - (time.time() - overall_start)
            cma_x, cma_f, cma_calls = self._cmaes(func, x0, f0, cma_budget, rng, time_limit=time_left)
            calls_made += cma_calls
            if cma_f < best_f:
                best_f = cma_f
                best_x = cma_x.copy()

        # Phase 3: (1+1)-ES from the best point for fine-tuning
        remaining_budget = total_budget - calls_made
        es_budget = max(10, remaining_budget)
        if es_budget > 5:
            time_left = overall_deadline - (time.time() - overall_start)
            x0 = best_x.copy()
            f0 = best_f
            es_x, es_f, es_calls = self._es11(func, x0, f0, es_budget, rng, time_limit=time_left)
            calls_made += es_calls
            if es_f < best_f:
                best_f = es_f
                best_x = es_x.copy()

        # Phase 4: If we still have budget and time, try a few more restarts
        remaining_budget = total_budget - calls_made
        time_left = overall_deadline - (time.time() - overall_start)
        restarts = 0
        while remaining_budget > 30 and time_left > 20 and restarts < 5:
            # Try a random restart with a small local search
            x_r = rng.uniform(-5, 5, dim)
            fx_r = func(x_r)
            calls_made += 1
            if fx_r < best_f:
                best_f = fx_r
                best_x = x_r.copy()
                # Local search from here
                local_budget = min(remaining_budget, 20)
                time_left_local = overall_deadline - (time.time() - overall_start)
                if time_left_local > 5:
                    lx, lf, lc = self._es11(func, best_x, best_f, local_budget, rng, time_limit=time_left_local)
                    calls_made += lc
                    if lf < best_f:
                        best_f = lf
                        best_x = lx.copy()
            remaining_budget = total_budget - calls_made
            time_left = overall_deadline - (time.time() - overall_start)
            restarts += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
