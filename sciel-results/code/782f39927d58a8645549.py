"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: grid warm-up for localization + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        call_count = 0
        best_x = None
        best_f = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse grid search to bracket the minimum
        # Use a 3^5 grid = 243 evaluations for initial localization
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            grid_points.append([
                                -4.0 + i * 4.0,
                                -4.0 + j * 4.0,
                                -4.0 + k * 4.0,
                                -4.0 + l * 4.0,
                                -4.0 + m * 4.0,
                            ])
        grid_points = np.array(grid_points)

        # Evaluate grid points, but stop if we hit deadline or budget
        grid_results = []
        for idx in range(len(grid_points)):
            if call_count >= budget:
                break
            if time.time() > deadline - 60:  # reserve time for CMA-ES
                break
            x = grid_points[idx]
            f = tracked_func(x)
            grid_results.append((f, x))

        if grid_results:
            # Sort by function value and take the best few for CMA-ES initialization
            grid_results.sort(key=lambda t: t[0])
            best_grid_f, best_grid_x = grid_results[0]
        else:
            best_grid_f = best_f
            best_grid_x = best_x if best_x is not None else np.zeros(dim)

        # Determine initial distribution for CMA-ES based on grid results
        # Use the best 5 grid points to estimate a local region
        if len(grid_results) >= 5:
            top_k = grid_results[:5]
            top_x = np.array([t[1] for t in top_k])
            center = np.mean(top_x, axis=0)
            # Estimate spread from top points
            spread = np.std(top_x, axis=0)
            sigma_init = np.clip(np.mean(spread), 0.5, 3.0)
        elif len(grid_results) >= 1:
            center = best_grid_x.copy()
            sigma_init = 1.5
        else:
            center = np.zeros(dim)
            sigma_init = 2.0

        # Ensure center is within bounds
        center = np.clip(center, -4.5, 4.5)

        # Phase 2: CMA-ES refinement
        # Implement a lightweight CMA-ES
        remaining_budget = budget - call_count
        if remaining_budget <= 0 or time.time() > deadline - 30:
            return best_x, best_f

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = int(mu / 2)
        ca = (mueff + 2) / (dim + mueff + 5)
        cs = (mueff + 1.7) / (dim + mueff + 2.5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial state
        mean = center.copy()
        sigma = sigma_init
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        # CMA-ES loop
        max_cma_iters = min(remaining_budget // lam, 100)
        for it in range(max_cma_iters):
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Generate population
            pop = []
            for i in range(lam):
                if call_count >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                z = np.random.randn(dim)
                y = mean + sigma * (B @ (D * z))
                y = np.clip(y, -5.0, 5.0)
                pop.append((y, tracked_func(y)))

            if not pop:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[1])

            # Update best
            if pop[0][1] < best_f:
                best_f = pop[0][1]
                best_x = pop[0][0].copy()

            # Update mean
            w = weights[:len(pop)]
            if len(pop) < mu:
                # Pad weights
                w = np.concatenate([w, np.zeros(mu - len(pop))])
            new_mean = np.zeros(dim)
            for i in range(min(mu, len(pop))):
                new_mean += w[i] * pop[i][0]
            mean = new_mean

            # Update evolution paths
            ydiff = (mean - center) / sigma
            pc = (1 - ca) * pc + np.sqrt(ca * (2 - ca) * mueff) * ydiff
            norm_pc = np.linalg.norm(pc) / np.sqrt(1 - (1 - ca) ** (it + 1))
            if norm_pc < (1.4 + 1 / dim) * chi_n:
                cs_eff = 1 - cs
            else:
                cs_eff = 1
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (
                B.T @ (D * ydiff)
            )
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))
                    < (1.4 + 1 / dim) * chi_n)
            hsig = float(hsig)

            # Update covariance matrix
            delta_sig = cs * (np.linalg.norm(ps) / chi_n - 1)
            rank_one = np.outer(ps, ps)
            rank_mu = np.zeros((dim, dim))
            for i in range(min(mu, len(pop))):
                yi = (pop[i][0] - center) / sigma
                rank_mu += w[i] * np.outer(B.T @ (D * yi), B.T @ (D * yi))

            C = ((1 + c1 * delta_sig * hsig - c1) * C
                 + c1 * (rank_one + (2 / c1 - 1) * np.eye(dim) * (1 - hsig))
                 + cmu * rank_mu) if hasattr(self, '_C') else None

            # Actually, let's maintain C properly
            if not hasattr(self, '_C_init'):
                C = np.eye(dim)
                self._C_init = True
            else:
                C = ((1 + c1 * delta_sig * hsig - c1) * self._C
                     + c1 * (rank_one + (2 / c1 - 1) * np.eye(dim) * (1 - hsig))
                     + cmu * rank_mu)
            self._C = C

            # Update B and D via eigen decomposition (done less frequently for speed)
            if it % 5 == 0 or it == 0:
                try:
                    D2, B_new = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-10))
                    B = B_new
                    D = D2
                except np.linalg.LinAlgError:
                    pass

            # Update sigma
            sigma *= np.exp(cs / (5 * dim) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

        # Final: use remaining budget for local refinement around best point
        remaining = budget - call_count
        if remaining > 0 and best_x is not None and time.time() < deadline - 5:
            local_sigma = max(sigma * 0.5, 0.1)
            for _ in range(min(remaining, 50)):
                if call_count >= budget:
                    break
                if time.time() > deadline - 2:
                    break
                y = np.clip(best_x + local_sigma * np.random.randn(dim), -5.0, 5.0)
                fy = tracked_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                else:
                    local_sigma *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))

        return best_x, best_f
# EVOLVE-BLOCK-END
