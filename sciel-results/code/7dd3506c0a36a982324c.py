"""Structure-aware optimizer for BBOB f24 (Lunacek bi-Rastrigin).

Strategy:
  1. Coarse global grid over the box to locate the hidden shift region (the
     two minima of f24 are always close to each other, ~2.6 apart in the
     rotated frame, so a grid point will be near both).
  2. Local CMA-ES refinement starting from the best grid point.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid global search + CMA-ES local refinement for 5D box problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_local(self, func, x0, budget_left, deadline, rng, n):
        """Run CMA-ES from x0 until budget_left exhausted or deadline hit.

        Returns (best_x, best_f, evals_used).
        """
        if budget_left <= 0:
            return x0, func(x0), 1

        # Population size
        lam = max(4 + int(3 * np.log(n)), 2 * n + 1)
        lam = min(lam, max(4, budget_left // 3))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        x = np.array(x0, dtype=float)
        sigma = 1.5
        C = np.eye(n)
        C_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        fx = func(x)
        evals = 1
        best_x = x.copy()
        best_f = fx
        stall = 0

        while evals < budget_left:
            if time.time() > deadline:
                break

            # Sample offspring
            ys = []
            fvals = []
            for i in range(lam):
                if evals >= budget_left:
                    break
                z = rng.multivariate_normal(np.zeros(n), C, size=1).flatten()
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                ys.append(y)
                fvals.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    stall = 0
            if len(ys) == 0:
                break

            order = np.argsort(fvals)
            mu_sel = order[:mu]
            ys_sel = [ys[i] for i in mu_sel]
            x_new = np.sum([weights[i] * ys_sel[i] for i in range(mu)], axis=0)

            if not (np.any(fvals[mu_sel] < fvals[mu_sel[-1]] + 1e-8)):
                pass
            # stagnation detection: if weighted mean did not improve, reduce sigma
            if np.min(fvals) >= best_f - 1e-10:
                stall += 1
                if stall >= 3:
                    sigma *= 0.5
                    stall = 0
            else:
                stall = 0

            # Update paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv @ (x_new - x) / sigma
            hsig = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (evals + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv @ (x_new - x) / sigma

            delta_h = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 + delta_h) * C
            C -= c_1 * (np.outer(p_c, p_c) + delta_h * C)
            for i in range(mu):
                y_i = (x_new - x) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(n)

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 8.0)

            x = x_new
            C_inv = np.linalg.inv(C)

        return best_x, best_f, evals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 300s safety margin before 330s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(777)

        # ---- Phase 1: coarse global grid ----
        # For f24 the two minima are ~2.6 apart. A grid with spacing ~2.0
        # over [-5,5]^5 (6 points per dim = 7776) is too many. Use a
        # stratified/space-filling approach instead: a few levels.
        # Level 1: 3^5 = 243 points (spacing ~3.33)
        # Level 2: 5^5 = 3125 points (spacing ~2.0) -- too many for budget 500
        # So use a smarter approach: random + grid hybrid.
        #
        # Actually, budget is only 500. Let's be careful:
        # - Use a 3^5 = 243 grid (spacing ~3.33) for global search.
        # - Then CMA-ES with remaining budget (~257) for local refinement.

        # 3^5 grid
        levels = 3
        per_dim = np.linspace(-5, 5, levels)
        grid_points = []
        for a in per_dim:
            for b in per_dim:
                for c in per_dim:
                    for d in per_dim:
                        for e in per_dim:
                            grid_points.append(np.array([a, b, c, d, e]))
        grid_points = np.array(grid_points)

        # Add a few random points for diversity
        n_rand = 20
        rand_points = rng.uniform(-5, 5, size=(n_rand, n))
        all_points = np.vstack([grid_points, rand_points])

        # Evaluate all grid + random points
        best_x = None
        best_f = np.inf
        evals = 0
        for p in all_points:
            if evals >= budget:
                break
            if time.time() > deadline:
                break
            fp = func(p)
            evals += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # ---- Phase 2: CMA-ES local refinement ----
        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            bx, bf, used = self._cma_local(func, best_x, remaining, deadline, rng, n)
            evals += used
            if bf < best_f:
                best_f = bf
                best_x = bx

        return best_x, best_f
# EVOLVE-BLOCK-END
