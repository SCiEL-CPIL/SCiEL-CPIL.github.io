"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_run(self, func, x0, sigma, max_evals, rng, deadline):
        """Run CMA-ES for up to max_evals evaluations or until deadline."""
        n = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        evals = 1
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))
        lambda_ = pop_size
        mu = lambda_ // 2  # number of selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_1 = (2 + 2.0 / (n + 1.3)) / ((n + 2) ** 1.7 + mu_eff)
        c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1.0 / mu_eff)) / ((n + 2) ** 1.7 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        step_size = sigma
        eval_count = 1

        # Precompute weights for ranking
        w = weights

        while eval_count < max_evals:
            if time.time() > deadline:
                break

            # Sample population
            ys = np.zeros((lambda_, n))
            fs = np.zeros(lambda_)
            for i in range(lambda_):
                z = rng.standard_normal(n)
                # Transform using Cholesky of C
                y = x + step_size * C_inv_sqrt @ z
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                eval_count += 1
                if eval_count >= max_evals:
                    break
                if time.time() > deadline:
                    break

            if eval_count >= max_evals or time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sel = ys[idx[:mu]]
            fs_sel = fs[idx[:mu]]

            # Update best
            if fs_sel[0] < best_fx:
                best_fx = fs_sel[0]
                best_x = ys_sel[0].copy()

            # New mean
            x_new = np.sum(ys_sel * w[:, None], axis=0)

            # Update step size (CSA)
            y_mean = (x_new - x) / (step_size * chi_n) if step_size > 1e-15 else np.zeros(n)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, eval_count)))) < (1.4 + 2 / (n + 1)) * chi_n
            step_size *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            step_size = np.clip(step_size, 1e-15, 1e3)

            # Update C (covariance matrix adaptation)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / step_size
            rank_c = (1 - c_1) * (C + (1 + (1 - c_1) * hsig * (1 - c_c)) * np.outer(pc, pc))
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                zi = (ys_sel[i] - x) / step_size if step_size > 1e-15 else np.zeros(n)
                rank_mu += w[i] * np.outer(zi, zi)
            C = rank_c + c_mu * rank_mu

            # Update inverse square root via eigen decomposition (for small n)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            x = x_new

        return best_x, best_fx, eval_count

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start with adaptive restarts
        best_x = None
        best_fx = np.inf
        total_evals = 0
        num_restarts = 0
        max_restarts = 5

        # Initial broad exploration: try several random starts
        # First, do a quick random search to find a good starting point
        n_initial = min(20, budget)
        candidates = []
        for i in range(n_initial):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_evals += 1
            candidates.append((fx, x))
            if time.time() > deadline:
                break

        candidates.sort(key=lambda t: t[0])
        best_fx = candidates[0][0]
        best_x = candidates[0][1].copy()

        # Now run CMA-ES from the best few starting points
        starts = [c[1] for c in candidates[:min(3, len(candidates))]]

        for s_idx, start in enumerate(starts):
            if total_evals >= budget or time.time() > deadline:
                break

            remaining = budget - total_evals
            if remaining < 10:
                break

            # Adaptive sigma based on initial spread
            sigma = 1.0 / np.sqrt(dim)

            x_best, fx_best, evals_used = self._cma_run(
                func, start, sigma, remaining, rng, deadline
            )
            total_evals += evals_used

            if fx_best < best_fx:
                best_fx = fx_best
                best_x = x_best.copy()

            # Check if we should restart with different strategy
            if total_evals >= budget or time.time() > deadline:
                break

            # If progress was slow, try a different approach
            # Simple local refinement around best point
            if evals_used > 0:
                # Do a few more evaluations with smaller steps
                local_budget = min(20, budget - total_evals)
                for _ in range(local_budget):
                    if time.time() > deadline:
                        break
                    step = 0.1 / (s_idx + 1)
                    y = np.clip(best_x + step * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    total_evals += 1
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()

        # Final local polish
        if total_evals < budget and time.time() < deadline:
            polish_budget = min(30, budget - total_evals)
            for i in range(polish_budget):
                if time.time() > deadline:
                    break
                scale = 0.05 * (1.0 - i / polish_budget) + 0.01
                y = np.clip(best_x + scale * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_evals += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
