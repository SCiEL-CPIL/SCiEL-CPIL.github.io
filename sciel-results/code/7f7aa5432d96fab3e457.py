"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer with multi-start and local polish for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 325.0

        # Weierstrass f16 is multimodal with features at multiple scales.
        # Strategy: 
        # 1. Multi-start CMA-ES with different initializations
        # 2. Use remaining budget for local refinement
        # 3. Track best solution across all runs

        best_x = None
        best_f = np.inf

        # Number of restarts: use ~40% of budget for global search, rest for local
        # With dim=5, CMA-ES needs ~200-300 evals per run for good convergence
        n_restarts = 2
        evals_per_restart = int(budget * 0.35)  # 175 evals per restart
        local_budget = budget - n_restarts * evals_per_restart

        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 5:
                break

            # Different starting points for diversity
            if restart == 0:
                x0 = np.zeros(dim)
            else:
                rng = np.random.default_rng(42 + restart)
                x0 = rng.uniform(-3, 3, dim)

            # Run CMA-ES from x0
            x_best, f_best = self._cmaes(func, x0, evals_per_restart, dim, time_limit)
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Local polish with remaining budget
        if best_x is not None and local_budget > 0:
            if time.time() - start_time < time_limit - 3:
                best_x, best_f = self._local_polish(func, best_x, best_f, local_budget, dim, time_limit)

        return best_x, best_f

    def _cmaes(self, func, x0, budget, dim, time_limit):
        """Run CMA-ES from x0 with given budget."""
        start_time = time.time()
        time_limit = time_limit

        # CMA-ES parameters tuned for dim=5
        lam = 8  # Population size
        mu = lam // 2

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 * (mueff - 2 + 2 / mueff)) / ((dim + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

        # Initial state
        m = x0.copy()
        sigma = 2.0  # Larger initial step for multimodal landscape
        sigma_min = 1e-12
        sigma_max = 10.0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

        best_x = m.copy()
        best_f = func(best_x)
        evals_used = 1

        gen = 0
        while evals_used < budget:
            if time.time() - start_time > time_limit - 5:
                break

            # Sample population
            ys = np.random.randn(lam, dim)
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            
            xs = m[None, :] + sigma * (ys @ L.T)
            xs = np.clip(xs, -5, 5)

            # Evaluate
            fs = np.array([func(x) for x in xs])
            evals_used += lam

            # Update best
            idx_best = np.argmin(fs)
            if fs[idx_best] < best_f:
                best_f = fs[idx_best]
                best_x = xs[idx_best].copy()

            # Sort
            idx_sorted = np.argsort(fs)
            y_sorted = ys[idx_sorted]
            x_sorted = xs[idx_sorted]

            # Weighted mean
            new_m = np.dot(weights, x_sorted[:mu])

            # Update sigma
            diff = (new_m - m) / sigma
            v = np.linalg.solve(L.T, diff)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v

            hsig = 1.0 if np.linalg.norm(ps) < (1.4 + 2.0 / (dim + 1.0)) * chiN else 0.0
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Update covariance
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * diff
            rank1_update = np.outer(pc, pc)
            v_mu = (x_sorted[:mu] - m) / sigma
            rankmu_update = np.dot((weights[:, None] * v_mu).T, v_mu)
            C = (1 + c1 * hsig * (2 - hsig) - c1 - cmu) * C + c1 * rank1_update + cmu * rankmu_update
            C = (C + C.T) / 2.0

            m = new_m
            gen += 1

            # Stop if sigma is too small
            if sigma < 1e-10:
                break

        return best_x, best_f

    def _local_polish(self, func, x0, f0, budget, dim, time_limit):
        """Local refinement using coordinate descent and small random steps."""
        start_time = time.time()
        x = x0.copy()
        f = f0
        evals_used = 0

        # Adaptive step size
        step = 0.5

        while evals_used < budget:
            if time.time() - start_time > time_limit - 3:
                break

            # Try coordinate-wise moves
            improved = False
            for i in range(dim):
                if evals_used >= budget:
                    break

                # Positive step
                x_new = x.copy()
                x_new[i] += step
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                evals_used += 1

                if f_new < f:
                    x = x_new
                    f = f_new
                    improved = True
                else:
                    # Negative step
                    x_new = x.copy()
                    x_new[i] -= 2 * step
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    evals_used += 1

                    if f_new < f:
                        x = x_new
                        f = f_new
                        improved = True

            # Try small random perturbation
            if evals_used < budget:
                x_new = x + step * 0.1 * np.random.randn(dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                evals_used += 1

                if f_new < f:
                    x = x_new
                    f = f_new
                    improved = True

            # Adapt step size
            if improved:
                step *= 1.05
            else:
                step *= 0.7

            # Stop if step is very small
            if step < 1e-8:
                break

        return x, f

# EVOLVE-BLOCK-END
