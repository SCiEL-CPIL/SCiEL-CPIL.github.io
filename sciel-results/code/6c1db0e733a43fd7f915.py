"""CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lambda_ = int(4 + 3 * np.log(dim))  # population size
        # Standard CMA-ES parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) ** 2 + 4)
        cs = (dim + 1.7) / (dim + 9 + cc)
        c1 = 2 / ((dim + 1.3) ** 2 + 6)
        cmu = min(1 - c1, 2 * (muo - 1) / ((dim + 2) ** 2 - 2))
        muo = lambda_ // 2
        weights = np.log(muo + 0.5) - np.log(np.arange(1, muo + 1))
        weights /= weights.sum()
        mueff = int(np.sum(weights[:muo]) ** 2 / np.sum(weights ** 2))

        best_x = None
        best_f = np.inf
        calls_used = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def time_left():
            return deadline - time.time()

        # Multiple restarts: try different starting points
        # Start with a few diverse initializations
        n_restarts = 5
        # Split budget across restarts, giving more to earlier ones
        # But also allow early restarts if stuck
        base_budget_per_restart = budget // n_restarts

        for restart_idx in range(n_restarts):
            if calls_used >= budget or time_left() < 2.0:
                break

            # Choose starting point
            if restart_idx == 0:
                # First restart: random center
                x_mean = rng.uniform(-3, 3, dim)
            elif best_x is not None:
                # Subsequent restarts: perturb the best found
                x_mean = clip(best_x + rng.standard_normal(dim) * 1.0)
            else:
                x_mean = rng.uniform(-4, 4, dim)

            # Initial sigma: moderate
            sigma = 2.0

            # CMA-ES state
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            calls_this_restart = 0
            max_calls_this_restart = min(base_budget_per_restart, budget - calls_used)

            for gen in range(max_calls_this_restart):
                if calls_used >= budget or time_left() < 1.0:
                    break

                # Generate population
                B = rng.standard_normal((lambda_, dim))
                # Transform using Cholesky of C
                # Use np.linalg.cholesky for efficiency
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                Y = x_mean + sigma * (B @ L.T)
                Y = clip(Y)

                # Evaluate
                F = np.empty(lambda_)
                for i in range(lambda_):
                    if calls_used >= budget or time_left() < 1.0:
                        F[i:] = np.inf
                        break
                    F[i] = func(Y[i])
                    calls_used += 1
                    calls_this_restart += 1

                # Sort by fitness
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]

                # Update best
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = Y[0].copy()

                # Weighted mean
                y_w = np.sum(weights[:, None] * Y[:muo], axis=0)

                # Step size control (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(L) * (L @ y_w)
                hsig = (1 - np.sqrt(1 - (1 - cs) ** 2) * np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < \
                       np.sqrt((dim + 1) / (dim - 1) - 1) * (1 - cs)
                pc = (1 - cc) * pc + hsig * cc * y_w

                # Covariance update
                delta = (2 - cc) * (1 - hsig)
                C = (1 + c1 * delta - c1 - cmu) * C \
                    + c1 * np.outer(pc, pc) \
                    + cmu * np.sum(weights[:, None, None] * np.outer(Y[:muo] - y_w, Y[:muo] - y_w), axis=0)

                # Update mean
                x_mean = clip(y_w)

                # Update sigma
                chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / chi_n - 1))

                # Clamp sigma
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Early stop if convergence
                if sigma < 1e-12:
                    break

        # Optional: Nelder-Mead polish on best_x if time remains
        if best_x is not None and time_left() > 10.0 and calls_used < budget:
            try:
                from scipy.optimize import minimize
                # Simple local polish
                res = minimize(
                    lambda x: func(np.clip(x, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(50, budget - calls_used),
                             'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.success or (res.fun < best_f):
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
