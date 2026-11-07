"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-stage optimizer for Lunacek bi-Rastrigin (f24):
    1. CMA-ES with restarts (iCMA-ES style) for global search
    2. L-BFGS-B polish for local convergence
    3. Time guard to stay within wall-clock limits
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # 30s safety margin

        # Track best solution
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if x.shape != (dim,):
                x = x.reshape(dim)
            val = func(x)
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # === Phase 1: CMA-ES with restarts ===
        # We implement a lightweight CMA-ES with restart strategy.
        # For dim=5 and budget=500, we can afford a few restarts.

        def cma_es(func_eval, dim, budget_cma, x0, sigma0=2.0, seed=42):
            """Simple CMA-ES implementation."""
            rng = np.random.default_rng(seed)
            x = np.array(x0, dtype=float)
            fx = func_eval(x)
            sigma = sigma0
            calls = 1

            # CMA-ES parameters
            lam = max(4, int(4 + 3 * np.log(dim)))  # population size
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = int(1 / np.sum(weights[:mu] ** 2))
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

            ps = np.zeros(dim)
            hs = np.zeros(dim)
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)
            C_sqrt = np.eye(dim)

            # Precompute for efficiency
            def mat_sqrt_inv(M):
                """Compute inverse square root of symmetric positive definite matrix."""
                eigvals, eigvecs = np.linalg.eigh(M)
                eigvals = np.clip(eigvals, 1e-20, None)
                return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            def mat_sqrt(M):
                eigvals, eigvecs = np.linalg.eigh(M)
                eigvals = np.clip(eigvals, 1e-20, None)
                return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            C_sqrt = np.eye(dim)
            C_sqrt_inv = np.eye(dim)

            for gen in range(budget_cma // lam):
                if calls >= budget_cma or time.time() > deadline - 30:
                    break

                # Sample offspring
                z = rng.standard_normal((lam, dim))
                y = x + sigma * (z @ C_sqrt.T)
                y = np.clip(y, -5, 5)
                fy = np.array([func_eval(yi) for yi in y])
                calls += lam

                # Sort by fitness
                idx = np.argsort(fy)
                y = y[idx]
                fy = fy[idx]

                # Update mean
                x_new = np.sum(weights[:, None] * y[:mu], axis=0)
                x_new = np.clip(x_new, -5, 5)

                # Update step size control
                d = np.linalg.norm(C_sqrt_inv @ (x_new - x))
                hsig = (1 - np.prod(1 - cc) ** (1 / cs)) < 1.7 + 2 / (dim + 1)
                ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ (x_new - x) / sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update covariance
                diff = y[:mu] - x
                delta = x_new - x
                C = (1 + c1 * (1 - cc)) * C + c1 * np.outer(delta, delta)
                C += cmu * np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)

                # Numerical stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-15, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                C_sqrt = mat_sqrt(C)
                C_sqrt_inv = mat_sqrt_inv(C)

                x = x_new
                fx = np.min(fy)

            return x, fx, calls

        # Use CMA-ES with a few restarts
        # Split budget: ~350 for CMA-ES, ~150 for polish
        cma_budget = min(350, budget - 50)
        if cma_budget < 20:
            cma_budget = budget - 10

        # Initial point: try a few starting points
        rng_init = np.random.default_rng(0)
        start_points = [
            np.zeros(dim),
            rng_init.uniform(-3, 3, dim),
            rng_init.uniform(-5, 5, dim),
        ]

        # Run CMA-ES from best starting point
        # First do a quick random search to pick a good start
        quick_budget = min(20, cma_budget // 10)
        if quick_budget >= 5:
            for sp in start_points:
                f_sp = eval_func(sp)
            # Pick best
            start_idx = np.argmin([eval_func(sp) for sp in start_points[:1]])  # just use first
            x_start = start_points[0]
        else:
            x_start = start_points[0]

        # Run CMA-ES
        x_cma, f_cma, cma_calls = cma_es(eval_func, dim, cma_budget - call_count, x_start, sigma0=2.0, seed=42)
        if f_cma < best_f:
            best_f = f_cma
            best_x = x_cma

        # === Phase 2: L-BFGS-B polish ===
        remaining = budget - call_count
        if remaining > 10 and best_f < np.inf:
            try:
                # Use L-BFGS-B from best point
                res = minimize(
                    func_eval,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max(5, remaining // 5), 'maxfun': max(10, remaining)}
                )
                if res.success or res.fun < best_f:
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # === Phase 3: If time and budget remain, do another CMA-ES restart from best ===
        remaining = budget - call_count
        if remaining > 40 and time.time() < deadline - 60:
            try:
                x_cma2, f_cma2, cma_calls2 = cma_es(
                    eval_func, dim, remaining, best_x, sigma0=0.5, seed=123
                )
                if f_cma2 < best_f:
                    best_f = f_cma2
                    best_x = x_cma2

                # Polish again
                remaining2 = budget - call_count
                if remaining2 > 10:
                    res = minimize(
                        func_eval,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': max(5, remaining2 // 5), 'maxfun': max(10, remaining2)}
                    )
                    if (res.success or res.fun < best_f) and np.isfinite(res.fun):
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Ensure we don't exceed budget - if we somehow did, the eval_func already guards
        # But we should stop if deadline is close
        return best_x, best_f
# EVOLVE-BLOCK-END
