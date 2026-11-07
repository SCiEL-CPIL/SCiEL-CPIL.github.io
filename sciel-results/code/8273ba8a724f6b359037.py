import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def time_ok():
            return time.time() < deadline

        def count_calls(n=1):
            nonlocal total_calls
            total_calls += n

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        # Restart-based CMA-ES
        max_restarts = 5
        calls_per_restart = budget // max_restarts

        for restart in range(max_restarts):
            if total_calls >= budget or not time_ok():
                break

            # Initialize
            if restart == 0:
                x = np.zeros(dim)
            else:
                # Perturb from best found so far
                x = best_x + rng.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)

            sigma = 2.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            n_restarts = 0

            calls_in_restart = 0
            f_prev = np.inf

            while calls_in_restart < calls_per_restart and total_calls < budget and time_ok():
                # Generate lam candidates
                candidates = []
                for i in range(lam):
                    if not time_ok() or total_calls >= budget:
                        break
                    z = rng.standard_normal(dim)
                    # Use eigendecomposition of C
                    evals, evecs = np.linalg.eigh(C)
                    evals = np.sqrt(np.maximum(evals, 1e-12))
                    y = x + sigma * (evecs @ (evals * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    count_calls(1)
                    candidates.append((fy, y))
                    calls_in_restart += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(candidates) < mu:
                    break

                # Sort by fitness
                candidates.sort(key=lambda t: t[0])
                y_best = np.array([c[1] for c in candidates[:mu]])
                m = np.sum(weights[:, None] * y_best, axis=0)

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (m - x)

                # Covariance matrix update
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_in_restart + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                    delta_h = 0
                    pc = (1 - c_c) * pc + c_c * (m - x) / sigma
                else:
                    delta_h = 1
                    pc = (1 - c_c) * pc

                # Update covariance matrix
                C = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * np.outer(pc, pc)
                # Ensure positive semi-definite
                evals_C, evecs_C = np.linalg.eigh(C)
                evals_C = np.maximum(evals_C, 1e-12)
                C = evecs_C @ np.diag(evals_C) @ evecs_C.T

                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update mean
                x = m

                # Convergence check: if no improvement for many iterations, restart
                if candidates[0][0] >= f_prev - 1e-10:
                    n_restarts += 1
                else:
                    n_restarts = 0
                f_prev = candidates[0][0]

                if n_restarts > 20:
                    break

        # Final polish with Nelder-Mead if we have budget left
        remaining = budget - total_calls
        if remaining > 10 and best_x is not None and time_ok():
            try:
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining, 200), 'xatol': 1e-6, 'fatol': 1e-8}
                )
                # Count approximate calls made by NM (each iteration uses dim+1 evaluations)
                nm_calls = min(result.nfev, remaining)
                total_calls += nm_calls
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
