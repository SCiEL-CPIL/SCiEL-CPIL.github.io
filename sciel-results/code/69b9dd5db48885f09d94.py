"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        # Best solution tracking
        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                calls_made += 1
                if np.isfinite(f) and f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        # CMA-ES implementation
        def cma_es_run(x0, sigma0, max_iter):
            nonlocal calls_made, best_x, best_f, deadline
            n = self.dim
            # Population size
            m = int(4 + 3 * np.log(n))
            # Strategy parameters
            cs = (m / 2 + n) ** -0.5
            ds = 1 + 2 * max(0, np.sqrt((m / 2 + n - 1) / (n + 1) - 1)) + cs
            c1 = (2 / ((n + 1.3) ** 2 + m)) * (2 - (2 / ((n + 1.3) ** 2 + m)))
            cmu = min(1 - c1, 2 * ((m / 2 - 1 + 1 / n) / ((n + 2) ** 2 + m)))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initialize
            x = np.array(x0, dtype=np.float64)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            steps = 0

            for it in range(max_iter):
                # Check time
                if time.time() > deadline:
                    break
                # Check budget
                if calls_made >= max_calls:
                    break

                # Generate population
                ws = np.zeros(m)
                for i in range(m):
                    if calls_made >= max_calls:
                        break
                    z = np.random.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    fy = safe_call(y)
                    if fy is None:
                        continue
                    ws[i] = fy

                if calls_made >= max_calls:
                    break

                # Sort by fitness
                valid = np.isfinite(ws)
                if np.sum(valid) < 2:
                    break
                ws_valid = ws[valid]
                ys_valid = []
                # Regenerate to get corresponding ys (we need to store them)
                # Actually, let's restructure to store ys
                break  # This simple version won't work well, let's use a different approach

            return x, sigma

        # Better CMA-ES with stored population
        def cma_es_v2(x0, sigma0, max_iter):
            nonlocal calls_made, best_x, best_f, deadline
            n = self.dim
            m = int(4 + 3 * np.log(n))
            cs = (m / 2 + n) ** -0.5
            ds = 1 + 2 * max(0, np.sqrt((m / 2 + n - 1) / (n + 1) - 1)) + cs
            c1 = (2 / ((n + 1.3) ** 2 + m)) * (2 - (2 - (2 / ((n + 1.3) ** 2 + m))))
            cmu = min(1 - c1, 2 * ((m / 2 - 1 + 1 / n) / ((n + 2) ** 2 + m)))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x = np.array(x0, dtype=np.float64)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            for it in range(max_iter):
                if time.time() > deadline or calls_made >= max_calls:
                    break

                # Generate population
                ys = []
                fs = []
                for i in range(m):
                    if calls_made >= max_calls or time.time() > deadline:
                        break
                    z = np.random.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    fy = safe_call(y)
                    if fy is not None:
                        ys.append(y)
                        fs.append(fy)

                if len(ys) < 2:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                fs = [fs[i] for i in order]

                # Update mean
                weights = np.log(m / 2 + 1) - np.log(np.arange(1, m + 1))
                weights = weights / np.sum(weights)
                mu_eff = 1.0 / np.sum(weights[:m] ** 2)
                x_new = np.sum([weights[i] * ys[i] for i in range(min(m, len(ys)))], axis=0)

                # Update covariance matrix
                ydiff = np.array([y - x for y in ys[:m]])
                C_new = (1 - c1 - cmu) * (B @ np.diag(D ** 2) @ B.T)
                C_new += c1 * (pc @ pc.T)
                C_new += cmu / mu_eff * np.sum([np.outer(w * yd, w * yd) for w, yd in zip(weights[:m], ydiff)], axis=0)

                # Update step size
                ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) / (chi_n * sigma) * (B @ (D * (x_new - x)))
                hsig = (1 - np.exp(-cs * ds)) * (1 - np.exp(-2 * cs * ds)) if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) < (1 + 1 / cs) * chi_n else 0
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))

                # Update eigen decomposition
                pc = (1 - cs) * pc + hsig * cs * (x_new - x) / sigma
                x = x_new

                # Re-orthogonalize B and D
                try:
                    eigvals, eigvecs = np.linalg.eigh(C_new)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except:
                    break

                if time.time() > deadline or calls_made >= max_calls:
                    break

            return x, sigma

        # Multi-start CMA-ES
        np.random.seed(42)
        num_restarts = 5
        budget_per_restart = max(50, max_calls // num_restarts)

        for restart in range(num_restarts):
            if calls_made >= max_calls or time.time() > deadline:
                break

            if restart == 0:
                x0 = np.random.uniform(-5, 5, self.dim)
                sigma0 = 2.0
            else:
                # Perturb best solution found so far
                x0 = best_x + np.random.uniform(-1, 1, self.dim)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.0

            cma_es_v2(x0, sigma0, budget_per_restart)

        return best_x, best_f
# EVOLVE-BLOCK-END
