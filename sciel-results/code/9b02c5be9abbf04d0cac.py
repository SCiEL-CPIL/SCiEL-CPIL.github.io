import numpy as np
import time
from math import gamma, sqrt, log

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # ---- Phase 1: Structured global search with multiple scales ----
        # Büche-Rastrigin has many local minima. We need good coverage.
        # Use a combination of Halton sequence and random sampling at different scales.

        # Evaluate origin first
        if calls < budget and time_ok():
            call_func(np.zeros(dim))

        # Halton sequence for low-discrepancy sampling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate Halton points mapped to [-4, 4]^5
        n_halton = 80
        for i in range(1, n_halton + 1):
            if calls >= budget or not time_ok():
                break
            x = np.array([halton(i, b) for b in [2, 3, 5, 7, 11]])
            x = x * 8.0 - 4.0
            call_func(x)

        # Add random points for diversity
        n_random = 40
        for _ in range(n_random):
            if calls >= budget or not time_ok():
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            call_func(x)

        if calls >= budget or not time_ok():
            return best_x, best_f

        # ---- Phase 2: Multi-start CMA-ES with restarts ----
        # Use the best point found so far as the starting point for CMA-ES
        # Implement a simplified but effective CMA-ES with adaptive step size

        def cma_es(m_init, sigma_init, max_iter, lambda_, mu, seed):
            nonlocal calls, best_x, best_f
            rng_local = np.random.default_rng(seed)

            m = m_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            pc = np.zeros(dim)

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w**2)

            # Rates
            cs = (mueff + 2.0) / (dim + mueff + 5.0)
            ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
            cmu = min(1.0 - cs, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0)**2 + mueff))

            # Chi mean
            chi_mean = sqrt(2.0) * gamma((dim + 1.0) / 2.0) / gamma(dim / 2.0)

            stagnation = 0
            best_f_local = best_f
            max_stagnation = 20

            for _ in range(max_iter):
                if calls >= budget or not time_ok():
                    break

                m_old = m.copy()

                # Generate offspring
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    L = np.linalg.cholesky(C)

                xs = []
                ys = []
                fs = []

                for _ in range(lambda_):
                    if calls >= budget or not time_ok():
                        break
                    z = rng_local.standard_normal(dim)
                    y = L @ z
                    x = m_old + sigma * y
                    f_val = call_func(x)
                    if f_val is not None:
                        xs.append(x)
                        ys.append(y)
                        fs.append(f_val)

                if len(xs) < mu:
                    break

                # Sort by fitness
                indices = np.argsort(fs)
                xs_sorted = [xs[i] for i in indices[:mu]]
                ys_sorted = [ys[i] for i in indices[:mu]]

                # Update mean
                m = np.zeros(dim)
                for i in range(mu):
                    m += w[i] * xs_sorted[i]

                # Update cumulation
                y_1 = ys_sorted[0]
                pc = (1.0 - cs) * pc + cs * np.sqrt(mueff) * y_1

                # Update sigma
                sigma *= np.exp((cs / ds) * (np.linalg.norm(pc) / chi_mean - 1.0))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance
                sum_ww_yy = np.zeros((dim, dim))
                for i in range(mu):
                    sum_ww_yy += w[i] * np.outer(ys_sorted[i], ys_sorted[i])

                C = (1.0 - cs - cmu) * C + cs * np.outer(pc, pc) + cmu * sum_ww_yy
                C = (C + C.T) / 2.0

                # Regularize
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-12:
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Stagnation check
                if best_f < best_f_local - 1e-10:
                    stagnation = 0
                    best_f_local = best_f
                else:
                    stagnation += 1

                if stagnation >= max_stagnation:
                    break

            return m, sigma

        # Multi-start: try CMA-ES from the best point and a few random points
        # First, try from the best point found in global search
        m_init = best_x.copy()
        sigma_init = 1.5
        lambda_ = 14
        mu = lambda_ // 2

        # Run CMA-ES from best point
        m_init, sigma_init = cma_es(m_init, sigma_init, 30, lambda_, mu, 42)

        if calls >= budget or not time_ok():
            return best_x, best_f

        # Try a few more starts from other good points
        # Use points from the global search that had relatively good fitness
        # We don't have stored fitness values, so use random perturbations of best_x
        for restart in range(3):
            if calls >= budget or not time_ok():
                break

            # Create a new starting point near the best
            m_new = best_x + rng.normal(0, 0.5, size=dim)
            m_new = np.clip(m_new, -4.5, 4.5)

            m_new, _ = cma_es(m_new, 1.0, 20, lambda_, mu, 100 + restart)

        if calls >= budget or not time_ok():
            return best_x, best_f

        # ---- Phase 3: Fine-tuning with smaller step sizes ----
        # Do a final CMA-ES run with smaller sigma for precision
        m_init = best_x.copy()
        sigma_init = 0.3
        m_init, _ = cma_es(m_init, sigma_init, 15, 10, 5, 200)

        return best_x, best_f
