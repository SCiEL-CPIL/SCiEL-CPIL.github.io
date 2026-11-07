import numpy as np
import time
from math import gamma, sqrt

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s margin to stay well under 300s/330s kill
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

        # ---- Phase 1: Sobol-based stratified global search (first ~120 calls) ----
        # Use a quasi-random sequence for better space-filling than pure random
        n_global = 120
        
        # Evaluate origin first
        if calls < budget and time_ok():
            call_func(np.zeros(dim))

        # Generate Sobol-like points using Halton sequence (simpler and effective)
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        for i in range(1, n_global):
            if calls >= budget or not time_ok():
                break
            x = np.array([halton(i, b) for b in [2, 3, 5, 7, 11]])
            # Map [0,1]^5 to [-4, 4]^5
            x = x * 8.0 - 4.0
            call_func(x)

        if calls >= budget or not time_ok():
            return best_x, best_f

        # ---- Phase 2: CMA-ES local optimization ----
        m = best_x.copy()
        sigma = 1.5  # Initial sigma: large enough to escape shallow local minima
        C = np.eye(dim)
        pc = np.zeros(dim)

        lambda_ = 14
        mu = lambda_ // 2

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

        # Stagnation tracking for potential restart
        stagnation_counter = 0
        best_f_before = best_f
        max_stagnation = 40  # If no improvement in 40 generations, consider restart

        while calls < budget and time_ok():
            m_old = m.copy()

            # Generate offspring
            xs = []
            ys = []
            fs = []

            # Precompute Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C)

            for _ in range(lambda_):
                if calls >= budget or not time_ok():
                    break
                z = rng.standard_normal(dim)
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
            fs_sorted = [fs[i] for i in indices[:mu]]

            # Update mean
            m = np.zeros(dim)
            for i in range(mu):
                m += w[i] * xs_sorted[i]

            # Update cumulation vector
            y_1 = ys_sorted[0]
            pc = (1.0 - cs) * pc + cs * np.sqrt(mueff) * y_1

            # Update sigma
            sigma *= np.exp((cs / ds) * (np.linalg.norm(pc) / chi_mean - 1.0))

            # Update covariance
            sum_ww_yy = np.zeros((dim, dim))
            for i in range(mu):
                sum_ww_yy += w[i] * np.outer(ys_sorted[i], ys_sorted[i])

            C = (1.0 - cs - cmu) * C + cs * np.outer(pc, pc) + cmu * sum_ww_yy

            # Symmetrize and regularize
            C = (C + C.T) / 2.0
            # Ensure PSD
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Stagnation check
            if best_f < best_f_before - 1e-10:
                stagnation_counter = 0
                best_f_before = best_f
            else:
                stagnation_counter += 1

            # If stagnated, perform a local restart with larger sigma
            if stagnation_counter >= max_stagnation:
                stagnation_counter = 0
                sigma *= 2.0
                # Slightly perturb the mean
                m = m + 0.5 * sigma * rng.standard_normal(dim)
                m = np.clip(m, -5.0, 5.0)
                # Reset covariance
                C = np.eye(dim)
                pc = np.zeros(dim)

        return best_x, best_f
