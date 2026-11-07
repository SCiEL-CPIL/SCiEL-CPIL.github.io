"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned).

Combines multi-start (1+1)-ES (all 500 calls as single-point evaluations,
critical for the AOCC integral) with covariance adaptation (rank-one updates
from accepted steps) to handle the ill-conditioning of Schaffers F7.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with covariance adaptation for ill-conditioned problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _es_1plus1_cov(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        (1+1)-ES with 1/5 success rule + rank-one covariance adaptation.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        calls = 1

        # Covariance matrix (start with identity, adapt via rank-one updates)
        C = np.eye(dim)
        # Cache C^0.5 (recompute periodically)
        C_05 = np.eye(dim)
        C_05_valid = True

        # 1/5 success rule parameters
        c_inc = 1.05
        c_dec = 0.95
        n_window = 10
        success_count = 0

        best_x = x.copy()
        best_f = fx
        last_update = 0

        while calls < max_calls and time.time() < deadline:
            z = rng.standard_normal(dim)
            # Apply covariance transformation: C^0.5 @ z
            if C_05_valid:
                z_cov = C_05 @ z
            else:
                z_cov = z  # fallback to isotropic
            y = x + sigma * z_cov
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Accepted step: rank-one update of C toward the step direction
                step = (y - x) / (sigma + 1e-15)
                # Rank-one update: C <- (1 - c1) * C + c1 * outer(step, step)
                c1 = 1.0 / (dim + 1)
                C = (1.0 - c1) * C + c1 * np.outer(step, step)
                # Recompute C^0.5 every 20 accepted steps
                last_update += 1
                if last_update % 20 == 0:
                    try:
                        w, V = np.linalg.eigh(C)
                        w = np.maximum(w, 1e-10)
                        C_05 = (V * np.sqrt(w)) @ V.T
                        C_05_valid = True
                    except:
                        C_05_valid = False

                x = y
                fx = fy
                success_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                success_count -= 1 if success_count > 0 else 0

            # Adjust sigma based on success rate over window
            if calls % n_window == 0:
                success_rate = success_count / n_window
                if success_rate > 0.5:
                    sigma *= c_inc
                elif success_rate < 0.5:
                    sigma *= c_dec
                success_count = 0

            sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(123)

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # Phase 1: Initial random sampling to find good starting points
        n_samples = min(15, budget // 10)
        samples = rng.uniform(-4, 4, size=(n_samples, dim))
        sample_values = []
        for i in range(n_samples):
            if total_calls >= budget or time.time() > deadline - 5.0:
                break
            x = samples[i]
            fx = func(x)
            total_calls += 1
            sample_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Multi-start (1+1)-ES with covariance adaptation
        if len(sample_values) > 0:
            sorted_idx = np.argsort(sample_values)
            n_starts = min(4, len(sorted_idx))

            # Allocate remaining budget among starts
            remaining_budget = budget - total_calls
            budget_per_start = max(10, remaining_budget // (n_starts + 1))

            for start_i in range(n_starts):
                if total_calls >= budget or time.time() > deadline - 5.0:
                    break

                x0 = samples[sorted_idx[start_i]].copy()
                # Estimate initial sigma from sample spread
                sigma0 = max(0.5, np.std(samples, axis=0).mean())

                x_best, f_best, calls_used = self._es_1plus1_cov(
                    func, rng, x0, sigma0, budget_per_start, deadline
                )
                total_calls += calls_used

                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()

        # Phase 3: Final local polish from the best point (high-precision)
        remaining = budget - total_calls
        if remaining > 5 and time.time() < deadline - 5.0:
            x_local = best_x.copy()
            fx_local = best_f
            sigma_local = 0.05  # tighter initial sigma for polish

            local_calls = 0
            while local_calls < remaining and time.time() < deadline - 5.0:
                z = rng.standard_normal(dim)
                y = x_local + sigma_local * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                local_calls += 1
                total_calls += 1

                if fy < fx_local:
                    x_local = y
                    fx_local = fy
                    sigma_local *= 1.08
                    if fx_local < best_f:
                        best_f = fx_local
                        best_x = x_local.copy()
                else:
                    sigma_local *= 0.92

                sigma_local = np.clip(sigma_local, 1e-9, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
