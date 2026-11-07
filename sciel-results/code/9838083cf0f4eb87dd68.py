"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with Rank-1 Covariance Adaptation and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_fx = fx
        best_x = x.copy()

        sigma = 2.0
        C = np.eye(self.dim)

        calls = 1
        # Reserve last 100 calls for polish
        polish_budget = min(100, self.budget // 5)
        main_budget = self.budget - polish_budget

        # 1/5 rule for sigma adaptation
        window = []
        window_size = 10

        while calls < main_budget:
            if time.time() > deadline:
                break

            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                C = np.eye(self.dim)
                L = np.eye(self.dim)

            z = rng.standard_normal(self.dim)
            step = sigma * np.dot(L, z)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls += 1

            success = fy < fx
            window.append(success)
            if len(window) > window_size:
                window.pop(0)

            if success:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()

                # Update covariance towards the successful direction in standard space
                mu_c = 0.15
                C = (1 - mu_c) * C + mu_c * np.outer(z, z)
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (self.dim / trace_C)
            else:
                # Failed step: shrink sigma
                sigma *= 0.95
                sigma = np.clip(sigma, 1e-4, 10.0)

                # Occasionally re-center C to avoid drift
                if calls % 25 == 0:
                    C = 0.85 * C + 0.15 * np.eye(self.dim)
                    trace_C = np.trace(C)
                    if trace_C > 1e-10:
                        C = C * (self.dim / trace_C)

            # 1/5 rule for sigma adaptation
            if len(window) == window_size:
                success_rate = np.mean(window)
                if success_rate > 0.5:
                    sigma *= 1.05
                elif success_rate < 0.2:
                    sigma *= 0.95
                sigma = np.clip(sigma, 1e-4, 10.0)

            # Periodic eigenvalue clipping for stability
            if calls % 100 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 0.05, 20.0)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                trace_C = np.trace(C)
                if trace_C > 1e-10:
                    C = C * (self.dim / trace_C)

        # Polish phase: use Nelder-Mead style local search around best_x
        if calls < self.budget and time.time() < deadline:
            try:
                remaining = self.budget - calls
                result = minimize(func, best_x, method='Nelder-Mead',
                                  options={'maxiter': remaining, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': remaining})
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
