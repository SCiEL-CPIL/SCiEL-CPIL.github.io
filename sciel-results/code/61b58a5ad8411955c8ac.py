"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, covariance adaptation, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # Start at origin: shift is in [-4,4]^5, so origin is a good prior
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        n_calls = 1

        sigma = 2.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Adaptation parameters for covariance
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)

        # Track success rate for sigma adaptation
        successes = 0
        total = 0
        window = 10
        success_count = 0

        # Phase 1: Exploration with (1+1)-ES and covariance adaptation
        while n_calls < budget - 60:
            if time.time() > deadline:
                break

            y = x + sigma * (C_sqrt @ rng.standard_normal(dim))
            y = np.clip(y, -5, 5)
            fy = func(y)
            n_calls += 1

            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            if fy < fx:
                # Success
                successes += 1
                total += 1
                x = y
                fx = fy

                # Update sigma: increase on success
                sigma *= 1.1
                sigma = min(sigma, 10.0)

                # Update covariance
                d = (y - x) / sigma
                C = (1 - c_1) * C + c_1 * (np.outer(d, d) - C)
                C = (C + C.T) / 2

                # Periodically recompute square root
                if n_calls % 10 == 0:
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        eigvals = np.clip(eigvals, 1e-10, None)
                        C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    except:
                        C_sqrt = np.eye(dim)
                        C = np.eye(dim)
            else:
                # Failure
                total += 1
                sigma *= 0.9
                sigma = max(sigma, 1e-6)

            # Reset success counter periodically
            if total >= window:
                success_rate = successes / total
                # Adjust sigma based on success rate (1/5 rule approximation)
                if success_rate > 0.2:
                    sigma *= 1.05
                elif success_rate < 0.1:
                    sigma *= 0.95
                successes = 0
                total = 0
                sigma = np.clip(sigma, 1e-6, 10.0)

        # Phase 2: Polish with Nelder-Mead if time and budget remain
        remaining = budget - n_calls
        if remaining > 15 and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize

                call_count = [0]

                def wrapped_func(z):
                    nonlocal best_f, best_x
                    call_count[0] += 1
                    if call_count[0] > remaining:
                        return best_f
                    z = np.clip(z, -5, 5)
                    fz = func(z)
                    if fz < best_f:
                        best_f = fz
                        best_x = z.copy()
                    return fz

                # Start from current best
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
