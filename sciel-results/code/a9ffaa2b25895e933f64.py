"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Strategy:
  1. CMA-ES with population size 4 and adaptive path-based covariance matrix updates
     to exploit the non-separable, different-powers structure of f14.
  2. Large initial sigma (5.0) to rapidly cover the box and identify the global basin.
  3. Restart logic: if CMA-ES stagnates, restart from best with reduced sigma or random point.
  4. Final Nelder-Mead polish around the best solution for local convergence.

The hidden affine transform (shift + rotation + offset) preserves the unimodal,
smooth nature of f14, so CMA-ES can align its covariance with the anisotropic
landscape and NM can exploit local geometry near the optimum.
"""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES + Nelder-Mead polish optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        calls = 0

        def elapsed():
            return time.time() - start

        def still_ok():
            return elapsed() < deadline

        def make_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            fx = float(func(x))
            calls += 1
            return fx

        # Initial evaluation at origin
        x0 = np.zeros(dim)
        fx0 = make_call(x0)
        if fx0 is None:
            return x0, np.inf

        best_x = x0.copy()
        best_f = fx0

        # CMA-ES implementation
        def run_cma(x_mean, f_mean, sigma_init, max_calls):
            nonlocal calls, best_x, best_f

            # CMA-ES parameters
            pop_size = 4
            mu = pop_size // 2  # number of selected individuals
            weights = np.zeros(mu)
            wi = 1.0 / mu
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Adaptation rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initialize
            x = x_mean.copy()
            sigma = sigma_init
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            no_improve = 0
            stagnation_limit = 30 * dim

            while still_ok() and calls < max_calls:
                # Generate population
                offspring = []
                f_vals = []

                for i in range(pop_size):
                    # Sample from N(x, sigma^2 * C)
                    z = rng.standard_normal(dim)
                    # Transform using Cholesky of C
                    L = np.linalg.cholesky(C)
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5.0, 5.0)
                    fy = make_call(y)
                    if fy is None:
                        return x, f_mean, sigma
                    offspring.append(y)
                    f_vals.append(fy)

                # Sort by fitness
                indices = np.argsort(f_vals)
                offspring = [offspring[i] for i in indices]
                f_vals = [f_vals[i] for i in indices]

                # Update mean
                y_mean = np.zeros(dim)
                for i in range(mu):
                    y_mean += weights[i] * offspring[i]
                x_new = x + y_mean

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C_inv_sqrt @ y_mean)
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update covariance
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                v = p_c @ C_inv_sqrt
                c_1_term = c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * (np.linalg.norm(p_c) ** 2 / chi_n ** 2 - 1) * C)
                c_mu_term = c_mu * sum(weights[i] * np.outer(offspring[i], offspring[i]) for i in range(mu))
                C = (1 + c_1_term + c_mu_term) * C

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update C_inv_sqrt via Cholesky
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L)

                x = x_new
                f_mean = min(f_vals)

                if f_mean < best_f:
                    best_f = f_mean
                    best_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= stagnation_limit:
                    break

            return x, f_mean, sigma

        # Run CMA-ES with restarts
        n_restarts = 3
        calls_per_restart = max(20, (budget - calls) // n_restarts)

        # First CMA-ES run from origin with large sigma
        x, f, sigma = run_cma(best_x, best_f, 5.0, calls + calls_per_restart)

        for r in range(1, n_restarts):
            if not still_ok() or calls >= budget:
                break
            # Restart strategy: alternate between random and near-best
            if r % 2 == 1:
                x_init = rng.uniform(-5.0, 5.0, dim)
                sigma_init = 3.0
            else:
                x_init = best_x + 1.0 * rng.standard_normal(dim)
                x_init = np.clip(x_init, -5.0, 5.0)
                sigma_init = 1.0

            f_init = make_call(x_init)
            if f_init is None:
                break

            if f_init < best_f:
                best_f = f_init
                best_x = x_init.copy()

            x, f, sigma = run_cma(x_init, f_init, sigma_init, calls + calls_per_restart)

        # Final Nelder-Mead polish
        if still_ok() and calls < budget:
            def nm_func(x_nm):
                nonlocal calls
                if calls >= budget:
                    return 1e10
                fx = float(func(x_nm))
                calls += 1
                return fx

            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Final short (1+1)-ES refinement with very small sigma
        if still_ok() and calls < budget:
            x = best_x.copy()
            f = best_f
            sigma = 0.01
            for _ in range(50):
                if not still_ok() or calls >= budget:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break
                if fy < f:
                    x = y.copy()
                    f = fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-10, 0.1)

        return best_x, float(best_f)
