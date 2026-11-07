"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with local refinement for Rastrigin (f3)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.8) ** 2 + mu_eff)
        c_mu = 2 / ((dim + 1.3) ** 2 + mu_eff)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        x_mean = np.zeros(dim)
        sigma = 1.5
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        # Precompute matrix square root
        eigvals, eigvecs = np.linalg.eigh(C)
        sqrt_C = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial evaluation
        best_f = safe_call(x_mean)

        # CMA-ES main loop
        gen = 0
        while call_count < budget:
            if time.time() > deadline - 15.0:
                break

            # Generate and evaluate population
            ys = []
            fvals = []
            for i in range(lam):
                if call_count >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (sqrt_C @ z)
                ys.append(y)
                fvals.append(safe_call(y))
            gen += 1

            if len(ys) < mu:
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            y_best = np.array([ys[i] for i in indices[:mu]])
            f_best = [fvals[i] for i in indices[:mu]]

            # Save old mean
            x_old = x_mean.copy()

            # Update mean
            x_mean = weights @ y_best

            # Update step-size control
            y_sigma = (y_best - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (weights @ y_sigma.T)
            norm_p_sigma = np.linalg.norm(p_sigma)
            if norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * gen)) < (1.4 + 1 / dim) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))

            # Update evolution path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (weights @ ((y_best - x_old) / sigma).T)

            # Update covariance matrix
            C = (1 + c_1 * c_mu - c_1 - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * C) \
                + c_mu * (y_sigma.T @ np.diag(weights) @ y_sigma)

            # Ensure positive semi-definite
            if gen % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-15)) @ eigvecs.T
                sqrt_C = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T

            # Clamp sigma
            sigma = float(np.clip(sigma, 1e-10, 10.0))

        # Local refinement
        if best_x is not None and call_count < budget and time.time() < deadline - 5.0:
            x = best_x.copy()
            fx = best_f

            # Coordinate descent with adaptive step
            step = 0.5
            for _ in range(100):
                if call_count >= budget or time.time() > deadline - 2.0:
                    break
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        x_try = x.copy()
                        x_try[d] = np.clip(x_try[d] + sign * step, -5, 5)
                        f_try = safe_call(x_try)
                        if f_try < fx - 1e-12:
                            x, fx = x_try, f_try
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-7:
                        break

            # Nelder-Mead polish
            if call_count < budget - 30 and time.time() < deadline - 3.0:
                from scipy.optimize import minimize
                try:
                    res = minimize(func, x, method='Nelder-Mead',
                                   options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                    if res.fun < fx:
                        x = np.clip(res.x, -5, 5)
                        fx = safe_call(x)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
