"""Rewritten optimizer for BBOB f15 (Rastrigin rotated) with CMA-ES + coordinate refinement."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with coordinate refinement and Nelder-Mead polish for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: CMA-ES for global exploration
        # Initial point near center (Rastrigin minimum is at 0 in standard form,
        # but hidden shift/rotation means we need broad search)
        x = np.zeros(dim)
        sigma = 2.5  # initial step size
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # CMA-ES parameters
        mu = max(1, dim // 2)
        lambda_ = 2 * mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cmaes_step(x, sigma, C, C_inv, C_sqrt, C_inv_sqrt, p_sigma, p_c, rng, func):
            """One CMA-ES step. Returns updated state and best point."""
            # Generate lambda_ candidates
            candidates = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5)
                fx = func(x_new)
                candidates.append((fx, x_new, y))

            # Sort by fitness
            candidates.sort(key=lambda t: t[0])

            # Update mean
            x_new = sum(weights[i] * candidates[i][1] for i in range(mu))
            x_new = np.clip(x_new, -5, 5)

            # Compute evolution path for sigma
            y_w = sum(weights[i] * candidates[i][2] for i in range(mu))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_w
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2)) / chi_n < (1.4 + 2 / (dim + 1))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update path for covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x)

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            outer = np.outer(p_c, p_c)
            C = (1 + c_1 * (dim + 1) + c_mu - c_1 - delta_h) * C \
                + c_1 * (outer - C) \
                + c_mu * sum(weights[i] * np.outer(candidates[i][2], candidates[i][2]) for i in range(mu))

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2  # Symmetrize

            # Update inverse and square roots
            try:
                C_inv = np.linalg.inv(C)
                C_inv = (C_inv + C_inv.T) / 2
                eigvals_inv, eigvecs_inv = np.linalg.eigh(C_inv)
                eigvals_inv = np.maximum(eigvals_inv, 1e-12)
                C_inv = eigvecs_inv @ np.diag(eigvals_inv) @ eigvecs_inv.T

                eigvals_c, eigvecs_c = np.linalg.eigh(C)
                eigvals_c = np.maximum(eigvals_c, 1e-12)
                C_sqrt = eigvecs_c @ np.diag(np.sqrt(eigvals_c)) @ eigvecs_c.T
                C_inv_sqrt = eigvecs_inv @ np.diag(np.sqrt(eigvals_inv)) @ eigvecs_inv.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_inv_sqrt = np.eye(dim)

            best_fx, best_x, _ = candidates[0]
            return x_new, sigma, C, C_inv, C_sqrt, C_inv_sqrt, p_sigma, p_c, best_fx, best_x

        # Track best
        best_x = x.copy()
        best_fx = func(x)
        calls_made = 1

        # Run CMA-ES for a portion of the budget
        cma_budget = min(300, self.budget - 50)  # Leave some for refinement
        for step in range(cma_budget):
            if time.time() > deadline - 60:
                break
            x, sigma, C, C_inv, C_sqrt, C_inv_sqrt, p_sigma, p_c, step_best_fx, step_best_x = \
                cmaes_step(x, sigma, C, C_inv, C_sqrt, C_inv_sqrt, p_sigma, p_c, rng, func)
            calls_made += lambda_

            if step_best_fx < best_fx:
                best_fx = step_best_fx
                best_x = step_best_x.copy()

            if calls_made >= self.budget:
                break

        # Phase 2: Coordinate descent refinement
        if calls_made < self.budget and time.time() < deadline - 30:
            x_refine = best_x.copy()
            fx_refine = best_fx
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
            for ss in step_sizes:
                for coord in range(dim):
                    if calls_made >= self.budget or time.time() > deadline - 20:
                        break
                    for direction in [-1, 1]:
                        x_trial = x_refine.copy()
                        x_trial[coord] += direction * ss
                        x_trial = np.clip(x_trial, -5, 5)
                        calls_made += 1
                        ft = func(x_trial)
                        if ft < fx_refine:
                            x_refine = x_trial
                            fx_refine = ft
                            if ft < best_fx:
                                best_fx = ft
                                best_x = x_refine.copy()
                if calls_made >= self.budget or time.time() > deadline - 20:
                    break

        # Phase 3: Nelder-Mead polish if budget remains
        if calls_made < self.budget - 20 and time.time() < deadline - 10:
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead with tight bounds handled by clipping
                def bounded_func(x):
                    nonlocal calls_made
                    x_clip = np.clip(x, -5, 5)
                    calls_made += 1
                    return func(x_clip)

                nm_result = minimize(bounded_func, best_x, method='Nelder-Mead',
                                     options={'maxiter': max(1, self.budget - calls_made - 5),
                                              'xatol': 1e-6, 'fatol': 1e-8,
                                              'adaptive': True})
                if nm_result.fun < best_fx:
                    best_fx = nm_result.fun
                    best_x = np.clip(nm_result.x, -5, 5)
            except Exception:
                pass

        return best_x, best_fx
