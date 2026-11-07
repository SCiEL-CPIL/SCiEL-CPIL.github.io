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
        # Initial point near center
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

        calls_made = 0
        best_x = x.copy()
        best_fx = float('inf')

        # Evaluate initial point
        fx0 = func(x)
        calls_made += 1
        best_fx = fx0
        best_x = x.copy()

        def cmaes_step(x, sigma, C, C_inv, C_sqrt, C_inv_sqrt, p_sigma, p_c, rng, func):
            """One CMA-ES step. Returns updated state and best point."""
            nonlocal calls_made, best_x, best_fx
            
            # Check if we have enough budget for a full step
            if calls_made + lambda_ > self.budget:
                return None

            # Generate lambda_ candidates
            candidates = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = C_sqrt @ z
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5)
                fx = func(x_new)
                calls_made += 1
                candidates.append((fx, x_new, y))

            # Sort by fitness
            candidates.sort(key=lambda t: t[0])

            # Update best
            if candidates[0][0] < best_fx:
                best_fx = candidates[0][0]
                best_x = candidates[0][1].copy()

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

            return x_new, sigma, C, C_inv, C_sqrt, C_inv_sqrt, p_sigma, p_c

        # Run CMA-ES
        while calls_made + lambda_ <= self.budget:
            if time.time() > deadline - 10:
                break
            
            result = cmaes_step(x, sigma, C, C_inv, C_sqrt, C_inv_sqrt, p_sigma, p_c, rng, func)
            if result is None:
                break
                
            x, sigma, C, C_inv, C_sqrt, C_inv_sqrt, p_sigma, p_c = result

        # Phase 2: If we have some calls left, do a quick local polish
        if calls_made < self.budget and time.time() < deadline - 5:
            x_refine = best_x.copy()
            fx_refine = best_fx
            # Simple local search with small steps
            steps_left = self.budget - calls_made
            # Use at most 20 calls for refinement
            max_refine_calls = min(20, steps_left)
            
            # Try a few small perturbations
            for _ in range(max_refine_calls // dim): # Limit iterations
                if calls_made >= self.budget:
                    break
                improved = False
                for coord in range(dim):
                    if calls_made >= self.budget:
                        break
                    for direction in [-1, 1]:
                        if calls_made >= self.budget:
                            break
                        x_trial = x_refine.copy()
                        # Adaptive step size based on sigma or fixed small step
                        step = 0.1 * sigma if sigma > 0.01 else 0.01
                        x_trial[coord] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        ft = func(x_trial)
                        calls_made += 1
                        if ft < fx_refine:
                            x_refine = x_trial
                            fx_refine = ft
                            improved = True
                            if ft < best_fx:
                                best_fx = ft
                                best_x = x_refine.copy()
                if not improved:
                    break

        return best_x, best_fx
