"""Multi-start Powell with CMA-ES initialization and L-BFGS-B polish for BBOB f12."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES for global exploration, then multi-start Powell + L-BFGS-B local refinement
    for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275  # Safety margin before 300s limit
        d = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls, best_x, best_f
            if time.time() >= deadline:
                raise Exception("timeout")
            if calls >= budget:
                raise Exception("budget")
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES for global exploration
        # Use 60% of budget for CMA-ES
        cma_budget = int(budget * 0.6)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(d))
        lam = max(10, lam)
        mu = lam // 2

        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2)**2 + mu_eff))

        # Damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(20*d*d))

        # State
        m = np.zeros(d)
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        sigma = 2.0

        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        # Main CMA-ES Loop
        while calls < cma_budget:
            if time.time() >= deadline:
                break

            # Check if we have enough budget for a full generation
            if cma_budget - calls < lam:
                break

            # Generate Population
            Y = np.zeros((lam, d))
            X = np.zeros((lam, d))
            F = np.zeros(lam)

            m_old = m.copy()

            for i in range(lam):
                if calls >= cma_budget:
                    break
                if time.time() >= deadline:
                    break

                z = np.random.randn(d)
                x_i = m + sigma * (B @ (D * z))
                x_i = np.clip(x_i, -5, 5)

                Y[i] = z
                X[i] = x_i
                try:
                    F[i] = safe_eval(x_i)
                except Exception:
                    break

            if calls >= cma_budget or time.time() >= deadline:
                break

            # Sort
            order = np.argsort(F)
            Y = Y[order]
            X = X[order]
            F = F[order]

            # Update Mean
            w_mu = w[:mu]
            m_new = np.sum(w_mu[:, None] * X[:mu], axis=0)

            # Update Paths
            c_sqrt_sigma = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            step = (m_new - m_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sqrt_sigma * w_mu[0] * (B.T @ (D * step))

            # h_sigma
            h_sigma = 1.0
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma)**(2*(calls+1)/d))
            if denom > 0:
                if norm_p_sigma / denom < (1.4 + 2/(d+1)) * chi_n:
                    h_sigma = 1.0
                else:
                    h_sigma = 0.0

            # p_c update
            c_sqrt_c = np.sqrt(c_c * (2 - c_c) * mu_eff)
            p_c = (1 - c_c) * p_c + h_sigma * c_sqrt_c * w_mu[0] * ((m_new - m_old) / sigma)

            # Update C
            V = (X[:mu] - m_old) / sigma
            factor = 1 - c_1 - c_mu + h_sigma * c_1 * (2 - c_1)
            C = factor * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c))
            C += c_mu * (V.T * w_mu) @ V

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.maximum(eigval, 1e-12)
            D = np.sqrt(eigval)
            B = eigvec

            # Update Sigma
            sigma = sigma * np.exp(1.0/d_sigma * (norm_p_sigma/chi_n - 1))

            # Update Mean
            m = m_new

            # Clamp Sigma
            if sigma < 1e-10:
                sigma = 1e-10
            if sigma > 100:
                sigma = 100

        # Phase 2: Multi-start Powell + L-BFGS-B local refinement
        remaining = budget - calls

        if remaining > 20 and best_x is not None:
            # Generate multiple starting points for Powell
            # Use CMA-ES mean, best found, and some random perturbations
            start_points = [best_x.copy(), m.copy()]

            # Add some points around the best
            for scale in [0.5, 1.0, 2.0]:
                for _ in range(2):
                    perturbed = best_x + scale * 0.1 * np.random.randn(d)
                    perturbed = np.clip(perturbed, -5, 5)
                    start_points.append(perturbed)

            # Add a few random points
            for _ in range(3):
                start_points.append(np.random.uniform(-5, 5, d))

            # For each starting point, try Powell then L-BFGS-B
            powell_budget_per_start = max(10, remaining // (len(start_points) + 1))

            for sp in start_points:
                if time.time() >= deadline:
                    break
                if calls >= budget - 10:
                    break

                # Powell method
                try:
                    result = minimize(
                        func,
                        sp,
                        method='Powell',
                        bounds=[(-5, 5)] * d,
                        options={'maxiter': powell_budget_per_start, 'xtol': 1e-10, 'ftol': 1e-12}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

                # L-BFGS-B from Powell result
                if time.time() >= deadline:
                    break
                if calls >= budget - 5:
                    break

                try:
                    result2 = minimize(
                        func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * d,
                        options={'maxiter': 20, 'maxfun': 20, 'gtol': 1e-10}
                    )
                    if result2.fun < best_f:
                        best_f = result2.fun
                        best_x = result2.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
