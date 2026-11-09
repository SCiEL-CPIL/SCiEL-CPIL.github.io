"""Rewritten optimizer: CMA-ES with multi-start initialization and polish phase."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start, adaptive step-size, and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def call_func(x):
            nonlocal calls_used, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Quick random sampling to find a good starting region
        n_init = min(40, budget // 10)
        # Include origin and some structured points
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, dim))
        
        for pt in init_points:
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            call_func(pt)

        if calls_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: CMA-ES from best point
        x = best_x.copy()
        sigma = 0.5  # Moderate initial step size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = 4 / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        def compute_C_inv_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Reserve some calls for polish
        polish_budget = min(50, budget - calls_used - lambda_)
        cma_budget = budget - polish_budget

        gen = 0
        while calls_used < cma_budget:
            if time.time() > deadline:
                break

            # Generate population
            pop = []
            pop_f = []
            for i in range(lambda_):
                if calls_used >= cma_budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C_inv_sqrt @ z)
                y_clipped = np.clip(y, -5, 5)
                fy = call_func(y_clipped)
                pop.append(y_clipped)
                pop_f.append(fy)

            if len(pop) < lambda_:
                break

            indices = np.argsort(pop_f)
            pop = [pop[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Mean adaptation
            x_old = x.copy()
            for i in range(mu):
                x = x + weights[i] * (pop[i] - x_old)

            # Evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x - x_old) / sigma
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(max(1e-15, 1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (x - x_old) / sigma

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_c * delta_h - c_c - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(pop[i] - x_old, pop[i] - x_old) for i in range(mu))
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                C = eigvecs @ np.diag(np.clip(eigvals, 1e-12, None)) @ eigvecs.T

            # Update inverse square root every 3 generations for better accuracy
            if gen % 3 == 2:
                C_inv_sqrt = compute_C_inv_sqrt(C)

            # Step-size adaptation
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 5.0)

            gen += 1

            if sigma < 1e-7:
                break

        # Phase 3: Final polish with local search (Nelder-Mead style)
        if calls_used < budget and best_x is not None and time.time() < deadline:
            # Simple coordinate descent + small perturbations
            step = 0.1
            for _ in range(min(30, budget - calls_used)):
                if time.time() > deadline or calls_used >= budget:
                    break
                improved = False
                for d in range(dim):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    for delta in [step, -step]:
                        trial = best_x.copy()
                        trial[d] += delta
                        trial = np.clip(trial, -5, 5)
                        f_trial = call_func(trial)
                        if f_trial < best_f - 1e-12:
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
