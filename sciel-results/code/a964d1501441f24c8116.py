import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin before 330s kill

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return 1e12
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = func(x_clipped)
            except Exception:
                fx = 1e12
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # --- Phase 1: CMA-ES for global search ---
        # Parameters tuned for dim=5
        lam = 8
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        # Initial state
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))

        # Initial evaluation
        safe_call(x_mean)

        # CMA-ES loop
        # We use a subset of the budget for CMA-ES, say 70%
        cma_budget = int(0.7 * budget)

        while calls_made < cma_budget and time.time() < deadline:
            # 1. Compute C^{1/2} and C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T

            # 2. Sample population
            ys = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                z = np.random.default_rng(42 + calls_made).standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                ys.append(y)

            if not ys:
                break

            # 3. Evaluate
            fs = [safe_call(y) for y in ys]

            if len(ys) < mu:
                break

            # 4. Sort
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            fs = [fs[i] for i in order]

            # 5. Update Mean
            x_mean_old = x_mean.copy()
            x_mean = np.zeros(dim)
            for i in range(mu):
                x_mean += weights[i] * ys[i]

            # 6. Update Step Size
            diff = x_mean - x_mean_old
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_inv_sqrt @ diff)
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1))

            # 7. Update Covariance
            norm_pc = np.linalg.norm(pc)
            h_sigma = (norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1))) < (1 + 2 / (dim + 1)) * chi_n)

            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (C_inv_sqrt @ diff)

            rank1 = np.outer(pc, pc)
            if not h_sigma:
                rank1 += (1 - h_sigma) * c_c * (2 - c_c) * C

            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = (ys[i] - x_mean_old) / sigma
                rank_mu += weights[i] * np.outer(diff_i, diff_i)

            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            C = (C + C.T) / 2

            # Regularize
            eigvals, _ = np.linalg.eigh(C)
            if np.any(eigvals < 1e-12):
                C = C + (1e-12 - np.min(eigvals)) * np.eye(dim)

        # --- Phase 2: (1+1)-ES for local refinement ---
        # Use the remaining budget
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with a moderate step size for local search

            # (1+1)-ES with 1/5 success rule
            c_sigma_es = 1.0 / (dim + 4.0)
            h = 0.84 + 2.0 / (dim + 1.0)
            sigma_exp = 0.0
            rng_local = np.random.default_rng(999)

            while calls_made < budget and time.time() < deadline:
                y = np.clip(x + sigma * (1.0 + sigma_exp) * rng_local.standard_normal(dim), -5.0, 5.0)
                fy = safe_call(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma_exp = (1.0 - c_sigma_es) * sigma_exp + c_sigma_es * h
                else:
                    sigma_exp = (1.0 - c_sigma_es) * sigma_exp

                sigma = np.exp(sigma_exp)
                sigma = float(np.clip(sigma, 1e-5, 5.0))

        return best_x, best_f
