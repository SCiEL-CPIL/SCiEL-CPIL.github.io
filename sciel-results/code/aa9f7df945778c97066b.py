"""CMA-ES optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with covariance matrix adaptation, multi-start, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        d = self.dim

        best_x = None
        best_f = float('inf')
        calls_used = 0
        max_calls = self.budget

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Quick multi-start to find a good initial basin
        starts = [np.zeros(d)]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, d))

        for s in starts:
            s_clipped = np.clip(s, -5, 5)
            fx = try_eval(s_clipped)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = s_clipped.copy()

        if calls_used >= max_calls or time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: CMA-ES
        def run_cmaes(x0, f0, budget_left):
            nonlocal best_x, best_f, calls_used

            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(d)))
            # lambda = population size
            lam = pop_size

            # Step size
            sigma = 0.5 * np.std(x0) if np.std(x0) > 1e-6 else 1.0
            sigma = max(0.5, min(sigma, 2.0))

            # Mean
            m = x0.copy()

            # Covariance matrix
            C = np.eye(d)

            # Inverse square root of C
            C_eigval, C_eigvec = np.linalg.eigh(C)
            C_eigval = np.clip(C_eigval, 1e-12, None)
            C_inv_sqrt = C_eigvec @ np.diag(1.0 / np.sqrt(C_eigval)) @ C_eigvec.T

            # Learning rates
            mu = lam // 2  # number of best solutions used for update
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 - mu_eff + c_sigma) / (d + 4 - mu_eff)
            c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

            # Evolution paths
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)

            # Hessian update counter
            h = 0
            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            # Track best
            local_best_x = x0.copy()
            local_best_f = f0

            # Evaluate initial population
            ys = []
            fs = []
            for i in range(lam):
                z = rng.standard_normal(d)
                y = m + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                ys.append(y)
                fs.append(fy)

            if len(ys) < lam:
                return local_best_x, local_best_f

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = np.array([ys[i] for i in idx])
            fs_sorted = np.array([fs[i] for i in idx])

            # Update best
            if fs_sorted[0] < local_best_f:
                local_best_f = fs_sorted[0]
                local_best_x = ys_sorted[0].copy()

            # Main loop
            while calls_used < max_calls and time.time() < deadline - 5:
                # Check convergence
                if sigma < 1e-8 or np.all(np.abs(C - np.eye(d)) < 1e-10):
                    break

                # Update mean
                m_new = m + sum(weights[i] * (ys_sorted[i] - m) for i in range(mu))

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                          (C_inv_sqrt.T @ ((m_new - m) / sigma))
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_d - 1))
                sigma = max(1e-10, min(sigma, 50.0))

                # Update C
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (d + 1)) * chi_d else 0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m) / sigma

                # Rank-one and rank-mu updates
                C = (1 + c_1 * h_sigma * (2 - c_1) - c_1) * C
                C += c_1 * np.outer(p_c, p_c)
                C += c_mu * sum(weights[i] * np.outer(ys_sorted[i] - m, ys_sorted[i] - m) for i in range(mu))

                # Ensure positive semi-definite
                C_eigval, C_eigvec = np.linalg.eigh(C)
                C_eigval = np.clip(C_eigval, 1e-12, None)
                C = C_eigvec @ np.diag(C_eigval) @ C_eigvec.T
                C_inv_sqrt = C_eigvec @ np.diag(1.0 / np.sqrt(C_eigval)) @ C_eigvec.T

                m = m_new

                # Generate next population
                ys = []
                fs = []
                for i in range(lam):
                    if calls_used >= max_calls or time.time() > deadline - 5:
                        break
                    z = rng.standard_normal(d)
                    y = m + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = try_eval(y)
                    if fy is None:
                        break
                    ys.append(y)
                    fs.append(fy)

                if len(ys) < lam:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                ys_sorted = np.array([ys[i] for i in idx])
                fs_sorted = np.array([fs[i] for i in idx])

                # Update best
                if fs_sorted[0] < local_best_f:
                    local_best_f = fs_sorted[0]
                    local_best_x = ys_sorted[0].copy()

            return local_best_x, local_best_f

        # Run CMA-ES from best point
        x_start = best_x.copy()
        f_start = best_f
        cma_best_x, cma_best_f = run_cmaes(x_start, f_start, max_calls - calls_used)

        if cma_best_f < best_f:
            best_f = cma_best_f
            best_x = cma_best_x.copy()

        # Phase 3: Final polish - local (1+1)-ES with small steps
        if calls_used < max_calls and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(50, max_calls - calls_used)):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.85

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
