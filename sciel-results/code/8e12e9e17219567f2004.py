import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        max_time = 285.0  # Safety margin before 300s limit

        # CMA-ES parameters
        lam = int(4 + 3 * np.log(n))
        if lam < 4:
            lam = 4
        mu = lam // 2

        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)

        m = np.zeros(n)
        sigma = 2.0

        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)

        rng = np.random.default_rng(42)

        evals = 0
        best_x = None
        best_f = np.inf

        # Initial evaluation
        x_init = m.copy()
        f_init = func(x_init)
        evals += 1
        best_x = x_init
        best_f = f_init

        # Pre-compute Cholesky for C = I
        C_sqrt = np.linalg.cholesky(C)
        inv_C_sqrt = np.linalg.inv(C_sqrt)

        # Main Loop
        while evals + lam <= budget:
            if time.time() - t_start > max_time:
                break

            # Generate offspring
            X = np.zeros((lam, n))
            F = np.zeros(lam)

            for i in range(lam):
                z = rng.standard_normal(n)
                # Transform to normal with covariance C
                y = C_sqrt @ z
                x = m + sigma * y
                X[i] = x
                F[i] = func(x)
                evals += 1

                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()

            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]

            # Compute weighted mean of selected points (in normalized space)
            y_weighted = np.zeros(n)
            for i in range(mu):
                y_weighted += w[i] * (X[i] - m) / sigma

            # Update Step-size Control (CSA)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (inv_C_sqrt @ y_weighted)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

            # h_sigma gate
            h_sigma = 1.0
            # Approximate the cumulative effect
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals // lam + 1))) < (1.4 + 2/(n+1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            else:
                h_sigma = 0.0

            # Update Covariance
            pc = (1 - c_c) * pc + c_c * y_weighted

            # Standard CMA-ES covariance update:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc^T + (1 - delta_sigma) * C) + c_mu * sum w_i * (z_i)(z_i)^T
            delta_sigma = c_sigma * (2 - c_sigma) * mueff

            C_new = (1 - c_1 - c_mu) * C
            C_new += c_1 * (np.outer(pc, pc) + (1 - delta_sigma) * C)

            for i in range(mu):
                z_i = (X[i] - m) / sigma
                C_new += c_mu * w[i] * np.outer(z_i, z_i)

            # Ensure positive semi-definite
            C_new = (C_new + C_new.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C_new + 1e-9 * np.eye(n))
                inv_C_sqrt = np.linalg.inv(C_sqrt)
                C = C_new
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset covariance
                C = np.eye(n)
                C_sqrt = np.linalg.cholesky(C)
                inv_C_sqrt = np.linalg.inv(C_sqrt)

            # Update Mean: m_new = m_old + sigma * y_weighted
            m = m + sigma * y_weighted

            if sigma < 1e-10 or sigma > 1e10:
                break

        # Final polish: local search around best point
        if evals < budget - 50 and time.time() - t_start < max_time - 10:
            base = best_x.copy()
            f_base = best_f
            step = max(sigma, 0.01)
            for _ in range(min(50, budget - evals)):
                if time.time() - t_start > max_time:
                    break
                improved = False
                for d in range(n):
                    for sign in [1, -1]:
                        trial = base.copy()
                        trial[d] += sign * step
                        trial = np.clip(trial, -5, 5)
                        f_trial = func(trial)
                        evals += 1
                        if f_trial < f_base:
                            f_base = f_trial
                            base = trial
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            if f_base < best_f:
                best_f = f_base
                best_x = base.copy()

        return best_x, best_f
