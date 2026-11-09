import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        # Parameters for (mu+lambda)-CMA-ES
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w**2).sum()

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1)) + c_sigma
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        m = np.zeros(dim)
        C = np.eye(dim)
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        calls = 0
        best_x = m.copy()
        best_f = func(best_x)
        calls += 1

        while calls + lam <= budget and time.time() < deadline:
            # Generate population
            Y = np.random.randn(lam, dim)

            # Compute sqrt(C) via eigen decomposition
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.maximum(eigval, 1e-12)
            sqrt_C = (eigvec * np.sqrt(eigval)) @ eigvec.T

            # X = m + sigma * (Y @ sqrt_C.T)
            X = m[None, :] + sigma * (Y @ sqrt_C.T)

            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lam

            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()

            # Sort
            sort_idx = np.argsort(F)
            Y_sorted = Y[sort_idx]
            X_sorted = X[sort_idx]

            # Update mean
            m_new = m + sigma * (w[:mu, None] * Y_sorted[:mu]).sum(axis=0)

            # Update evolution paths
            inv_sqrt_C = (eigvec * (1.0 / np.sqrt(eigval))) @ eigvec.T

            y_w = (w[:mu, None] * Y_sorted[:mu]).sum(axis=0)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff / dim) * (inv_sqrt_C @ y_w)

            # h_sigma
            norm_ps = np.linalg.norm(ps)
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2) + 1 / (8 * dim**4))
            h_sigma = (norm_ps / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lam + 1)))) / chi_n - (1 - c_sigma)

            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update C
            delta_h = (h_sigma - 1) * c_c * (2 - c_c)

            C = (1 - c_1 - c_mu) * C \
                + c_1 * (2 - c_1) * np.outer(pc, pc) \
                + c_1 * delta_h * np.outer(pc, pc) \
                + c_mu * np.sum(w[:mu, None, None] * np.outer(Y_sorted[:mu].T, Y_sorted[:mu].T), axis=0)

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            m = m_new

            # Clip sigma to prevent explosion
            if sigma > 10:
                sigma = 10
            if sigma < 1e-6:
                sigma = 1e-6

        # Use remaining budget for (1+1) local search
        while calls < budget and time.time() < deadline:
            y = np.random.randn(dim)
            x = m + sigma * (sqrt_C @ y) if 'sqrt_C' in locals() else m + sigma * y
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
