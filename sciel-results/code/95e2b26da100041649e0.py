import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_cma_es(self, func, x0, sigma0, budget, deadline):
        """Run CMA-ES starting from x0 with initial sigma0, using up to budget calls."""
        dim = self.dim
        lam = 8
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1) + 0.5) - np.log(mu + 0.5)
        w = w / np.sum(w)
        mu_eff = np.sum(w) ** 2 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = 4.0 / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        c_1_mu = c_1 + c_mu

        m = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_x = x0.copy()
        best_f = func(x0)
        calls = 1
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (25 * dim ** 2))

        while calls + lam <= budget:
            if time.time() >= deadline:
                break

            Z = np.random.standard_normal((lam, dim))
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            Y = Z @ C_sqrt.T
            X = m + sigma * Y
            X = np.clip(X, -5, 5)

            F = np.array([func(x) for x in X])
            calls += lam

            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()

            idx_sort = np.argsort(F)
            X_sorted = X[idx_sort]

            m_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)

            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            delta = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(c_1_mu * (2 - c_1_mu)) * (C_inv_sqrt @ delta)
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = max(sigma, 1e-8)

            p_c = (1 - c_c) * p_c + c_c * delta
            Y_sorted = (X_sorted[:mu] - m) / sigma
            C_new = (1 - c_1_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C_new += c_mu * w[i] * np.outer(Y_sorted[i], Y_sorted[i])
            C_new = (C_new + C_new.T) / 2

            m = m_new
            C = C_new

        return best_x, best_f, calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget

        # Phase 1: Minimal Random Initialization
        # Use a small number of samples to pick a starting point.
        # 10 samples is enough to avoid the worst spots and estimate local scale.
        n_samples = 10
        samples = np.random.uniform(-5, 5, (n_samples, dim))
        sample_vals = np.array([func(x) for x in samples])
        best_idx = np.argmin(sample_vals)
        best_x = samples[best_idx].copy()
        best_f = sample_vals[best_idx]
        calls = n_samples

        # Estimate initial sigma from the spread of the best few samples
        # If the best samples are close, the basin might be narrow.
        # If they are spread out, the landscape is flatter or we are far from optimum.
        # A simple heuristic: use the std of the sample values or just a fixed reasonable value.
        # Inspiration 1 used: local_spread = np.std(sample_vals[sample_vals < best_f + 2 * (sample_vals.max() - sample_vals.min())])
        # This is a bit complex. Let's stick to a robust default or simple estimate.
        # For f4, the global minimum is deep. Starting sigma of 1.0 or 2.0 is usually fine.
        # Let's use 1.0 as a safe default, or slightly larger if samples are noisy.
        sigma0 = 1.0
        
        # Phase 2: CMA-ES with the majority of the budget
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            x0 = best_x.copy()
            bx, bf, used = self._run_cma_es(func, x0, sigma0, remaining, deadline)
            calls += used
            if bf < best_f:
                best_f = bf
                best_x = bx

        # Phase 3: Optional Restart if budget remains
        # This only triggers if CMA-ES finished early (e.g. time limit) or if we
        # deliberately reserved budget. In the current flow, CMA-ES consumes
        # 'remaining' budget, so calls should be close to 'budget'.
        # However, if time runs out, 'used' might be less than 'remaining'.
        if calls < budget - 10 and time.time() < deadline - 10:
            # Try one more restart with the remaining budget
            x_rand = np.random.uniform(-5, 5, dim)
            f_rand = func(x_rand)
            calls += 1
            if f_rand < best_f:
                best_f = f_rand
                best_x = x_rand.copy()
            
            if calls < budget - 10 and time.time() < deadline - 5:
                sigma_small = 1.0
                bx, bf, used = self._run_cma_es(func, x_rand, sigma_small, budget - calls, deadline)
                calls += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx

        return best_x, best_f
