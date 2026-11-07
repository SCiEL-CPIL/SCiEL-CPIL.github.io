import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Phase 1: Find good starting points
        n_starts = 20
        starts = rng.uniform(-5, 5, (n_starts, dim))
        f_starts = np.array([func(x) for x in starts])
        
        # Track calls
        calls_used = n_starts
        
        # Get top 2 best starts
        top_idx = np.argsort(f_starts)[:2]
        top_starts = starts[top_idx]
        top_f = f_starts[top_idx]
        
        # Update global best
        min_idx = np.argmin(top_f)
        best_x = top_starts[min_idx].copy()
        best_f = top_f[min_idx]

        # Phase 2: Run CMA-ES from the best starts
        # Split remaining budget
        remaining = budget - calls_used
        # We have 2 starts. Let's give more budget to the better one?
        # Or just split evenly.
        budget_per_run = remaining // 2
        
        # Run 1 from best start
        if budget_per_run > 10:
            bx, bf, calls = self._cmaes(func, top_starts[0], 2.0, budget_per_run, rng, deadline)
            calls_used += calls
            if bf < best_f:
                best_f = bf
                best_x = bx
        
        # Run 2 from second best start
        if budget_per_run > 10 and calls_used < budget:
            bx, bf, calls = self._cmaes(func, top_starts[1], 2.0, budget - calls_used, rng, deadline)
            calls_used += calls
            if bf < best_f:
                best_f = bf
                best_x = bx

        return best_x, best_f

    def _cmaes(self, func, x0, sigma0, max_calls, rng, deadline):
        # Standard CMA-ES implementation
        n = self.dim
        lam = 10
        mu = lam // 2
        
        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_w = (w ** 2).sum()
        mu_eff = 1.0 / mu_w
        
        # Parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 1) / (mu_eff + 2) ** 2 - 1))
        chi_n = np.sqrt(n * (1 - 1 / (4 * n) - 1 / (21 * n ** 2)))
        
        # State
        mean = x0.copy()
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        
        # Precompute C_sqrt
        # C = I, so C_sqrt = I
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)
        
        calls = 0
        best_x = mean.copy()
        best_f = func(mean)
        calls = 1
        
        if calls >= max_calls:
            return best_x, best_f, calls
            
        while calls < max_calls:
            if time.time() > deadline:
                break
                
            # Sample
            Z = rng.standard_normal((lam, n))
            Y = Z @ C_sqrt
            X = mean[None, :] + sigma * Y
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                if calls >= max_calls:
                    break
                F[i] = func(X[i])
                calls += 1
            if calls < lam:
                F[calls:] = np.inf
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            
            # Update mean
            mean_old = mean.copy()
            mean = mean + sigma * np.sum(w[:, None] * X_sorted[:mu], axis=0)
            
            # Update ps
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_w) / sigma * (C_sqrt_inv @ (mean - mean_old))
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1)))) < (1 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * c_c * (mean - mean_old) / sigma
            
            # Update C
            delta_sig = hsig * (1 - c_c) + c_c
            C = (1 + c_1 + c_mu - c_1 - c_mu) * C + c_1 * (delta_sig * n - 1) * C + c_mu * (np.outer(mean - mean_old, mean - mean_old) / sigma ** 2 - C)
            # Wait, the standard update is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (n + delta_sig - 1) * C + c_mu * (1/sigma^2) * outer(mean-mean_old, mean-mean_old)
            # Let's use the correct formula from Hansen's paper.
            # C = (1 - c_1 - c_mu) * C + c_1 * (n + delta_sig - 1) * C + c_mu * (1/sigma^2) * outer(mean-mean_old, mean-mean_old)
            # Note: (1 - c_1 - c_mu) + c_1 * (n + delta_sig - 1) is not necessarily 1.
            # The term c_1 * (n + delta_sig - 1) * C is the "memory" term for the covariance.
            
            # Let's re-evaluate the C update formula carefully.
            # C_{t+1} = (1 - c_1 - c_mu) C_t + c_1 (n + delta_{hsig} - 1) C_t + c_mu (1/sigma^2) (mean_{t+1} - mean_t)(mean_{t+1} - mean_t)^T
            # This simplifies to:
            # C_{t+1} = (1 - c_1 - c_mu + c_1(n + delta - 1)) C_t + c_mu (1/sigma^2) outer(...)
            # Coeff of C_t: 1 - c_mu + c_1(n + delta - 1)
            
            coeff_c = 1 - c_mu + c_1 * (n + delta_sig - 1)
            C = coeff_c * C + c_mu * (np.outer(mean - mean_old, mean - mean_old) / sigma ** 2)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update C_sqrt
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            B = eigvecs
            C_sqrt = (B * D) @ B.T
            C_sqrt_inv = (B / D) @ B.T
            
            # If sigma is too small, break
            if sigma < 1e-10:
                break
                
        return best_x, best_f, calls
