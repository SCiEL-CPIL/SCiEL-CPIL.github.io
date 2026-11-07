import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters tuned for dim=5, budget=500
        # Larger population for better exploration early
        lam = 15
        mu = lam // 2  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates (Standard CMA-ES)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Chi-square approximation for sigma adaptation
        chi_lambda = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

        # Initial state: Start at origin
        m = np.zeros(dim)
        sigma = 2.0  # Larger initial step for exploration in [-5,5]^5
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute Cholesky factor of C
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)

        best_x = m.copy()
        best_f = func(m)
        evals = 1
        n_calls = 1

        # Stagnation tracking
        stagnation_count = 0
        stagnation_threshold = 10  # Generations without improvement

        rng = np.random.default_rng(42)

        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # We already made 1 call for initial point
        # best_f is already set

        while n_calls + lam <= budget and time.time() < deadline:
            # Generate population
            # Sample Z
            Z = rng.standard_normal((lam, dim))
            # Sample X = m + sigma * Z @ L.T
            X = m + sigma * (Z @ L.T)
            
            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([timed_call(x) for x in X])
            
            if n_calls >= budget:
                break

            # Sort
            order = np.argsort(F)
            F_sorted = F[order]
            X_sorted = X[order]

            # Update best is already handled in timed_call, but ensure consistency
            # (timed_call updates best_f/best_x)

            # Stagnation check
            if F_sorted[0] >= best_f:
                stagnation_count += 1
            else:
                stagnation_count = 0

            # If stagnated, restart or shrink sigma significantly
            if stagnation_count > stagnation_threshold:
                # Try a local random restart around best_x
                if n_calls + 5 <= budget:
                    for _ in range(5):
                        if n_calls >= budget or time.time() > deadline:
                            break
                        x_new = best_x + 0.5 * rng.standard_normal(dim)
                        x_new = np.clip(x_new, -5, 5)
                        f_new = timed_call(x_new)
                    # Reset stagnation
                    stagnation_count = 0
                    # Shrink sigma
                    sigma *= 0.5
                    # Reset paths
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    # Reset mean to best_x
                    m = best_x.copy()
                    # Reset C to identity (simplified restart)
                    C = np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        L = np.eye(dim)
                    continue

            # Update Mean
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update Sigma (CSA)
            # We need C^{-1/2} * (m_new - m) / sigma
            # Using Cholesky: C = L L^T. 
            # C^{-1/2} is not simply L^{-1}. 
            # However, for the path update, we can solve L^T y = (m_new - m)/sigma
            # then L z = y, so z = C^{-1/2} (m_new - m)/sigma
            try:
                diff = (m_new - m) / sigma
                # Solve L^T y = diff
                y = np.linalg.solve(L.T, diff)
                # Solve L z = y
                z = np.linalg.solve(L, y)
                # z is C^{-1/2} diff
            except:
                z = diff # Fallback

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * z
            
            # Adapt sigma
            norm_ps = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / chi_lambda * (norm_ps / chi_lambda - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)

            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * z

            # Update C
            # Rank-mu update
            C_rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_diff = (X_sorted[i] - m) / sigma
                C_rank_mu += weights[i] * np.outer(y_diff, y_diff)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 + (1 - c_c) * (2 - c_c)) * c_c * (1 - c_c) * np.eye(dim)) + c_mu * C_rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2

            # Ensure positive definite
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Regularize
                C = C + 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(dim)
                    C = np.eye(dim)

            m = m_new

        return best_x, best_f
