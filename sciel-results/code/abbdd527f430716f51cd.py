import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters for N=5
        # lambda = 4 + floor(3 * log(n)) = 4 + 4 = 8
        lam = 8
        mu = lam // 2  # 4
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Initial state
        # Start at 0? The problem says minimizer is in [-4, 4]^5.
        # Random start might be better to avoid symmetry issues if any, but 0 is standard.
        # Let's try a random start within [-2, 2] to be safe? No, 0 is usually fine for BBOB.
        # However, "Attractive Sector" has a global minimum that is not at 0.
        # Actually, the shift is hidden. So starting at 0 is fine.
        
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hs = (1.4 + 2.0 / (n + 1)) * np.sqrt((mu_eff - 1) + 1)
        
        best_x = m.copy()
        best_f = func(best_x)
        calls = 1
        
        # Precompute chs
        chs = 1 + 2 * max(0, (mu_eff - 1) / (n + 1))
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if budget - calls < lam:
                break

            # Generate lambda samples
            Z = np.random.default_rng().standard_normal((lam, n))
            X = np.empty((lam, n))
            for i in range(lam):
                y_i = B @ (D * Z[i])
                X[i] = m + sigma * y_i
            
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                F[i] = func(X[i])
                calls += 1
            
            # Sort
            indices = np.argsort(F)
            F = F[indices]
            X = X[indices]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Recombination
            m_old = m.copy()
            m = np.sum(w[:, None] * X[:mu], axis=0)
            
            # Step size adaptation (CSA)
            y = (m - m_old) / sigma
            z = B.T @ (y / D)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            
            hs_norm = np.linalg.norm(p_sigma)
            hs_threshold = (1.4 + 2.0 / (n + 1)) * (1 - c_sigma)
            if hs_norm / hs < hs_threshold:
                sigma *= np.exp(c_sigma / chs * (hs_norm / hs - 1))
            
            # Covariance matrix adaptation
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * z
            
            # h_c calculation
            # h_c = 1 if ||p_c|| / sqrt(1 - (1-c_c)^(2 * gen)) < (1.4 + 2/(n+1)) * sqrt((mu_eff - 1) + 1)
            # We need to track the generation or the decay factor.
            # Let's approximate h_c = 1 for simplicity, or implement the decay.
            # For robustness, let's implement the decay factor.
            # decay_c = (1 - c_c)**2
            # We need to store the current decay factor.
            
            # Rank-1 update
            # C = (1 - c_1) * C + c_1 * (np.outer(y, y) + (1 - h_c) * C)
            # If h_c = 1, C = (1 - c_1) * C + c_1 * np.outer(y, y)
            
            # Let's use h_c = 1 for now to keep it simple, as the previous attempt might have failed due to complexity.
            h_c = 1.0
            
            C = (1 - c_1) * C + c_1 * (np.outer(y, y) + (1 - h_c) * C)
            
            # Rank-mu update
            for i in range(mu):
                y_i = (X[i] - m_old) / sigma
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Eigen decomposition
            try:
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.clip(D, 1e-12, None))
            except np.linalg.LinAlgError:
                # If eigen decomposition fails, restart or break
                break
            
            # Check for convergence
            if np.max(D) / np.min(D) > 1e14:
                break
            if sigma < 1e-15:
                break

        # Polish with (1+1)-ES if budget remains
        if calls < budget and time.time() < deadline:
            sigma_polish = 0.01
            x_polish = best_x.copy()
            fx_polish = best_f
            rng = np.random.default_rng(1)
            while calls < budget and time.time() < deadline:
                y = np.clip(x_polish + sigma_polish * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx_polish:
                    x_polish, fx_polish = y, fy
                    sigma_polish *= 1.2
                    if fx_polish < best_f:
                        best_f = fx_polish
                        best_x = x_polish.copy()
                else:
                    sigma_polish *= 0.9
                if sigma_polish < 1e-6:
                    break

        return best_x, best_f
