import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != dim:
                x = np.resize(x, dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                val = func(x)
                if not np.isfinite(val):
                    return np.inf
                return float(val)
            except Exception:
                return np.inf

        def time_left():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Integer-grid initialization to exploit Rastrigin structure
        # Rastrigin has minima at integer points, so we start there
        # But the function is shifted and rotated, so we try a grid of candidate points
        # We'll use a coarse grid first, then refine with CMA-ES

        # Try integer points and their neighbors
        candidates = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    for l in range(-2, 3):
                        for m in range(-2, 3):
                            candidates.append([i, j, k, l, m])
        
        candidates = np.array(candidates, dtype=float)
        candidates = np.clip(candidates, -5.0, 5.0)
        
        # Evaluate a subset of candidates (we have limited budget)
        # Prioritize points closer to origin
        dists = np.linalg.norm(candidates, axis=1)
        sorted_idx = np.argsort(dists)
        
        # Evaluate up to 50 closest integer points
        n_eval = min(50, len(candidates), budget // 5)
        for idx in sorted_idx[:n_eval]:
            if not time_left() or call_count >= budget - 10:
                break
            x = candidates[idx]
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES for refinement
        if best_x is not None and time_left() and call_count < budget - 10:
            x = best_x.copy()
            sigma = 1.0
            remaining = budget - 1 - call_count
            
            # Simple CMA-ES implementation
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            C_sqrt = np.eye(dim)
            
            # Update rules
            alpha_mu = 1.0
            c_sigma = 1.0 / (dim + 4)
            c_c = (4 + dim / 4) / (dim + 4 + 3)
            c_1 = 2 / ((dim + 1.3) ** 2 + dim)
            c_mu = min(1 - c_1, 2 * ((mu_eff + 2) * (mu_eff - 2 + 1 / dim)) / ((dim + 2) ** 2 + mu_eff))
            
            mu = max(1, dim // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights ** 2)
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Use a smaller population for CMA-ES
            lambda_ = min(dim + 3, 15)
            
            # Initial population
            X = x + sigma * rng.multivariate_normal(np.zeros(dim), C, lambda_)
            X = np.clip(X, -5.0, 5.0)
            
            F = np.array([safe_func(xi) for xi in X])
            
            # Sort
            idx_sorted = np.argsort(F)
            X = X[idx_sorted]
            F = F[idx_sorted]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
            
            # CMA-ES iterations
            while time_left() and call_count < budget - 5:
                # Compute mean
                m = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
                
                # Update step size
                sigma *= np.exp(c_sigma / lambda_ * (np.linalg.norm(m - x) / sigma - np.sqrt(2 * mu / (dim + 1)) - 1))
                sigma = np.clip(sigma, 1e-8, 5.0)
                
                # Update evolution paths
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (m - x) / sigma
                h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 2) else 0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma
                
                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_1 - c_mu + delta_h) * C + c_1 * (np.outer(p_c, p_c) - C) + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(X[:mu] - x, X[:mu] - x), axis=0)
                
                # Recompute Cholesky
                try:
                    C_eigvals, C_eigvecs = np.linalg.eigh(C)
                    C_eigvals = np.clip(C_eigvals, 1e-12, None)
                    C_sqrt = C_eigvecs @ np.diag(np.sqrt(C_eigvals)) @ C_eigvecs.T
                    C_inv_sqrt = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
                
                x = m
                
                # Generate new population
                X = x + sigma * (rng.standard_normal((lambda_, dim)) @ C_sqrt.T)
                X = np.clip(X, -5.0, 5.0)
                F = np.array([safe_func(xi) for xi in X])
                
                idx_sorted = np.argsort(F)
                X = X[idx_sorted]
                F = F[idx_sorted]
                
                if F[0] < best_f:
                    best_f = F[0]
                    best_x = X[0].copy()
        
        # Phase 3: Final local refinement with (1+1)-ES
        if best_x is not None and time_left() and call_count < budget - 1:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - 1 - call_count
            
            for _ in range(remaining):
                if not time_left() or call_count >= budget - 1:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = safe_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                
                sigma = np.clip(sigma, 0.001, 2.0)

        return best_x, best_f
