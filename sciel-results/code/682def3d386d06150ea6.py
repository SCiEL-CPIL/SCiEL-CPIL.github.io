import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Setup
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # CMA-ES parameters
        lambda_ = 20  # Population size
        mu_ = lambda_ // 2  # Number of parents for recombination
        
        # Weights
        weights = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
        weights /= np.sum(weights)
        mu_eff = np.sum(weights**2) ** -1
        
        # Rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(0.3, (2 * (mu_eff - 2 + 2/mu_eff)) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 2) * (2 - mu_eff)) / (mu_eff + 1) - 1) + c_sigma
        
        # Initial state
        # Start with uniform random points
        X = lo + (hi - lo) * np.random.rand(lambda_, dim)
        F = np.array([problem(x) for x in X])
        
        # Initial mean and covariance
        mean = np.mean(X, axis=0)
        # Scale covariance to the range of the problem
        scale = (hi - lo) / 4.0
        C = np.eye(dim) * np.mean(scale**2)
        # Cholesky factorization for sampling
        try:
            L = np.linalg.cholesky(C)
        except:
            L = np.eye(dim) * np.sqrt(np.mean(scale**2))
            
        sigma = np.mean(hi - lo) / 10.0  # Initial step size
        
        # Path vectors
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track used budget
        used = lambda_
        
        # Time limit
        start_time = time.time()
        time_limit = 280  # Safety margin for 300s limit
        
        # Pareto Front Management
        # We will maintain an archive of non-dominated points
        # For efficiency, we just keep the current population's non-dominated points
        # and select parents from them.
        
        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            # Simple O(N^2) non-dominated sort
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        ranks[j] += 1
            return ranks

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices)) * np.inf
            F_sub = F[indices]
            n = len(indices)
            cd = np.zeros(n)
            for k in range(n_obj):
                min_val = np.min(F_sub[:, k])
                max_val = np.max(F_sub[:, k])
                if max_val == min_val:
                    continue
                # Sort by objective k
                sort_idx = np.argsort(F_sub[:, k])
                cd[sort_idx[0]] = np.inf
                cd[sort_idx[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[sort_idx[i]] += (F_sub[sort_idx[i+1], k] - F_sub[sort_idx[i-1], k]) / (max_val - min_val)
            return cd

        while used < budget:
            if time.time() - start_time > time_limit:
                break
                
            # 1. Sample Offspring
            # Transform standard normal to CMA-ES space
            z = np.random.randn(lambda_, dim)
            # y = z @ L.T (since C = L L.T)
            # Actually, we want x = mean + sigma * L @ z.T
            # But we also need to update C based on y = (x - mean_old) / sigma
            # So x = mean + sigma * (C^{1/2} z)
            # Let's stick to standard CMA-ES: x = mean + sigma * C^{1/2} z
            # C^{1/2} is L (lower triangular from cholesky)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C)
            except:
                # Fallback if numerical issues
                C = np.eye(dim) * np.trace(C) / dim
                L = np.eye(dim) * np.sqrt(np.trace(C) / dim)
                
            # Sample
            Z = np.random.randn(lambda_, dim)
            # x = mean + sigma * (L @ Z.T).T
            offspring = mean + sigma * (Z @ L.T)
            
            # Clip to bounds
            offspring = np.clip(offspring, lo, hi)
            
            # Evaluate
            F_off = np.array([problem(x) for x in offspring])
            used += lambda_
            
            # 2. Combine and Select
            # Combine current population X and offspring
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])
            
            # Non-dominated sort
            ranks = non_dominated_sort(F_all)
            
            # Select best lambda_ points
            # Priority: Rank, then Crowding Distance
            # Get unique ranks
            unique_ranks = np.unique(ranks)
            selected_indices = []
            for r in unique_ranks:
                if len(selected_indices) >= lambda_:
                    break
                idx_in_rank = np.where(ranks == r)[0]
                # Calculate crowding distance for this rank
                cd = crowding_distance(F_all, idx_in_rank)
                # Sort by cd descending
                sorted_cd_idx = np.argsort(-cd)
                needed = min(lambda_ - len(selected_indices), len(idx_in_rank))
                for i in range(needed):
                    selected_indices.append(idx_in_rank[sorted_cd_idx[i]])
            
            # If we didn't select enough (should not happen), fill with random
            while len(selected_indices) < lambda_:
                selected_indices.append(np.random.randint(len(F_all)))
                
            X = X_all[selected_indices]
            F = F_all[selected_indices]
            
            # 3. Update CMA-ES Parameters
            # Mean
            mean_old = mean.copy()
            # Weighted mean of selected parents
            # We need to map selected parents back to the "y" space for covariance update
            # y_i = (x_i - mean_old) / sigma
            Y = (X - mean_old) / sigma
            
            # Weighted sum of Y
            # weights are defined for the top mu_ parents
            # We need to sort X by fitness to get the top mu_
            # For simplicity, we use the weights on the first mu_ selected points
            # But standard CMA-ES uses the best mu_ points.
            # Let's identify the best mu_ points from X
            # Sort F by sum of objectives (simple proxy) or just take the first mu_ from non-dominated sort?
            # Standard CMA-ES uses the selection mechanism to pick the parents.
            # Here, our selection mechanism picked the best lambda_ points.
            # We assume the first mu_ of these are the best parents for recombination.
            # This is an approximation.
            
            Y_parents = Y[:mu_]
            w = weights
            
            # p_sigma update
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^-1/2 * (mean_new - mean_old) / sigma
            # C^-1/2 is inv(L)
            try:
                L_inv = np.linalg.inv(L)
            except:
                L_inv = np.eye(dim)
                
            mean_new = np.sum(w[:, np.newaxis] * Y_parents, axis=0) * sigma + mean_old
            # Wait, mean update is:
            # mean_new = mean_old + sigma * sum(w_i * y_i)
            # So mean_new = mean_old + sigma * (w @ Y_parents.T).T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L_inv @ ((mean_new - mean_old) / sigma))
            
            # sigma update
            norm_p_sigma = np.linalg.norm(p_sigma)
            expected_norm = np.sqrt(dim) * (1 - 1/(4*dim) - 1/(8*dim**2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / expected_norm - 1))
            
            # p_c update
            # p_c = (1 - c_1) * p_c + sqrt(c_1 * (2 - c_1) * mu_eff) * C^-1/2 * (mean_new - mean_old) / sigma
            # Note: The formula for p_c in CMA-ES is slightly different, usually involving the same term but different rate
            # Standard: p_c = (1 - c_1) p_c + sqrt(c_1 (2 - c_1) mu_eff) C^{-1/2} (mean_new - mean_old)/sigma
            # Wait, the term for p_c uses c_1, and for p_sigma uses c_sigma.
            
            p_c = (1 - c_1) * p_c + np.sqrt(c_1 * (2 - c_1) * mu_eff) * (L_inv @ ((mean_new - mean_old) / sigma))
            
            # Covariance update
            # C = (1 - c_1 - c_mu) C + c_1 * p_c * p_c.T + c_mu * sum(w_i * y_i * y_i.T)
            # Note: The sum is over the weighted outer products
            rank1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu_):
                rank_mu += w[i] * np.outer(Y_parents[i], Y_parents[i])
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Add small regularization if needed
            # Check eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += np.eye(dim) * 1e-10
                
            mean = mean_new

        return X
