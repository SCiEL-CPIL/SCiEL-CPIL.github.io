"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Multi-Objective CMA-ES Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Population size for CMA-ES
        # Standard is 4 + floor(3 ln n)
        N = max(10, int(4 + 3 * np.log(dim)))
        if budget < N:
             # If budget is too small, just random sample
             X = lo + (hi - lo) * np.random.rand(budget, dim)
             for x in X:
                 problem(x)
             return X

        # Initialize CMA-ES parameters
        mean = (lo + hi) / 2.0
        sigma = (hi - lo) / 4.0 # Initial step size
        C = np.eye(dim) # Covariance matrix
        B = np.eye(dim) # Eigenvectors
        D = np.ones(dim) # Eigenvalues
        # Precompute weights
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial population
        X = np.zeros((N, dim))
        F = np.zeros((N, self.n_obj))
        used = 0
        
        for i in range(N):
            z = np.random.randn(dim)
            x = mean + sigma * (B @ (D * z))
            x = np.clip(x, lo, hi)
            X[i] = x
            F[i] = problem(x)
            used += 1
            
        # Main loop
        while used < budget:
            # Select parents using non-dominated sorting
            # We need to select 'mu' parents for CMA-ES update
            # For MO, we can use the first front(s)
            ranks, crowding = self._nsga2_sort(F)
            
            # Sort by rank, then crowding
            sort_idx = np.lexsort((-crowding, ranks))
            # Take top 'mu' individuals as parents
            parents_idx = sort_idx[:mu]
            parents_X = X[parents_idx]
            
            # Update CMA-ES parameters
            # New mean
            new_mean = np.sum(weights[:, np.newaxis] * parents_X, axis=0)
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D @ ((new_mean - mean) / sigma)))
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            # Clip sigma to avoid divergence
            sigma = np.clip(sigma, 1e-6, (hi - lo).max() / 2)
            
            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * ((new_mean - mean) / sigma)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (used // N)))) / np.sqrt(mu_eff)
            h = h_sigma < (1.4 + 2 / (dim + 1)) # Threshold
            
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, np.newaxis] * (np.outer(parents_X - mean, parents_X - mean) / (sigma ** 2)), axis=0)
            
            C = (1 + c_1 * h * (1 - h) - c_1 - c_mu) * C + c_1 * (rank_one + (h - 1) * C) + c_mu * rank_mu
            
            # Re-orthogonalize C (ensure symmetry and positive definite)
            C = (C + C.T) / 2
            # Eigen decomposition
            D2, B = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            D2 = np.maximum(D2, 1e-10)
            D = np.sqrt(D2)
            
            mean = new_mean
            
            # Generate new population
            X_new = np.zeros((N, dim))
            F_new = np.zeros((N, self.n_obj))
            
            # Check if we have enough budget for a full generation
            if used + N > budget:
                # Generate only remaining individuals
                rem = budget - used
                for i in range(rem):
                    z = np.random.randn(dim)
                    x = mean + sigma * (B @ (D * z))
                    x = np.clip(x, lo, hi)
                    X_new[i] = x
                    F_new[i] = problem(x)
                # We don't replace X and F with X_new directly if we stop, 
                # but we need to return the best points. 
                # The contract says "Every returned objective vector is recorded".
                # So we just evaluate and return.
                X = X_new[:rem]
                F = F_new[:rem]
                used = budget
                break
            else:
                for i in range(N):
                    z = np.random.randn(dim)
                    x = mean + sigma * (B @ (D * z))
                    x = np.clip(x, lo, hi)
                    X_new[i] = x
                    F_new[i] = problem(x)
                X = X_new
                F = F_new
                used += N
                
        return X

    def _nsga2_sort(self, F):
        N = len(F)
        n_obj = self.n_obj
        ranks = np.zeros(N, dtype=int)
        crowding_dist = np.zeros(N)
        
        S = [[] for _ in range(N)]
        n = np.zeros(N, dtype=int)
        
        for i in range(N):
            for j in range(i + 1, N):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n[i] += 1
        
        front = []
        for i in range(N):
            if n[i] == 0:
                ranks[i] = 0
                front.append(i)
        
        k = 0
        while len(front) > 0:
            next_front = []
            for i in front:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            front = next_front
            k += 1
        
        for r in range(k + 1):
            front_indices = [i for i in range(N) if ranks[i] == r]
            if len(front_indices) <= 2:
                for i in front_indices:
                    crowding_dist[i] = np.inf
                continue
            
            for m in range(n_obj):
                sorted_front = sorted(front_indices, key=lambda x: F[x][m])
                crowding_dist[sorted_front[0]] = np.inf
                crowding_dist[sorted_front[-1]] = np.inf
                
                f_min = F[sorted_front[0]][m]
                f_max = F[sorted_front[-1]][m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                    
                for i in range(1, len(sorted_front) - 1):
                    idx = sorted_front[i]
                    idx_prev = sorted_front[i - 1]
                    idx_next = sorted_front[i + 1]
                    crowding_dist[idx] += (F[idx_next][m] - F[idx_prev][m]) / f_range

        return ranks, crowding_dist

# EVOLVE-BLOCK-END
