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
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def check_deadline():
            return time.time() > deadline

        # Phase 1: Latin Hypercube Sampling (LHS) for initialization
        # 50 points is a good trade-off between coverage and budget preservation
        n_lhs = 50
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            # Generate stratified samples
            perm = rng.permutation(n_lhs)
            lhs_points[:, i] = (perm + rng.random(n_lhs)) / n_lhs
            # Scale to [-4.5, 4.5] to focus on the likely minimum region
            lhs_points[:, i] = -4.5 + 9.0 * lhs_points[:, i]

        for i in range(n_lhs):
            if check_deadline():
                break
            f_val = func(lhs_points[i])
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = lhs_points[i].copy()

        # Phase 2: CMA-ES
        # Parameters for 5D
        lam = 10
        mu = lam // 2
        
        # Initial sigma: 2.0 is a reasonable step size for 5D box
        sigma = 2.0
        mean = best_x.copy()
        
        # Weights
        weights = np.log(np.arange(mu, 0, -1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # State
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        
        def update_eig():
            nonlocal C, C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        update_eig()
        
        gen = 0
        while calls_made < budget:
            if check_deadline():
                break
            
            # Generate offspring
            z_samples = rng.standard_normal((lam, dim))
            candidates = []
            for j in range(lam):
                y = C_inv_sqrt @ z_samples[j]
                x_j = mean + sigma * y
                x_j = np.clip(x_j, -5, 5)
                candidates.append(x_j)
            
            # Evaluate
            fvals = []
            for x_j in candidates:
                if check_deadline() or calls_made >= budget:
                    fvals.append(best_f)
                    continue
                f_j = func(x_j)
                calls_made += 1
                fvals.append(f_j)
                if f_j < best_f:
                    best_f = f_j
                    best_x = x_j.copy()
            
            if calls_made >= budget:
                break
                
            fvals = np.array(fvals)
            order = np.argsort(fvals)
            x_selected = np.array([candidates[i] for i in order[:mu]])
            
            # Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * x_selected, axis=0)
            
            # Update paths and covariance
            if np.linalg.norm(mean - mean_old) > 1e-10:
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ (mean - mean_old))
                
                # Condition for hsig
                # Using a simplified threshold to avoid complex history tracking
                hsig = np.linalg.norm(p_sigma) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim)
                
                p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) / sigma * (mean - mean_old)
                
                # Rank-1 update
                C = (1 + c1 * (hsig * (2 - cs) - cs)) * C - c1 * np.outer(p_c, p_c)
                
                # Rank-mu update
                for k in range(mu):
                    y_k = C_inv_sqrt @ (x_selected[k] - mean_old)
                    C += cmu * weights[k] * np.outer(y_k, y_k)
                
                update_eig()
                
                # Update sigma
                # Simplified adaptation rule
                sigma = sigma * np.exp(cs / (dim + 3) * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)
            
            gen += 1

        return best_x, best_f
