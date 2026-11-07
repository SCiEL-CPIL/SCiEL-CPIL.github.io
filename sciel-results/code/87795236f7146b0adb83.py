import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For n=5, use lambda=10, mu=5
        lam = 10
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1 / np.sum(w**2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # State
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        calls_made = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Initial evaluation
        f0 = func(x_mean)
        calls_made += 1
        best_f = f0
        best_x = x_mean.copy()
        
        if calls_made >= budget:
            return best_x, best_f
            
        # Pre-compute some constants for speed
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        while calls_made < budget:
            if time.time() >= deadline:
                break
                
            # Check if we have enough budget for a full generation
            if calls_made + lam > budget:
                # Try to do a partial generation if possible, but CMA-ES needs full pop
                # If we can't do a full pop, break.
                # However, we can still evaluate individual points if we want to explore,
                # but without updating CMA-ES.
                # Let's just break to be safe and return best so far.
                # Or, we can do a simple local search on the remaining budget.
                # Given the small budget (500), breaking is fine.
                break
                
            # Generate population
            zs = np.random.normal(size=(lam, n))
            ys = (zs * D[None, :]) @ B.T
            pop = x_mean + sigma * ys
            
            # Clip
            pop = np.clip(pop, -5, 5)
            
            # Evaluate
            fs = np.empty(lam)
            for i in range(lam):
                fs[i] = func(pop[i])
                calls_made += 1
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = pop[i].copy()
                
                if calls_made >= budget:
                    break
            
            if calls_made >= budget:
                break
                
            # If we didn't evaluate all, break
            if not np.all(np.isfinite(fs)):
                break

            # Sort
            order = np.argsort(fs)
            pop_sorted = pop[order]
            fs_sorted = fs[order]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * pop_sorted[:mu], axis=0)
            
            # Update step size
            y = (x_mean - x_mean_old) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
            
            # Standard hsig check
            hsig = (np.linalg.norm(ps) < (1 + 2/(n+1)) * chi_n)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)
            
            # Update covariance
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * y
            
            # Rank-mu term
            y_parents = (pop_sorted[:mu] - x_mean_old) / sigma
            rank_mu_term = np.einsum('i,ij,ik->jk', w, y_parents, y_parents)
            
            # Rank-one term
            rank_one_term = pc @ pc.T + (1 - hsig) * c_c * (2 - c_c) * np.eye(n)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one_term + c_mu * rank_mu_term
            
            # Cholesky decomposition
            C += 1e-10 * np.eye(n)
            try:
                L = np.linalg.cholesky(C)
                B, D = np.linalg.qr(L)
                D = np.diag(D)
                D = np.abs(D)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
                sigma *= 0.5
        
        return best_x, best_f
