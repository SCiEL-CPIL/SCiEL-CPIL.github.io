import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 270s of 330s limit
        deadline = start_time + 270.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0
        
        def call_func(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return global_best_f
            val = func(x)
            calls_made += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = x.copy()
            return val

        # CMA-ES Parameters
        # Standard CMA-ES defaults
        pop_size = 4 + int(3 * np.log(dim))  # lambda
        sigma = 1.0  # Initial step size
        c1 = (4 + np.log(dim)) / (dim**1.6 + 4)
        cmu = 2 * (1 - c1) / ((dim + 1.3)**2 - 1)
        cs = (dim + 1.6) / (dim + 5)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        cc = (4 + dim / cs) / (dim + 4)
        
        # Initialize
        x0 = np.zeros(dim)
        f0 = call_func(x0)
        
        # If initial point is bad, do a few random evals to find a better start
        # But for CMA-ES, starting at 0 is usually fine for BBOB f19 which is centered-ish.
        # However, the task mentions hidden shifts. Let's do a tiny random search first to ensure we don't start in a terrible local spot.
        # Actually, CMA-ES is robust to starting point if sigma is large enough.
        # Let's stick to standard CMA-ES from x=0.
        
        x_mean = x0.copy()
        C = np.eye(dim)
        sqrtC = np.eye(dim)
        inv_sqrtC = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-compute constants
        # Cholesky factorization helper
        def update_sqrtC(C):
            nonlocal sqrtC, inv_sqrtC
            # Use scipy if available? No, stick to numpy.
            # Numpy doesn't have a stable in-place cholesky that returns the matrix directly in a way that's easy to invert for the inverse sqrt.
            # We can use eigendecomposition for stability and simplicity in code, though Cholesky is faster.
            # For dim=5, eig is fine.
            try:
                w, v = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                w = np.clip(w, 1e-10, None)
                sqrtC = (v * np.sqrt(w)).dot(v.T)
                inv_sqrtC = (v / np.sqrt(w)).dot(v.T)
            except np.linalg.LinAlgError:
                # Fallback to identity if it fails (shouldn't happen often)
                sqrtC = np.eye(dim)
                inv_sqrtC = np.eye(dim)

        update_sqrtC(C)
        
        while calls_made < budget and time.time() < deadline:
            # Generate population
            # y_i ~ N(0, I)
            Y = np.random.randn(pop_size, dim)
            # x_i = x_mean + sigma * (sqrtC * y_i)
            X = x_mean[None, :] + sigma * (Y.dot(sqrtC.T))
            
            # Evaluate
            F = np.array([call_func(x) for x in X])
            
            # Check for budget overrun during evaluation (call_func handles it by returning best, but we stop here)
            if calls_made >= budget:
                break
                
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update step size (sigma)
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff / lambda) * inv_sqrtC * (x_best - x_old_mean) / sigma
            # mu_eff = sum(mu_i)^2 / sum(mu_i^2)
            # For simplicity, use mu_eff = pop_size / 2
            mu_eff = pop_size / 2.0
            weight = 1.0 / np.sqrt(mu_eff)
            
            # Vector of best point
            x_best = X_sorted[0]
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff / pop_size) * (inv_sqrtC.dot((x_best - x_mean) / sigma))
            
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1 + 2 / (dim + 1)) * np.sqrt(dim) else 0
            
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            # Update mean
            # x_mean_new = sum(mu_i * x_(i))
            # mu_i = 1/2 for i < pop_size/2 ? No, standard weights.
            # w_i = ln(pop_size/2 + 0.5) - ln(i + 1) for i < pop_size/2
            # Normalize
            mu = pop_size // 2
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(pop_size / 2.0 + 0.5) - np.log(i + 1)
            weights /= np.sum(weights)
            
            x_mean_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - hsig) * cs * (2 - cs) * ps * ps.T) + cmu * sum(weights_i * (x_i - x_mean_new) * (x_i - x_mean_new).T)
            # pc = (1 - cc) * pc + cc * (x_best - x_mean) / sigma
            
            pc = (1 - cc) * pc + cc * (x_best - x_mean) / sigma
            
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            
            # Rank-one update
            rank1 = (1 - c1 - cmu) * C + c1 * (pc[:, None] * pc[None, :] + delta_hsig * ps[:, None] * ps[None, :])
            
            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = X_sorted[i] - x_mean_new
                rank_mu += weights[i] * np.outer(diff, diff)
            rank_mu *= cmu
            
            C = rank1 + rank_mu
            
            # Ensure positive semi-definite
            # Regularize if necessary
            C = (C + C.T) / 2.0
            # Add small identity if eigenvalues become too small
            # Simple check: trace
            if np.trace(C) < 1e-10:
                 C += 1e-10 * np.eye(dim)
                 
            x_mean = x_mean_new
            update_sqrtC(C)
            
            # Convergence check
            # If sigma is very small or spread is small
            if sigma < 1e-8:
                break
                
        return global_best_x, global_best_f
