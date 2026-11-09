"""CMA-ES based optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s kill
        
        rng = np.random.default_rng(42)
        d = self.dim
        
        # CMA-ES parameters
        # lambda: population size
        # For d=5, lambda=10 is a good balance between exploration and convergence speed
        lam = max(4 + int(3 * np.log(d)), 10) 
        # mu: number of selected individuals
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initial state
        # Start near center, slightly offset to avoid symmetry issues if any
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        C_inv = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Precompute square root of C for sampling
        # We will update this periodically or use Cholesky
        def update_chol(C):
            # Use Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                # Fallback if matrix becomes non-positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                return L

        L = update_chol(C)
        
        calls_used = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_used += 1
        
        # Track history for step size adaptation
        # chi_n: expected norm of standard normal
        chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d**2))
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            ys = []
            for i in range(lam):
                z = rng.standard_normal(d)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5.0, 5.0)
                ys.append(y)
            
            # 2. Evaluate population
            fs = []
            for y in ys:
                if calls_used >= self.budget:
                    break
                f_y = func(y)
                fs.append(f_y)
                calls_used += 1
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()
            
            # If we ran out of budget during evaluation, break
            if calls_used >= self.budget:
                break
                
            # Ensure we have enough evaluations to proceed with CMA update
            if len(fs) < mu:
                break

            # 3. Sort by fitness
            indices = np.argsort(fs)
            y_selected = np.array([ys[i] for i in indices[:mu]])
            f_selected = np.array([fs[i] for i in indices[:mu]])
            
            # 4. Update mean
            y_new = np.sum(weights[:, None] * y_selected, axis=0)
            
            # 5. Update evolution paths
            # pc = (1 - c_c) * pc + c_c * sqrt(mu_eff) / sigma * (y_new - x_mean)
            # Note: y_new - x_mean is the step in x-space, so divide by sigma to get z-space step
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) / sigma * (y_new - x_mean)
            
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (y_new - x_mean)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (y_new - x_mean)
            
            # 6. Update sigma (step size)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # 7. Update covariance matrix C
            # h_sigma = (||ps|| / sqrt(1 - (1 - c_sigma)^{2*(gen+1)})) / chi_n < 1.4 + 2/(d+1)
            # We don't track generations explicitly here, so we approximate h_sigma
            # A simpler heuristic for h_sigma is to check if ps is small relative to chi_n
            # Standard CMA-ES uses a history of ps, but for robustness we use a simplified version
            # or just set h_sigma = 1 if ps is not too large.
            # For d=5, 1.4 + 2/6 = 1.73.
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1.0 if norm_ps < 1.73 * chi_n else 0.0
            
            # Outer product term
            outer = np.outer(y_new - x_mean, y_new - x_mean)
            
            # Delta h term
            if h_sigma > 0:
                # c_c(2 - c_c) pc pc^T - c_c pc pc^T = c_c(1-c_c) pc pc^T
                # Wait, the formula is:
                # C <- (1 - c_1 - c_mu + h_sigma * c_c * (2 - c_c) - h_sigma * c_c) * C
                # Actually, the standard update is:
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc pc^T + (1 - h_sigma) * c_c * (2 - c_c) * C) 
                #     + c_mu * sum(w_i (y_i - x_mean)(y_i - x_mean)^T)
                
                # Let's stick to the standard formula from Hansen's paper:
                # C <- (1 - c_1 - c_mu + h_sigma * c_c * (2 - c_c) - h_sigma * c_c) * C 
                #      + c_1 * (pc pc^T + (1 - h_sigma) * c_c * (2 - c_c) * C)
                #      + c_mu * sum(...)
                
                # This simplifies to:
                # C <- (1 - c_1 - c_mu) * C 
                #      + c_1 * (pc pc^T + (1 - h_sigma) * c_c * (2 - c_c) * C)
                #      + c_mu * sum(...)
                
                # Let's use the more stable form:
                # C = (1 - c_1 - c_mu) * C 
                #   + c_1 * (pc @ pc.T + (1 - h_sigma) * c_c * (2 - c_c) * C)
                #   + c_mu * sum(w_i * outer(y_i - x_mean))
                
                term_pc = pc @ pc.T
                term_c = (1 - h_sigma) * c_c * (2 - c_c) * C
                term_mu = np.sum(weights[:, None, None] * np.einsum('i,ij,ik->ijk', 
                                y_selected - x_mean, y_selected - x_mean, y_selected - x_mean), axis=0)
                # Actually, simpler: sum w_i (y_i - x_mean) (y_i - x_mean)^T
                term_mu = np.zeros((d, d))
                for i in range(mu):
                    diff = y_selected[i] - x_mean
                    term_mu += weights[i] * np.outer(diff, diff)
                
                C = (1 - c_1 - c_mu) * C + c_1 * (term_pc + term_c) + c_mu * term_mu
                
            else:
                # h_sigma = 0
                # C <- (1 - c_1 - c_mu) * C + c_1 * pc pc^T + c_mu * sum(...)
                term_pc = pc @ pc.T
                term_mu = np.zeros((d, d))
                for i in range(mu):
                    diff = y_selected[i] - x_mean
                    term_mu += weights[i] * np.outer(diff, diff)
                
                C = (1 - c_1 - c_mu) * C + c_1 * term_pc + c_mu * term_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = y_new
            
            # Update Cholesky factorization
            L = update_chol(C)
            
            # Optional: If sigma becomes very small, we might be converged
            # But we keep going to use the budget for AOCC

        return best_x, best_f
# EVOLVE-BLOCK-END
