"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For dim=5, lambda=4 or 5 is efficient.
        # mu = lambda / 2
        lam = 4
        mu = lam // 2
        
        # Initial parameters
        x0 = np.zeros(dim)
        sigma = 2.0  # Initial step size, box is [-5, 5]
        
        # Weights for the best mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2/3) - 2) / ((dim + 2)**2 + mueff))
        
        # State variables
        mean = x0.copy()
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Cholesky factorization helper
        def cholesky(C):
            L = np.linalg.cholesky(C)
            return L
            
        L = cholesky(C)
        
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        norm_s = 1.0
        norm_c = 1.0
        norm_c_inv = 1.0
        
        best_x = mean.copy()
        best_f = func(best_x)
        
        calls_made = 1
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Sample candidates
            ys = []
            for i in range(lam):
                z = np.random.randn(dim)
                y = mean + sigma * L @ z
                y = np.clip(y, -5, 5)
                ys.append(y)
                
            # Evaluate candidates
            fs = np.array([func(y) for y in ys])
            calls_made += lam
            
            # Check if we have a better best
            min_idx = np.argmin(fs)
            if fs[min_idx] < best_f:
                best_f = fs[min_idx]
                best_x = ys[min_idx].copy()
                
            # Sort by fitness
            order = np.argsort(fs)
            y_best = [ys[i] for i in order[:mu]]
            
            # Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * np.array(y_best), axis=0)
            
            # Update covariance matrix
            y_diff = np.array(y_best) - mean_old
            delta_mean = mean - mean_old
            
            # Update step size control variables
            # Note: ps and pc update depends on the inverse square root of C
            # Simplified update for robustness
            try:
                C_inv_sqrt = np.linalg.cholesky(C_inv)
            except np.linalg.LinAlgError:
                # Fallback if C_inv is not positive definite
                C_inv_sqrt = np.eye(dim)
                
            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_inv_sqrt @ delta_mean)
            
            # pc update
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * delta_mean
            
            # Heuristic for h_sigma and h_c
            norm_ps = np.linalg.norm(ps)
            norm_pc = np.linalg.norm(pc)
            
            # Expected norms
            norm_s_expected = np.sqrt(dim) * (1 - (1 - c_sigma)**(2/(budget+1)))
            norm_c_expected = np.sqrt(dim) * (1 - (1 - c_c)**(2/(budget+1)))
            
            # This is a simplified heuristic; standard CMA-ES uses more precise checks
            # For robustness in a short budget, we might skip the h_sigma/h_c checks 
            # or use a simple threshold.
            
            # Update C
            C = (1 + c_1 + c_mu - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) - C) + c_mu * np.sum(weights[:, None, None] * np.einsum('i,ji,kj->ijk', weights, y_diff, y_diff), axis=0)
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            try:
                # Add small regularization if needed
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C += 1e-10 * np.eye(dim)
                L = np.linalg.cholesky(C)
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                # If C becomes invalid, reset to identity and small sigma
                C = np.eye(dim)
                C_inv = np.eye(dim)
                L = np.eye(dim)
                sigma = max(sigma * 0.5, 0.1)
                
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / norm_s_expected - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Optional: restart if sigma is too small or too large
            if sigma < 1e-8:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
