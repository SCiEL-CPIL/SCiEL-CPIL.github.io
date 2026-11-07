"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for BBOB f23 (Katsuura)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Start near center with a reasonable initial sigma
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # standard formula, for dim=5: 4 + 3*1.609 = 8.8 -> 8
        # Actually for dim=5, let's use lambda_=10 or so. Let's stick to standard: 4+3*ln(5) ≈ 8.8, round to 9? 
        # Let's use a fixed population size for simplicity and robustness in low dim.
        lambda_ = 10
        
        # Learning rates
        cc = (4 + dim / lambda_) / (dim + 4 + 2 * dim / lambda_)
        cmu = min(1 - cc, 2 * (dim**0.5 + 1) / (dim + 2 + lambda_))
        
        # Step size
        chi_lambda = np.sqrt(lambda_) * (1 - 1 / (4 * dim) - 1 / (16 * lambda_))
        # Initial sigma
        sigma = 2.0  # Start with a moderate step size
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        # Precompute inverse square root of C for efficiency
        # We will update C and recompute inverse sqrt periodically or use a simpler update
        
        # Counters
        calls_made = 1
        t = 0
        
        # We need to be careful with the budget.
        # Total budget is 500.
        # Each generation uses lambda_ evaluations.
        # Plus 1 initial evaluation.
        
        # To handle the budget precisely, we can do full generations until we run out of budget,
        # then do single evaluations if needed, or just stop.
        # Since the scoring depends on the sequence of best values, doing full generations is better.
        
        # Let's implement a simplified CMA-ES that recomputes the inverse square root of C each generation.
        # For dim=5, this is cheap.
        
        while calls_made + lambda_ <= budget:
            # Sample lambda_ points
            points = []
            for i in range(lambda_):
                # Standard normal
                z = np.random.standard_normal(dim)
                # Transform using C^(1/2)
                # Compute inverse sqrt of C
                # eigvals, eigvecs = np.linalg.eigh(C)
                # C_sqrt_inv = eigvecs @ np.diag(1/np.sqrt(eigvals))
                # This is O(dim^3). For dim=5, it's fine.
                
                # Alternatively, use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                    # z' = L^{-1} z
                    # Solve L w = z
                    w = np.linalg.solve(L, z)
                    # y = x + sigma * C^(1/2) w ? No.
                    # The standard CMA-ES update is:
                    # y_i = m + sigma * B * diag(d) * z_i
                    # where B diag(d) B^T = C
                    # So we need the square root of C, not inverse.
                    # Let's stick to the eigendecomposition for clarity and correctness in low dim.
                    pass
                except np.linalg.LinAlgError:
                    # Fallback
                    pass
                
                # Let's use eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                points.append(y)
            
            # Evaluate points
            fvals = []
            for y in points:
                fy = func(y)
                fvals.append(fy)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            fvals = np.array(fvals)
            
            # Sort by fitness
            idx = np.argsort(fvals)
            points = [points[i] for i in idx]
            fvals = fvals[idx]
            
            # Update mean
            # Weights
            weights = np.log(lambda_ / 2 + 1) - np.log(np.arange(1, lambda_ + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights[:lambda_//2 + 1]**2)  # Effective population size
            
            new_x = np.sum([weights[i] * points[i] for i in range(lambda_)], axis=0)
            
            # Update covariance matrix
            # C = (1-cmu-cc)C + cmu * sum w_i (y_i - x)(y_i - x)^T + cc * (1/chi^2) p_sigma p_sigma^T ?
            # Actually, the standard update is:
            # C = (1 - cmu - cc) C + cmu * sum_{i=1}^{mu} w_i (y_{i:lambda} - x) (y_{i:lambda} - x)^T + cc ( (1/chi_lambda)^2 - 1 + 2 cc ) p_sigma p_sigma^T ?
            # Wait, the standard formula involves the evolution path p_sigma.
            
            # Let's use a simpler, robust update:
            # 1. Compute the weighted sum of outer products of the top mu points relative to the old mean x.
            # 2. Update C.
            
            mu = lambda_ // 2 + 1
            delta_h = (1.7 + 2 / (dim + 1)) * (cc + 1)
            
            # Update evolution path p_sigma
            # p_sigma = (1 - cs) p_sigma + h_sigma * cs / chi_lambda * C^{-1/2} (new_x - x) / sigma
            # We need C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals))
            
            diff = new_x - x
            norm_diff = np.linalg.norm(diff)
            h_sigma = 1 if norm_diff / np.sqrt(1 - (1 + t / self.t_sigma) ** -1) < (1.4 + 2 / (dim + 1)) * chi_lambda else 0
            # Simplified h_sigma check
            h_sigma = 1.0  # Often set to 1 for simplicity in small dims
            
            p_sigma = (1 - cs) * p_sigma + h_sigma * cs / chi_lambda * (C_inv_sqrt @ diff) / sigma
            # Note: cs is not defined above, let's define it
            cs = (2 + mu_eff / dim) / (dim + 2 + mu_eff / dim)
            
            # Update evolution path p_c
            # p_c = (1 - cc) p_c + h_c * cc * (new_x - x) / sigma
            # h_c = 1 if ||p_sigma||^2 / (1 - (1 - cs)^{2(t+1)}) < (2 + 2/mu_eff)(1 - cc^2) else 0
            h_c = 1.0  # Simplified
            
            p_c = (1 - cc) * p_c + h_c * cc * diff / sigma
            
            # Update C
            # C = (1 - cmu - cc) C + cmu * sum w_i (y_i - x)(y_i - x)^T + cc ( (1/chi_lambda)^2 - 1 + 2 cc ) p_sigma p_sigma^T ?
            # Actually, the rank-mu update uses the top mu points.
            # The rank-one update uses p_sigma.
            
            # Let's compute the rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = points[i] - x
                rank_mu += weights[i] * np.outer(diff_i, diff_i)
            
            # Rank-one update
            rank_one = np.outer(p_sigma, p_sigma)
            
            # Update C
            C = (1 - cmu - cc) * C + cmu * rank_mu + cc * (1 / chi_lambda**2 - 1 + 2 * cc) * rank_one
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update x and sigma
            x = new_x
            sigma *= np.exp(cs / chi_lambda * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            
            # Clip x to box (though points are clipped, x might drift)
            x = np.clip(x, -5, 5)
            
            t += 1
        
        # Use remaining budget for single evaluations if needed (optional, but good for score)
        while calls_made < budget:
            # Use a simple local search or just random steps around best_x
            # Let's do a small random step
            step = 0.1 * np.random.standard_normal(dim)
            y = np.clip(best_x + step, -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
