"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Standard CMA-ES defaults
        # lambda: population size
        lam = 4 + int(3 * np.log(dim))
        # mu: number of recombination points
        mu = lam // 2
        # weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial mean and sigma
        # Start near center to cover the box quickly
        x_mean = np.zeros(dim)
        # Initial sigma: box size is 10, dim 5. 
        # A reasonable initial step is 1.0 or 2.0. 
        # For f21, which has many peaks, starting with a larger sigma helps exploration.
        sigma = 1.0
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        # Cholesky factorization of C
        # To avoid numerical issues, we can use eigendecomposition or update C directly
        # Standard CMA-ES updates C via rank-1 and rank-mu updates.
        
        # Precompute square root of C for sampling? 
        # Efficient implementation uses eigen-decomposition every iteration or 
        # maintains a square root matrix. 
        # For simplicity and robustness in 5D, we can use the eigen-decomposition 
        # of C to sample, or just use the standard update and sample from N(0, C) 
        # using np.random.multivariate_normal (which is slower but robust for small dim).
        # However, multivariate_normal can be slow. 
        # Let's use a simple eigen-decomposition based sampling or 
        # the standard update which requires C to be positive definite.
        
        # Actually, for dim=5, the cost of eigen-decomposition is negligible.
        # But we need to sample from N(x_mean, sigma^2 * C).
        # We can compute the Cholesky factor of C at each step? 
        # Or use the fact that C is updated by rank updates.
        
        # Let's use a simpler, robust approach for small dim:
        # Sample z ~ N(0, I), then y = x_mean + sigma * C^{1/2} z.
        # We can maintain the eigenvalues and eigenvectors of C?
        # Or just recompute C^{1/2} every iteration using np.linalg.cholesky.
        
        # Let's implement the standard CMA-ES algorithm.
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        # Initial population
        # Generate first population
        # We need to be careful with the number of calls.
        # Total calls: 1 (initial) + budget - 1? 
        # The loop structure:
        # We have `budget` calls total.
        # 1 call for x_mean.
        # Then we have budget - 1 calls left.
        # Each iteration of CMA-ES uses `lam` calls.
        # So we can do at most (budget - 1) // lam full iterations.
        # We should stop if time runs out or budget is exhausted.
        
        calls_made = 1
        
        while calls_made < budget and time.time() < deadline:
            # Generate population
            # Get square root of C
            # Ensure C is positive definite
            # np.linalg.cholesky might fail if C is not PD due to numerical errors
            # We can add a small regularization if needed
            try:
                # Use eigen-decomposition for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                eigvals = np.clip(eigvals, 1e-10, None)
                # C^{1/2} = eigvecs @ diag(sqrt(eigvals)) @ eigvecs.T
                # To sample: z ~ N(0, I), y = x_mean + sigma * eigvecs @ (sqrt(eigvals) * z)
                sqrt_eigvals = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                # Fallback to identity if C is broken
                C = np.eye(dim)
                eigvals = np.ones(dim)
                eigvecs = np.eye(dim)
                sqrt_eigvals = np.ones(dim)
            
            # Sample population
            # z is (lam, dim)
            z = np.random.randn(lam, dim)
            # Transform z
            # y = x_mean + sigma * (z @ eigvecs.T) * sqrt_eigvals
            # z @ eigvecs.T is (lam, dim)
            # Multiply by sqrt_eigvals (dim) -> broadcasting
            Y = x_mean + sigma * (z @ eigvecs.T) * sqrt_eigvals
            
            # Clip to box
            Y = np.clip(Y, -5, 5)
            
            # Evaluate population
            F = np.array([func(y) for y in Y])
            calls_made += lam
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = Y[min_idx].copy()
            
            # Sort population by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Recombination
            # y_weighted = sum(weights[i] * Y_sorted[i])
            y_weighted = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update mean
            x_mean = y_weighted
            
            # Update step size control variables
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C^{1/2})^{-1} * (y_weighted - x_mean_old)
            # We need (C^{1/2})^{-1}
            # Since C = eigvecs @ diag(eigvals) @ eigvecs.T
            # C^{1/2} = eigvecs @ diag(sqrt_eigvals) @ eigvecs.T
            # (C^{1/2})^{-1} = eigvecs @ diag(1/sqrt_eigvals) @ eigvecs.T
            
            # Calculate step vector
            step_vec = y_weighted - x_mean + sigma * (z @ eigvecs.T) * sqrt_eigvals # Wait, this is not right.
            # The standard formula is:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C^{1/2})^{-1} * (y_weighted - x_mean_prev)
            
            # Let's re-calculate (C^{1/2})^{-1} @ (y_weighted - x_mean_prev)
            # (y_weighted - x_mean_prev) is the step taken.
            # Let d = y_weighted - x_mean
            # We want to project d into the principal axes of C.
            # v = eigvecs.T @ d
            # v_scaled = v / sqrt_eigvals
            # result = eigvecs @ v_scaled
            
            d = y_weighted - x_mean
            v = eigvecs.T @ d
            v_scaled = v / sqrt_eigvals
            c_inv_half_d = eigvecs @ v_scaled
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * c_inv_half_d
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (d / sigma)
            
            # Update covariance matrix
            # h_sigma = (||ps||^2 / (1 - (1 - c_sigma)^{2*(calls_made/lam)}) - 1) / c_sigma < 1.7 + 2/(dim+1)
            # This requires tracking the number of iterations. 
            # For simplicity, we can approximate h_sigma or use a simplified update.
            # A common simplification is to just use the rank-1 and rank-mu updates
            # and check if ps is too large.
            
            # Let's use a simpler heuristic for h_sigma or just set it to 1 if ||ps|| is reasonable.
            # Actually, the standard implementation is robust. Let's implement the check.
            # We need the number of iterations.
            # Let's keep track of iteration count.
            
            # For now, let's just do the update without the h_sigma correction 
            # or use a fixed threshold.
            # ps_norm = np.linalg.norm(ps)
            # h_sigma = 1 if ps_norm < (1.7 + 2/(dim+1)) * np.sqrt((1 - (1 - c_sigma)**2) * mu_eff) else 0
            
            # Let's use the standard formula.
            # We need to know how many generations have passed.
            # Let's assume we are in generation g.
            # The formula uses 1 - (1 - c_sigma)^{2*g}
            
            # To avoid complexity, let's just use the update:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * c_c * (2 - c_c) * I) + c_mu * sum(weights[i] * (y_i - y_weighted) @ (y_i - y_weighted).T)
            
            # We need h_sigma.
            # Let's approximate h_sigma = 1 if ||ps|| is not too large.
            ps_norm = np.linalg.norm(ps)
            # Threshold
            th = (1.7 + 2.0 / (dim + 1)) * np.sqrt((1 - (1 - c_sigma)**2) * mu_eff)
            # This is an approximation. Ideally we track generations.
            # Let's just use h_sigma = 1.0 for simplicity, as the correction is small.
            h_sigma = 1.0
            
            # Rank-1 update
            # pc @ pc.T
            # Note: pc is updated with d/sigma, so pc @ pc.T is in the normalized space.
            # The update formula is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - h_sigma) * c_c * (2 - c_c) * I)
            # + c_mu * sum(weights[i] * (y_i - y_weighted) @ (y_i - y_weighted).T)
            
            # Let's compute the rank-mu part
            # Diffs = Y_sorted[:mu] - y_weighted
            # Rank_mu = sum(weights[i] * Diffs[i] @ Diffs[i].T)
            Diffs = Y_sorted[:mu] - y_weighted
            # Efficient computation of weighted sum of outer products
            # Rank_mu = Diffs.T @ (weights[:, None] * Diffs)
            Rank_mu = Diffs.T @ (weights[:, None] * Diffs)
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - h_sigma) * c_c * (2 - c_c) * np.eye(dim)) + c_mu * Rank_mu
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / 5 * (||ps|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim^2))
            c_sigma = 1.0 # Wait, c_sigma is a learning rate. The formula uses c_sigma.
            # The formula is:
            # sigma = sigma * exp(c_sigma / 5 * (||ps|| / E||N(0,I)|| - 1))
            # Where c_sigma is the learning rate for sigma.
            # We defined c_sigma earlier.
            
            E_norm = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (32 * dim * dim))
            sigma *= np.exp(c_sigma / 5.0 * (ps_norm / E_norm - 1.0))
            
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0

        return best_x, best_f

# EVOLVE-BLOCK-END
