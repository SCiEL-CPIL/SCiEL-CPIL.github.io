"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES optimizer for BBOB f19.
    Strategy:
    1. Initialize CMA-ES with a mean at the center and a sigma covering the box.
    2. Adapt mean, covariance, and step size to navigate the rotated/shifted landscape.
    3. Handle boundaries via reflection to maintain distribution validity.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = np.zeros(dim)
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
        # Population size: 4 + floor(3 * log(n))
        lambda_ = int(4 + 3 * np.log(dim)) if dim > 1 else 2
        # Selection size: lambda / 2
        mu = max(1, lambda_ // 2)
        
        # Initial Mean: Center of the box
        mean = np.zeros(dim)
        
        # Initial Sigma: Cover the box. Box is [-5, 5], width 10.
        # Standard deviation ~ 1.0 covers ~68% of [-1, 1]. 
        # To cover [-5, 5] reasonably, sigma ~ 2.5 is good (10 sigma range).
        # But for f19 (Griewank-Rosenbrock), the valley is narrow.
        # Let's start with sigma = 1.0 to ensure exploration but not too much noise.
        # Actually, for BBOB, starting sigma is often 2.0.
        sigma = 2.0
        
        # Covariance Matrix: Identity
        C = np.eye(dim)
        
        # Eigen decomposition for sampling and updates
        # C = B * D * B^T
        # D is diagonal matrix of eigenvalues
        # B is matrix of eigenvectors
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Step size adaptation (CSA)
        # For simplicity, we can use a simpler sigma adaptation or fixed.
        # Standard CMA uses CSA. Let's implement a simplified version or standard one.
        # Given the small budget (500), a simplified adaptation might be safer.
        # Let's use the standard CMA-ES update rules.
        
        # Weights
        # mu_i proportional to log(mu+0.5) - log(i)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance-effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        # c_sigma: step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        # c_c: covariance adaptation (CM)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        # c_1: rank-one update
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        # c_mu: rank-mu update
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Damping
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Hessian of chi distribution (for CSA)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Evolution path for step size
        p_sigma = np.zeros(dim)
        # Evolution path for covariance
        p_c = np.zeros(dim)
        
        # Initial evaluation of mean
        f_mean = call_func(mean)
        
        # Main Loop
        # Each generation consumes lambda_ calls.
        # We have 500 calls. ~ 500/lambda_ generations.
        # lambda_ for dim=5 is 4 + 3*log(5) ~ 4 + 4.8 ~ 9.
        # ~ 55 generations.
        
        while calls_made < budget and time.time() < deadline:
            # Sample lambda_ points
            # y = B * D * z, where z ~ N(0, I)
            # x = mean + sigma * y
            
            # Update eigen decomposition if C changed (it does every step)
            # For efficiency, we can update eigvals/eigvecs incrementally, but for dim=5,
            # np.linalg.eigh is fast enough.
            eigvals, eigvecs = np.linalg.eigh(C)
            
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            
            # Sample
            z = np.random.randn(lambda_, dim)
            # Transform z to y
            # y_i = sigma * (eigvecs * (sqrt(eigvals) * z_i))
            # y = z @ (eigvecs * sqrt(eigvals))
            # Note: eigvecs are columns.
            # y = z @ (eigvecs * np.sqrt(eigvals))
            y = z @ (eigvecs * np.sqrt(eigvals))
            
            # x = mean + y
            x_pop = mean[None, :] + y
            
            # Boundary Handling: Reflection
            # If x_i < -5, x_i = -5 + (-5 - x_i) * alpha? 
            # Simple reflection: x_i = -5 - (x_i - (-5)) = -10 - x_i ? No.
            # If x < lb, x = 2*lb - x. If x > ub, x = 2*ub - x.
            lb = -5.0
            ub = 5.0
            
            # Vectorized reflection
            # For each dimension, check bounds.
            # This is a bit tricky to vectorize cleanly for all points/dims simultaneously
            # without loops, but dim is small.
            for i in range(dim):
                # Points below lb
                mask_low = x_pop[:, i] < lb
                if np.any(mask_low):
                    x_pop[mask_low, i] = 2 * lb - x_pop[mask_low, i]
                    # Ensure it's within bounds after reflection (should be if step was reasonable)
                    # If reflection goes out of bounds again (large step), clip?
                    # Reflection usually keeps it in bounds if step < box width.
                    # Here sigma=2, box=10. Max step ~ 4-5 sigma = 10. Might overshoot.
                    # If overshoot, clip to boundary?
                    x_pop[mask_low, i] = np.clip(x_pop[mask_low, i], lb, ub)
                    
                # Points above ub
                mask_high = x_pop[:, i] > ub
                if np.any(mask_high):
                    x_pop[mask_high, i] = 2 * ub - x_pop[mask_high, i]
                    x_pop[mask_high, i] = np.clip(x_pop[mask_high, i], lb, ub)

            # Evaluate
            f_pop = np.array([call_func(x_pop[i]) for i in range(lambda_)])
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            
            # Update Mean
            # mean = sum(weights_i * x_sorted[i])
            mean_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update Step Size (CSA)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B * D^-1 * (mean_new - mean) / sigma
            # D^-1 = 1 / sqrt(eigvals)
            # B * D^-1 * v = eigvecs @ (v / sqrt(eigvals))
            
            # Calculate normalized step
            # v = (mean_new - mean) / sigma
            v = (mean_new - mean) / sigma
            
            # Transform v to standard normal space
            # z_sigma = B^T * D^-1 * v = (eigvecs.T @ v) / sqrt(eigvals)
            z_sigma = (eigvecs.T @ v) / np.sqrt(eigvals)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_sigma
            
            # Adapt sigma
            # sigma = sigma * exp( c_sigma / d_sigma * (||p_sigma|| - chi_n) / chi_n )
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma - chi_n) / chi_n)
            
            # Update Covariance
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (mean_new - mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean_new - mean) / sigma
            
            # Rank-one update term
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2(g+1)}) < (1.4 + 2/(dim+1)) * chi_n
            # Approximate h_sigma. Since we don't track generation count g exactly for the decay,
            # we can use a heuristic or just check norm.
            # Let's use a simple check: if norm_p_sigma is not too large, h_sigma = 1.
            threshold = (1.4 + 2 / (dim + 1)) * chi_n
            h_sigma = 1.0 if norm_p_sigma < threshold * 1.5 else 0.0 # Heuristic
            
            # Rank-one term
            # delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            delta_h = (h_sigma - 1) * c_c * (2 - c_c)
            rank_one = (1 + delta_h) * np.outer(p_c, p_c) - np.eye(dim)
            
            # Rank-mu term
            # y_i = (x_sorted[i] - mean) / sigma
            # C_mu = c_mu * sum(weights_i * outer(y_i, y_i))
            y_mu = (x_sorted[:mu] - mean) / sigma
            # Weighted outer products
            # shape: (mu, dim, dim)
            rank_mu = c_mu * np.sum(weights[:, None, None] * y_mu[:, :, None] * y_mu[:, None, :], axis=0)
            
            # Update C
            C = (1 + c_1 * rank_one + c_mu * rank_mu - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) * 0 # Wait, formula is:
            # C = (1 - c_1 - c_mu) * C + c_1 * (1 + delta_h) * outer(p_c, p_c) + c_mu * sum(...)
            # My previous line was slightly off. Let's use the correct one.
            C = (1 - c_1 - c_mu) * C + c_1 * (1 + delta_h) * np.outer(p_c, p_c) + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Ensure positive definiteness (add small identity if needed)
            # Check eigenvalues
            eigvals_C = np.linalg.eigvalsh(C)
            if np.min(eigvals_C) < 1e-12:
                C += (1e-12 - np.min(eigvals_C)) * np.eye(dim)
                
            # Update mean
            mean = mean_new
            
            # Update global best if mean is better?
            # We already update global_best in call_func.
            
            # Check time
            if time.time() > deadline:
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
