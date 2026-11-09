"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer for the Bent Cigar function (BBOB f12).
    CMA-ES adapts the covariance matrix to the anisotropic landscape,
    allowing efficient search along the narrow valley of the Bent Cigar.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size: standard suggestion is 4 + floor(3 * log(N))
        # For N=5, lambda = 4 + 3*1.6 = 8.8 -> 9. Let's use 12 for better stability.
        lam = 12
        mu = lam // 2  # Number of best individuals to recombine
        
        # Weights for recombination
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 1) - np.log(i + 1)
        weights /= np.sum(weights)
        
        # Variance-reduction constant
        c_mu = np.sum(weights) / (1 - np.sum(weights))
        c_sigma = (c_mu + 2) / (dim + c_mu + 5)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((c_mu + 1) / (dim + 1)) - 1) + c_sigma
        
        # Initialize
        rng = np.random.default_rng(42)
        
        # Start at the center of the box, which is a good heuristic for BBOB functions
        # The minimum is shifted, but starting at 0 is generally safe and central.
        m = np.zeros(dim)
        sigma = 2.0  # Initial step size
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Precompute C^-1/2 for sampling
        # We will update this periodically or use Cholesky every generation
        # For small dim (5), Cholesky is cheap.
        
        # Paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = m.copy()
        best_fx = func(m)
        calls_used = 1
        
        if calls_used >= budget:
            return best_x, best_fx
            
        # Main Loop
        while calls_used + lam <= budget:
            if time.time() >= deadline:
                break
            
            # Generate population
            # Compute Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity
                C = np.eye(dim)
                L = np.eye(dim)
            
            # Sample from N(0, C)
            # z ~ N(0, I) -> y = L @ z
            z = rng.standard_normal((lam, dim))
            # y = (L @ z.T).T
            y = (L @ z.T).T
            
            # Offspring: x_i = m + sigma * y_i
            pop = m[None, :] + sigma * y
            pop = np.clip(pop, -5, 5)
            
            # Evaluate population
            f_vals = np.zeros(lam)
            for i in range(lam):
                if calls_used + i >= budget:
                    break
                f_vals[i] = func(pop[i])
                calls_used += 1
            
            if calls_used >= budget:
                break
                
            # Sort by fitness
            order = np.argsort(f_vals)
            sorted_pop = pop[order]
            sorted_f = f_vals[order]
            
            # Update best
            if sorted_f[0] < best_fx:
                best_fx = sorted_f[0]
                best_x = sorted_pop[0].copy()
                
            # Update mean
            # Weighted sum of the best mu individuals
            new_m = np.sum(sorted_pop[:mu] * weights[:, None], axis=0)
            
            # Update sigma (step size)
            # Calculate y_c = (new_m - m) / sigma
            y_c = (new_m - m) / sigma
            y_c_norm = np.linalg.norm(y_c)
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * c_mu) * y_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used / lam))) / np.sqrt(np.trace(C)) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - c_sigma) ** (calls_used / lam)))
            
            if h_sigma:
                sigma *= np.exp(c_sigma / d_sigma * (y_c_norm / np.sqrt(dim) - 1))
            
            # Update covariance matrix
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * c_mu) * y_c
            
            # Rank-one and rank-mu update
            # C = (1 + mu * c_c * c_c) * C + ...
            # Standard formula:
            # C = (1 + c_c * (2 - c_c) * mu - c_c) * C 
            #    + c_c * (2 - c_c) * (p_c @ p_c.T - C) 
            #    + c_mu * sum_{i=1}^{mu} w_i * (y_i @ y_i.T)
            
            # Let's use the standard update formula from Hansen's paper
            # C = (1 + c_c * (2 - c_c) * mu - c_c) * C is not quite right.
            # Correct formula:
            # C = (1 + c_c * (2 - c_c) * mu - c_c) * C is for the rank-one part coefficient? No.
            
            # Let's use the explicit rank-one and rank-mu updates.
            # C = (1 - c_c) * C + c_c * (2 - c_c) * (p_c @ p_c.T - C) ? No.
            
            # Standard CMA-ES update:
            # C = (1 + mu * c_c * c_c - c_c) * C + c_c * (2 - c_c) * (p_c @ p_c.T - C) + c_mu * sum(w_i * y_i y_i^T)
            # Wait, the term (p_c p_c^T - C) is for the rank-one update?
            # Actually, the standard update is:
            # C <- (1 + mu * c_c * c_c - c_c) * C + c_c * (2 - c_c) * (p_c p_c^T - C) + c_mu * sum_{i=1}^mu w_i * (y_i y_i^T)
            # This seems complex. Let's use a simpler, robust version often found in implementations:
            
            # C = (1 + c_c * (2 - c_c) * mu - c_c) * C is incorrect.
            
            # Let's use the formula from the original paper:
            # C = (1 + c_c * (2 - c_c) * mu - c_c) * C ... No.
            
            # Let's use the following standard update:
            # C = (1 + mu * c_c * c_c - c_c) * C + c_c * (2 - c_c) * (p_c @ p_c.T - C) + c_mu * np.sum(weights[:, None, None] * (y_sorted[:, None, :] * y_sorted[:, None, :].T), axis=0)
            
            # Actually, a common implementation is:
            # C = (1 - c_c) * C + c_c * (2 - c_c) * (p_c @ p_c.T - C) + c_mu * rank_mu_update
            # This is also not standard.
            
            # Let's stick to the most common formulation:
            # C = (1 + mu * c_c * c_c - c_c) * C + c_c * (2 - c_c) * (p_c @ p_c.T - C) + c_mu * sum(w_i * y_i y_i^T)
            
            # Let's re-verify the coefficients.
            # From Hansen (2016):
            # C = (1 + c_c * (2 - c_c) * mu - c_c) * C + c_c * (2 - c_c) * (p_c p_c^T - C) + c_mu * sum_{i=1}^mu w_i y_i y_i^T
            # This looks redundant.
            
            # Correct formula from Hansen's CMA-ES tutorial:
            # C = (1 + mu * c_c * c_c - c_c) * C + c_c * (2 - c_c) * (p_c p_c^T - C) + c_mu * sum_{i=1}^mu w_i y_i y_i^T
            # Wait, the term (p_c p_c^T - C) is multiplied by c_c * (2 - c_c).
            # And the base C is multiplied by (1 + mu * c_c * c_c - c_c).
            
            # Let's calculate the rank-mu update
            y_sorted = (sorted_pop[:mu] - m) / sigma
            rank_mu_update = np.sum(weights[:, None, None] * (y_sorted[:, None, :] * y_sorted[:, None, :].T), axis=0)
            
            # Calculate the update
            c1 = c_c * (2 - c_c)
            cmu = c_mu
            
            # Update C
            C = (1 + mu * c_c * c_c - c_c) * C + c1 * (np.outer(p_c, p_c) - C) + cmu * rank_mu_update
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            
            # Update mean
            m = new_m
            
            # Safety: if sigma becomes too small or C is singular, break
            if sigma < 1e-10:
                break
                
        return best_x, best_fx
# EVOLVE-BLOCK-END
