"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based black-box optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        call_count = [0]

        def counting_func(x):
            call_count[0] += 1
            return func(x)

        # CMA-ES Parameters
        # Population size
        lambda_pop = max(4 * dim + 1, 10)
        # Initial mean: center of the box
        mean = np.zeros(dim)
        # Initial sigma: width of box / 4 is a reasonable start
        sigma = 1.5
        
        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (approximation for sampling)
        # For simplicity in this implementation, we use a simplified update
        # or rely on the diagonal if full CMA-ES is too complex for the budget.
        # However, standard CMA-ES is robust. Let's implement a standard CMA-ES step.
        
        # CMA-ES specific parameters
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        # Eigenvalue updates
        # We need to update C. This requires SVD or similar.
        # For dim=5, this is cheap.

        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        best_x = mean.copy()
        best_f = counting_func(best_x)
        
        # Precompute inverse square root of C initially
        # C = I, so C^-0.5 = I
        C_inv_sqrt = np.eye(dim)
        
        # Helper to update C and its inverse square root
        def update_C_and_inv_sqrt(C, C_inv_sqrt, y_w, step_size):
            # Standard CMA-ES update for C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (c_sigma / (1 - c_sigma)) * (1 - c_sigma) * C) + c_mu * sum(w_i * y_i * y_i.T)
            # This is simplified. A more robust approach for small dim is to just recompute
            # the covariance from the selected points if we were doing a simpler ES.
            # But let's stick to the standard formula.
            
            # Note: This implementation is a simplified version. 
            # For high reliability, we might fall back to diagonal adaptation if C becomes ill-conditioned.
            
            # Update C
            # The term (c_sigma / (1 - c_sigma)) * (1 - c_sigma) is just c_sigma
            # Wait, the standard formula is:
            # C <- (1 - c1 - cmu)C + c1 (pc pc^T + (cs/(1-cs))(1-cs) C) + cmu sum wi yi yi^T
            # The term in the middle is c1 * pc pc^T + c_sigma * C? No.
            # Let's use the standard update:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (c_sigma / (1 - c_sigma)) * (1 - c_sigma) * C) 
            # Actually, the bias correction is (c_sigma / (1 - c_sigma)) * (1 - c_sigma) = c_sigma.
            # So the term is c_1 * (pc @ pc.T + c_sigma * C)? No, that's not right either.
            
            # Let's use the explicit formula from Hansen's paper:
            # C <- (1 - c1 - cmu) C + c1 (pc pc^T + (cs / (1 - cs)) (1 - cs) C) + cmu sum wi yi yi^T
            # The middle term simplifies to c1 * pc pc^T + c_sigma * C? 
            # No, (cs / (1 - cs)) * (1 - cs) = cs.
            # So it is c1 * (pc pc^T + cs * C)? No, the C term is separate.
            # Let's look at the standard implementation.
            # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T) + cmu * sum(wi * outer(yi, yi))
            # PLUS the bias correction term which is effectively added to the (1-c1-cmu)C part?
            # Actually, the standard update is:
            # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (cs / (1 - cs)) * (1 - cs) * C) + cmu * ...
            # This is confusing. Let's use a simpler, robust (1+1)-ES with adaptive sigma if CMA-ES is too bug-prone.
            
            # Given the constraints and the need for reliability, I will switch to a **Robust (1+1) Evolution Strategy** 
            # with **Adaptive Step Size** and **Multiple Restarts**. This is a distinct family (ES) 
            # and is very robust for dim=5.
            pass
            
        # Switching to a robust (1+1)-ES with restarts
        # This is simpler and less prone to implementation errors than a full CMA-ES from scratch.
        
        best_f = np.inf
        best_x = None
        
        # We will perform multiple restarts of (1+1)-ES
        # Each restart uses a portion of the budget.
        
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        for r in range(n_restarts):
            # Initialize mean randomly in the box, or near previous best if r>0
            if r == 0:
                m = np.random.uniform(-2, 2, dim)
            else:
                # Perturb the current best
                m = best_x + 0.5 * np.random.standard_normal(dim)
                m = np.clip(m, lower, upper)
                
            sigma = 1.0
            x = m.copy()
            f = counting_func(x)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            # (1+1)-ES loop
            while call_count[0] < budget - 5:
                # Generate offspring
                y = np.random.standard_normal(dim)
                x_new = x + sigma * y
                x_new = np.clip(x_new, lower, upper)
                
                f_new = counting_func(x_new)
                
                if f_new <= f:
                    # Success
                    x = x_new
                    f = f_new
                    # Increase sigma slightly? No, keep or decrease. 
                    # Standard (1+1)-ES: sigma <- sigma * exp(c_sigma / lambda * (||y||/sigma - chi_N))
                    # For simplicity, we adapt sigma based on success rate.
                    # If success, keep sigma. If fail, decrease sigma.
                    pass
                else:
                    # Failure
                    # Decrease sigma
                    sigma *= 0.9
                    if sigma < 1e-8:
                        break
                        
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    
                # If sigma is too small, break and restart
                if sigma < 0.01:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
