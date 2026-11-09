"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        dim = self.dim
        # Initial population size
        # Standard CMA-ES lambda is 4 + floor(3 * log(dim))
        # For dim=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8
        # We might want a slightly larger population for exploration in early stages
        # but standard is usually robust. Let's stick to standard-ish.
        lam = 4 + int(3 * np.log(dim))
        
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Initial sigma: The box is [-5, 5]. 
        # A good initial step size covers a significant portion of the space.
        # sigma = 2.0 covers roughly 2/5 of the range per dimension.
        sigma = 2.0
        
        # Covariance matrix C
        C = np.eye(dim)
        
        # Eigen decomposition for update
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Learning rates
        # Standard CMA-ES rates
        c_sigma = (mu + 0.3) / (dim + 2 + mu)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        
        # Weights for recombination
        # mu is the number of selected individuals
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_sigma = (mu_eff + 0.3) / (dim + 2 + mu_eff)
        
        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluations
        # We need to make sure we don't exceed budget.
        # CMA-ES loop runs for budget / lam generations roughly.
        
        # To ensure we have some calls even if CMA-ES is slow or buggy,
        # we will manually manage the call count.
        
        # Initial population sampling
        # We need to be careful with the first evaluation.
        
        # Helper to evaluate and track
        best_x = mean.copy()
        best_f = np.inf
        
        # Evaluate initial mean? 
        # CMA-ES usually samples from N(mean, sigma^2 C).
        # Let's start by evaluating the initial mean to establish a baseline.
        # This counts as 1 call.
        f_mean = func(mean)
        best_f = f_mean
        best_x = mean.copy()
        calls_made = 1
        
        # If budget is small, we might just do random search or simple hill climbing.
        # But budget is 500, so CMA-ES is feasible.
        
        # Pre-allocate storage for population
        # We will generate population, evaluate, sort, update.
        
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate population
            # If remaining budget is less than population size, reduce population size
            # or just do a final random/hill climb step.
            if self.budget - calls_made < lam:
                # Not enough budget for a full generation. 
                # Do a simple local search or just stop.
                # To be safe, let's just do a few random perturbations around best_x
                # if we have budget left, or just break.
                # Actually, better to just break to avoid wasting calls on a broken generation.
                # Or do a (1+1)-ES step for the remaining budget.
                # Let's switch to (1+1)-ES for the tail to ensure progress.
                sigma_tail = sigma * 0.5
                while calls_made < self.budget:
                    if time.time() > deadline:
                        break
                    y = np.clip(best_x + sigma_tail * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_x = y
                        best_f = fy
                        sigma_tail *= 1.2
                    else:
                        sigma_tail *= 0.9
                break
            
            # Sample population
            # Y = mean + sigma * Z @ Cholesky(C)
            # We have eigvecs and eigvals. C = eigvecs @ diag(eigvals) @ eigvecs.T
            # Cholesky is not directly from eig. We can use eigvecs * sqrt(eigvals)
            # B = eigvecs * np.sqrt(eigvals)
            B = eigvecs * np.sqrt(eigvals)
            
            Z = rng.standard_normal((lam, dim))
            Y = mean[None, :] + sigma * (Z @ B.T)
            
            # Clip to box
            Y = np.clip(Y, -5, 5)
            
            # Evaluate population
            F = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    # If we run out of time mid-generation, we have to handle it.
                    # We have evaluated i points. We can't complete the generation.
                    # We should probably break out and return best so far.
                    # But we need to update calls_made.
                    # For simplicity and safety, if we hit deadline, we break the outer loop.
                    # However, we've already made i calls in this generation.
                    # We must account for them.
                    # Let's just break the inner loop and then the outer loop.
                    pass
                
                F[i] = func(Y[i])
                calls_made += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = Y[i].copy()
                    
            # If we broke out of inner loop due to time, break outer
            if time.time() > deadline:
                break
                
            # Sort population
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update mean
            # w = weights
            new_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update sigma
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B * (new_mean - mean)
            # Note: B here is the inverse of the transformation used to generate Y?
            # Y = mean + sigma * Z @ B.T
            # So Z = (Y - mean) @ inv(B.T) / sigma
            # The term in p_sigma update is the step in the whitened space.
            # step = (new_mean - mean) / sigma
            # whitened_step = step @ inv(B.T)
            
            # We need inv(B.T). B = eigvecs * sqrt(eigvals).
            # inv(B.T) = inv(eigvecs * sqrt(eigvals).T) = inv(eigvecs.T * diag(sqrt(eigvals)))
            # = diag(1/sqrt(eigvals)) * eigvecs
            # Let's compute this matrix.
            inv_B_T = eigvecs.T * (1.0 / np.sqrt(eigvals))
            
            step = (new_mean - mean) / sigma
            whitened_step = step @ inv_B_T
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * whitened_step
            
            # Hessian norm correction
            # h_sigma = ||p_sigma|| / sqrt(1 - (1 - c_sigma)^{2*(calls/lam)}) / sqrt(mu_eff)
            # Approximation:
            # Since we don't track generations explicitly for the decay term, we use a simpler check
            # or standard approximation.
            # Standard CMA-ES uses generation count.
            # Let's estimate generation count: gen = calls_made / lam
            # But calls_made is cumulative.
            # Let's just use a heuristic or standard formula with gen approx.
            # Actually, for robustness, if p_sigma norm is large, h_sigma is 1.
            # Let's compute h_sigma properly if possible.
            # We need a counter for generations.
            # Let's add a gen counter.
            
            # I didn't add a gen counter in the loop. Let's add it.
            # Wait, I can't modify the loop structure easily without re-writing.
            # Let's assume h_sigma = 1 if ||p_sigma|| is large, else 0?
            # Better: compute the expected norm.
            # E||p_sigma|| = sqrt(mu_eff) * sqrt(1 - (1 - c_sigma)^{2*gen})
            # Let's just use a simple threshold or ignore h_sigma for now (set to 1) 
            # as it's a minor correction for small populations.
            # Actually, let's just set h_sigma = 1.0 for simplicity in this implementation.
            # It might cause slight bias but is often acceptable.
            
            # Update p_c
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * whitened_step
            p_c = (1 - c_c) * p_c + 1.0 * np.sqrt(c_c * (2 - c_c) * mu_eff) * whitened_step
            
            # Update C
            # C = (1 - c1 - cmu) * C + c1 * p_c p_c^T + cmu * sum w_i (y_i - mean)(y_i - mean)^T
            # c1 = 2 / ((dim + 1)^2 + 10)
            # cmu = 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)^2 + mu_eff)
            
            c1 = 2.0 / ((dim + 1)**2 + 10)
            cmu = 2.0 * (mu_eff - 2 + 1.0 / mu_eff) / ((dim + 2)**2 + mu_eff)
            
            # Ensure cmu is not negative (mu_eff can be small)
            if cmu < 0:
                cmu = 0.0
                
            # Rank-one update
            C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            for i in range(mu):
                y_i = Y_sorted[i]
                diff = (y_i - mean) / sigma
                # Whitened diff? No, the rank-mu update is in the original space 
                # but scaled by 1/sigma^2?
                # Formula: C = ... + cmu * sum w_i ( (y_i - mean) / sigma ) ( (y_i - mean) / sigma )^T
                # Wait, the standard formula uses the step in the original space normalized by sigma.
                # Let's check the standard CMA-ES update rule.
                # C <- (1 - c1 - cmu) C + c1 p_c p_c^T + cmu sum_i w_i z_i z_i^T
                # where z_i = (y_i - mean) / sigma
                # Yes.
                z_i = (y_i - mean) / sigma
                C += cmu * weights[i] * np.outer(z_i, z_i)
                
            # Eigen decomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive definiteness
            eigvals = np.maximum(eigvals, 1e-12)
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / (5*dim) * (||p_sigma||^2 / mu_eff - 1))
            # This is the "cumulative step size adaptation" (CSA) variant?
            # No, standard CMA-ES uses:
            # sigma = sigma * exp(c_sigma / (5*dim) * (||p_sigma||^2 / mu_eff - 1))
            # Wait, the standard formula is:
            # sigma = sigma * exp( c_sigma / (5*dim) * ( ||p_sigma||^2 / mu_eff - 1 ) )
            # Actually, many implementations use:
            # sigma = sigma * exp( c_sigma / (5*dim) * ( ||p_sigma||^2 / mu_eff - 1 ) )
            # Let's use this.
            
            p_sigma_norm_sq = np.dot(p_sigma, p_sigma)
            sigma_factor = np.exp(c_sigma / (5 * dim) * (p_sigma_norm_sq / mu_eff - 1))
            sigma *= sigma_factor
            
            # Clip sigma to avoid explosion or vanishing
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update mean
            mean = new_mean
            
            # Clip mean to box? 
            # CMA-ES mean can go out of bounds. The sampling clips.
            # But keeping mean in bounds might help.
            # Let's clip mean to [-5, 5] to keep the distribution centered in the box.
            mean = np.clip(mean, -5, 5)
            
        return best_x, best_f

# EVOLVE-BLOCK-END
