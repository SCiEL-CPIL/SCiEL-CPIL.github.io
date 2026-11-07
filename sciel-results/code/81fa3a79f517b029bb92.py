"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 330s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        lam = int(4 + 3 * dim)
        # Step size sigma
        sigma = 2.0
        # Initial mean (center of the box, assuming symmetric search space)
        mean = np.zeros(dim)
        
        # Covariance matrix C (initially identity)
        C = np.eye(dim)
        # Inverse square root of C, initially I
        C_sqrt_inv = np.eye(dim)
        
        # Adaptation parameters
        # CMASigmaPS adaptation
        alpha_mu = 1.0  # Simplified adaptation rate
        d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1)
        alpha_cs = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        cs = alpha_cs
        ds = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1) + alpha_cs
        
        # Path for step-size control
        p_sigma = np.zeros(dim)
        # Path for covariance matrix control
        p_c = np.zeros(dim)
        
        # Call counter
        calls = 0
        best_x = mean.copy()
        best_f = func(best_x)
        calls += 1
        
        # Generation loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sample population
            # We need lam samples
            samples = []
            for i in range(lam):
                # Generate random normal vector
                z = rng.standard_normal(dim)
                # Transform to x
                x = mean + sigma * (C_sqrt_inv @ z)
                # Clip to box [-5, 5]
                x = np.clip(x, -5.0, 5.0)
                samples.append(x)
            
            # Evaluate population
            fvals = np.empty(lam)
            for i in range(lam):
                if calls >= budget:
                    break
                fvals[i] = func(samples[i])
                calls += 1
            
            # If budget is exhausted, break
            if calls >= budget:
                # Ensure best is updated with the last valid evaluations if any
                for i in range(lam):
                    if calls - (i+1) < lam: # This logic is tricky if we break early. 
                                             # Better to just rely on the best found so far.
                                             # Actually, if we break, we haven't evaluated all.
                                             # The loop above increments calls.
                                             # We should just check if we ran out of budget mid-eval.
                                             # If so, the remaining fvals are uninitialized. 
                                             # Let's handle the break properly.
                                             pass
                
                # Re-evaluate best tracking logic properly if we broke early
                # Actually, the `if calls >= budget: break` inside the eval loop 
                # means some fvals might not be set. 
                # Let's restart the loop or handle this case.
                # For simplicity and robustness, we'll just break the outer loop 
                # and rely on the `best_x` found so far.
                break
            
            # Sort by fitness
            idx = np.argsort(fvals)
            y_sorted = [samples[i] for i in idx]
            f_sorted = fvals[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y_sorted[0].copy()
            
            # Select top mu individuals (mu = lam // 2)
            mu = lam // 2
            
            # Weights
            w = np.zeros(mu)
            w[0] = np.log(mu + 0.5) - np.log(mu) # Standard weights
            # Normalize weights
            w_sum = np.sum(w)
            w = w / w_sum
            mu_eff = 1.0 / np.sum(w**2)
            
            # Weighted mean of selected individuals
            m_new = np.zeros(dim)
            for i in range(mu):
                m_new += w[i] * y_sorted[i]
            
            # Update covariance matrix path
            p_c = (1 - cs) * p_c + cs * np.sqrt(mu_eff) * C_sqrt_inv @ (m_new - mean)
            
            # Update step-size control path
            p_sigma = (1 - d_sigma) * p_sigma + np.sqrt(mu_eff) * (C_sqrt_inv @ (m_new - mean))
            
            # Update sigma
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2) - 1))
            # Clip sigma to avoid overflow/underflow
            sigma = np.clip(sigma, 1e-10, 1e10)
            
            # Update covariance matrix C
            # Delta H correction
            delta_h = np.exp(
                cs / ds * (np.linalg.norm(p_sigma)**2 / dim - 1)
            )
            if delta_h < 1.0 / (1e10):
                delta_h = 0
            
            # Rank-1 + Rank-mu update
            c1 = (mu_eff - 2 + 1) / ((dim + 1.3)**2 + mu_eff)
            cmu = (2 - c1) * (mu_eff - 2) / ((dim + 2)**2 + mu_eff)
            
            # C = (1 - c1 - cmu + mu_eff * c1) * C + c1 * p_c @ p_c.T + cmu * sum(w_i * y_i @ y_i.T)
            
            # Optimization: Use the fact that C is updated via C_sqrt_inv
            # This is complex to do exactly in one step without updating C_sqrt_inv.
            # For simplicity and speed, we will update C explicitly and recompute C_sqrt_inv.
            
            C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c)
            for i in range(mu):
                y_i = y_sorted[i] - mean
                C += cmu * w[i] * np.outer(y_i, y_i)
            
            # Regularize C to maintain positive definiteness
            C = 0.5 * (C + C.T)
            C += 1e-10 * np.eye(dim)
            
            # Update C_sqrt_inv
            # Eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to be positive
            eigvals = np.clip(eigvals, 1e-10, None)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update mean
            mean = m_new
            # Clip mean to box
            mean = np.clip(mean, -5.0, 5.0)
            
            # If sigma is too small, we might be stuck. 
            # CMA-ES handles this via recentering or just stopping.
            # Given the budget, we just continue until budget is exhausted.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
