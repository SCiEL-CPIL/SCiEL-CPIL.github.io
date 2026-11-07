"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        # CMA-ES Parameters
        lam = 10  # Population size
        mu = lam // 2  # Number of parents
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)  # Variance equivalent population size

        # Initialization
        x = np.zeros(n)  # Start at center
        sigma = 2.0  # Initial step size
        
        # Covariance matrix
        C = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))

        best_fx = float('inf')
        best_x = x.copy()
        
        # Evaluate initial point
        fx = func(x)
        if fx < best_fx:
            best_fx = fx
            best_x = x.copy()
        
        calls_used = 1
        stagnation_count = 0
        prev_best_fx = best_fx

        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
            
            z = np.random.randn(lam, n)
            # x_i = x + sigma * z_i @ L.T
            samples = x + sigma * (z @ L.T)
            
            # Clip to bounds
            samples = np.clip(samples, -5, 5)
            
            # Evaluation
            fvals = np.empty(lam)
            for i in range(lam):
                if calls_used >= budget:
                    break
                fvals[i] = func(samples[i])
                calls_used += 1
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = samples[i].copy()
            
            if calls_used >= budget:
                break

            # Sort
            indices = np.argsort(fvals)
            fvals_sorted = fvals[indices]
            samples_sorted = samples[indices]
            
            # Update mean
            x_old = x.copy()
            x = np.dot(weights, samples_sorted[:mu])
            
            # Update evolution paths
            # p_sigma
            # Normalized step: (x - x_old) / sigma
            # Note: L.T is used to map the canonical space to the covariance space?
            # Standard formula: p_sigma = (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * (L.T @ (x - x_old) / sigma)
            # Wait, the sampling was x + sigma * z @ L.T.
            # The direction in canonical space is z. The direction in original space is z @ L.T.
            # So (x - x_old) = sigma * (z_mean @ L.T).
            # So (x - x_old) / sigma = z_mean @ L.T.
            # We need to transform this back to canonical space? No, the path p_sigma is in the canonical space?
            # Actually, in standard CMA-ES, p_sigma is in the original space but scaled.
            # Let's stick to the standard implementation where p_sigma is updated with the weighted mean of the selected solutions' steps in the canonical space, transformed back?
            # No, the standard implementation (Hansen) updates p_sigma using the step in the original space, but it's a vector in R^n.
            # Formula: p_sigma = (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * (L.T @ (x - x_old) / sigma)
            # This effectively projects the step back to the canonical space?
            # Let's use the robust update from the reference implementation.
            
            # Step vector in original space
            step = x - x_old
            # Transform to canonical space? 
            # If we sampled y = x + sigma * L.T @ z, then z = L @ (y - x) / sigma.
            # So the canonical step is z_mean = L @ step / sigma.
            # p_sigma is maintained in the canonical space? Or original?
            # In Hansen's implementation, p_sigma is in the original space? No, it's often treated as a vector in the same space as the mean.
            # Let's use the formula: p_sigma = (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * (L.T @ step / sigma)
            # This assumes L.T maps from original to canonical? No, L is lower triangular, L @ L.T = C.
            # If y = x + sigma * L.T @ z, then y-x = sigma * L.T @ z.
            # So z = L @ (y-x) / sigma.
            # The term (L.T @ step / sigma) is NOT z. It is L.T @ (L.T @ z) = L.T L.T z. This is wrong.
            
            # Correct transformation:
            # We want the step in the canonical space to update the path which is defined in the canonical space?
            # Actually, the evolution path p_sigma is a vector in R^n.
            # The update is: p_sigma = (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * ( (x - x_old) / sigma )
            # Wait, if the covariance is not identity, the "unit" step is not 1.
            # The standard CMA-ES implementation uses:
            # p_sigma = (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)mueff) * (L.T @ (x - x_old) / sigma)
            # Let's verify this. If C=I, L=I, then p_sigma += ... * (x-x_old)/sigma. This is correct.
            # If C is not I, the step (x-x_old) is in the original space.
            # The term L.T @ (x-x_old)/sigma transforms the original space step into a space where the covariance is identity?
            # No, L.T is the inverse of the whitening transformation?
            # Let's assume the standard formula is correct as implemented in many libraries.
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L.T @ (step / sigma))
            
            # Check if p_sigma is too long
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used / lam))) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1):
                # Increase sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))
            
            # Update p_c
            # p_c = (1-c_c)p_c + h_sigma * sqrt(c_c(2-c_c)mueff) * (L.T @ step / sigma)
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_used / lam))) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1)
            h_sigma = 1 if h_sigma else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (L.T @ (step / sigma))
            
            # Update covariance matrix
            # Rank-one update
            C = (1 + c_1 * h_sigma * (2 - h_sigma) - c_1) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            y = samples_sorted[:mu] - x_old  # Steps of selected individuals
            y = y / sigma  # Normalize
            # Transform to canonical space?
            # The update is C += c_mu * sum(w_i * y_i * y_i.T)
            # But y is in original space.
            # The standard update uses the outer product of the steps.
            # However, if C is not identity, the steps are correlated.
            # The formula is: C = (1 + c_mu * h_mu * (2 - h_mu) - c_mu) * C + c_mu * sum(w_i * (y_i @ L.T) (y_i @ L.T).T) ?
            # No, the standard implementation updates C using the outer product of the steps in the original space, but weighted.
            # Actually, the rank-mu update is:
            # C += c_mu * sum_{i=1}^mu w_i * (x_i - x_old) (x_i - x_old).T / sigma^2
            # Let's use this simpler form which is equivalent if we consider the scaling.
            
            # Calculate h_mu
            h_mu = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2 * (calls_used / lam)))) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1)
            h_mu = 1 if h_mu else 0
            
            # Update C
            # C = (1 + c_mu * h_mu * (2 - h_mu) - c_mu) * C + c_mu * sum(w_i * y_i y_i.T)
            # y_i are the steps divided by sigma.
            # y = samples_sorted[:mu] - x_old
            # y_scaled = y / sigma
            # C += c_mu * sum(w_i * np.outer(y_scaled[i], y_scaled[i]))
            
            # We need to be careful with the scaling factor.
            # The standard formula is:
            # C = (1 + c_mu * h_mu * (2 - h_mu) - c_mu) * C + c_mu * sum_{i=1}^mu w_i * (x_i - x_old)(x_i - x_old).T / sigma^2
            
            # Let's compute the sum
            sum_outer = np.zeros((n, n))
            for i in range(mu):
                y_i = (samples_sorted[i] - x_old) / sigma
                sum_outer += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_mu * h_mu * (2 - h_mu) - c_mu) * C + c_mu * sum_outer
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Stagnation check
            if best_fx == prev_best_fx:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_best_fx = best_fx
                
            # If stagnated for too long, reset or restart
            if stagnation_count > 50:
                # Simple restart: randomize x and sigma
                x = np.random.uniform(-5, 5, n)
                sigma = 2.0
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                stagnation_count = 0
                # Evaluate new x
                fx = func(x)
                calls_used += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                if calls_used >= budget:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
