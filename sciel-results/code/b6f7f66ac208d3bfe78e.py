"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin for 330s limit
        
        n = self.dim
        # CMA-ES parameters
        # Default pop size: 4 + floor(3 * ln(n))
        pop_size = 4 + int(3 * np.log(n))
        # Ensure pop size is at least 2
        if pop_size < 2:
            pop_size = 2
            
        # Initialization
        # Start at 0, which is often a good heuristic for BBOB functions centered near 0
        # or use a small random offset if we want to be less deterministic, but 0 is safe.
        x_mean = np.zeros(n)
        
        # Initial standard deviation
        # The search space is [-5, 5]. The minimizer is in [-4, 4].
        # A sigma of 1.0 or 2.0 is reasonable. 1.0 allows finer exploration initially.
        sigma = 1.0
        
        # CMA-ES state variables
        C = np.eye(n)  # Covariance matrix
        C_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Learning rates (standard CMA-ES values)
        mu = pop_size // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mu_eff / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Pre-compute square roots for efficiency
        # We need to sample from N(0, C). We can do this by transforming N(0, I)
        # using the Cholesky decomposition of C.
        
        def sample_from_covariance(C):
            # Use Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite (should be rare with CMA-ES)
                L = np.eye(n)
            z = np.random.normal(0, 1, (pop_size, n))
            return z @ L.T

        def inv_sqrt(C):
            # Compute C^(-1/2) for normalization in update rules
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                return C_inv_sqrt
            except np.linalg.LinAlgError:
                return np.eye(n)

        C_inv_sqrt = inv_sqrt(C)

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Main loop
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # Sample from N(x_mean, sigma^2 * C)
            z = np.random.normal(0, 1, (pop_size, n))
            # Transform to have covariance C
            # We need C^(1/2). We can use the Cholesky factor from sampling or recompute.
            # For efficiency in the loop, we can update the Cholesky factor or just use SVD/eigh every step.
            # Given n=5, eigh is fast enough.
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(n)
                
            pops = x_mean + sigma * (z @ C_sqrt.T)
            
            # Evaluate
            fvals = np.array([func(p) for p in pops])
            calls += pop_size
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pops[min_idx].copy()
                
            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = pops[order]
            f_sorted = fvals[order]
            
            # Weighted mean of the mu best points
            y_w = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            
            # Update mean
            x_mean_new = x_mean + c_c * (y_w - x_mean)
            
            # Update step size
            # p_sigma update
            # We need C^(-1/2) (x_mean_new - x_mean) / sigma
            diff = x_mean_new - x_mean
            p_sigma_new = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            # Update sigma
            # The h_sigma condition checks if the norm of p_sigma is small compared to expected norm
            # E[||N(0, I)||] = sqrt(n) * (1 - 1/(4n) + ...)
            # A common approximation is sqrt(n)
            # The exact formula involves the norm of the identity matrix transformed by C
            # For simplicity and robustness, we use the standard CMA-ES update
            # Note: The 'h_sigma' logic is slightly complex. A simplified version:
            # If the step is successful, sigma increases.
            # Standard formula:
            # sigma = sigma * exp( c_sigma / d_sigma * ( ||p_sigma|| / E[||N(0,I)||] - 1 ) )
            # E[||N(0,I)||] is approx sqrt(n)
            norm_p_sigma = np.linalg.norm(p_sigma_new)
            e_norm = np.sqrt(n)
            # To be more accurate, we can use the norm of the identity matrix in the transformed space
            # But sqrt(n) is the standard baseline.
            # We should check if the change in mean was small relative to sigma
            # A more robust check is comparing ||p_sigma|| to sqrt(1 - (1-c_sigma)^2) * E[||N(0,I)||]
            # Let's stick to the standard exponential update.
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / e_norm - 1))
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update covariance matrix
            # p_c update
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            # Wait, the update for p_c uses the difference in means divided by sigma?
            # Actually, p_c is updated with the step direction normalized by sigma.
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - x_mean) / sigma
            
            # Let's re-verify the CMA-ES update formulas.
            # 1. x_mean_{t+1} = x_mean_t + c_c * (y_w - x_mean_t)
            # 2. p_sigma_{t+1} = (1 - c_sigma) * p_sigma_t + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{-1/2} (x_mean_{t+1} - x_mean_t) / sigma_t
            # 3. sigma_{t+1} = sigma_t * exp( c_sigma / d_sigma * ( ||p_sigma_{t+1}|| / E[||N(0,I)||] - 1 ) )
            #    Where E[||N(0,I)||] = sqrt(n) * (1 - 1/(4n) + 1/(21n^2)) approx sqrt(n)
            # 4. p_c_{t+1} = (1 - c_c) * p_c_t + sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_{t+1} - x_mean_t) / sigma_t
            #    (Note: Some implementations use sigma_{t+1}, some sigma_t. Using sigma_t is standard for the 'success' signal)
            # 5. h_c = 1 if ||p_c|| / sqrt(1 - (1 - c_c)^{2(t+1)}) < (1.4 + 2/(n+1)) * E[||N(0,I)||] else 0
            # 6. C_{t+1} = (1 - c_1 - c_mu) * C_t + c_1 * (p_c_{t+1} p_c_{t+1}^T + (1 - h_c) * c_c * (2 - c_c) * C_t) + c_mu * sum w_i (y_i - x_mean_t)(y_i - x_mean_t)^T / sigma_t^2
            
            # Let's implement step 4, 5, 6 carefully.
            
            diff_mean = x_mean_new - x_mean
            p_c_new = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff_mean / sigma
            
            # h_c check
            # The term (1 - (1 - c_c)^{2(t+1)}) is the variance of the random walk.
            # For long runs, it approaches 1.
            # Let's use a simplified h_c = 1 if ||p_c_new|| < (1.4 + 2/(n+1)) * e_norm else 0
            # This is a common simplification.
            norm_p_c = np.linalg.norm(p_c_new)
            threshold = (1.4 + 2 / (n + 1)) * e_norm
            h_c = 1.0 if norm_p_c < threshold else 0.0
            
            # Update C
            # Rank-1 update
            # (p_c_new p_c_new^T + (1 - h_c) * c_c * (2 - c_c) * C)
            rank1 = np.outer(p_c_new, p_c_new)
            if h_c == 0.0:
                rank1 += c_c * (2 - c_c) * C
            
            # Rank-mu update
            # sum w_i (y_i - x_mean) (y_i - x_mean)^T / sigma^2
            # We use the sorted best mu individuals
            y_diffs = (y_sorted[:mu] - x_mean) / sigma
            rank_mu = np.sum(weights[:, None, None] * y_diffs[:, :, None] * y_diffs[:, None, :], axis=0)
            
            C_new = (1 - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rank_mu
            
            # Ensure C is positive definite and symmetric
            C_new = (C_new + C_new.T) / 2
            
            # Regularize C if necessary
            # Check condition number or eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(C_new)
                if np.min(eigvals) < 1e-10:
                    # Add small identity
                    C_new += 1e-10 * np.eye(n)
            except:
                C_new = np.eye(n)
            
            # Update state
            x_mean = x_mean_new
            p_sigma = p_sigma_new
            p_c = p_c_new
            C = C_new
            C_inv_sqrt = inv_sqrt(C)
            
            # Early stopping if progress is minimal
            if sigma < 1e-8:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
