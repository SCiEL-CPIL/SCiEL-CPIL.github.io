"""Combined optimizer for BBOB f18 (Schaffers F7 ill-conditioned) using simplified CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simplified CMA-ES optimizer with multi-start exploration.

    Phase 1: Multi-start exploration with short CMA-ES runs to find the basin.
    Phase 2: Committed CMA-ES run from the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x_start, fx_start, sigma, n_calls, rng, deadline):
        """Run a simplified CMA-ES for n_calls evaluations.

        Args:
            func: The objective function.
            x_start: Initial mean vector.
            fx_start: Function value at x_start.
            sigma: Initial step size.
            n_calls: Maximum number of function evaluations allowed.
            rng: NumPy random generator.
            deadline: Time limit in seconds.

        Returns:
            Tuple (x_best, fx_best, sigma, used_calls)
        """
        dim = self.dim
        # Population size lambda = 4 * dim
        lambda_ = 4 * dim
        # Number of selected individuals mu = lambda / 2
        mu = lambda_ // 2
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial covariance matrix
        C = np.eye(dim)
        # Initial inverse square root of C (identity for C=I)
        Cinv_sqrt = np.eye(dim)
        
        # Step size control parameters
        alpha_sigma = 1.0 / (mu_eff * 0.5)
        d_sigma = 1.0 + 2 * max(0, (mu_eff - 2) / (dim + 2))
        alpha_cs = (mu_eff + 2) / (dim + mu_eff + 5)
        cs = alpha_cs * (1 + 2 * max(0, (mu_eff - 1) / (dim + mu_eff)))
        alpha_hs = 0.84 + alpha_cs / 2
        hs = (1.0 / alpha_hs) * (np.sqrt((1 - np.exp(-2 * alpha_cs)) / (1 - np.exp(-2 * alpha_cs * 0.5))) if False else 1.0) # Simplified hs
        # Standard CMA-ES hs calculation is complex, we use a simpler heuristic or standard formula
        # Let's use the standard formula approximation
        hs = 1.0 # Default, will be updated
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial step size
        sigma_current = sigma
        
        x_mean = x_start.copy()
        f_best = fx_start
        x_best = x_start.copy()
        
        used = 0
        # We need to evaluate the initial point? No, fx_start is provided.
        # But CMA-ES usually evaluates the population.
        # We will use the budget for population evaluations.
        
        while used < n_calls:
            if time.time() > deadline:
                break
            
            # Generate population
            # We need lambda_ evaluations per generation.
            # If remaining budget < lambda_, we can do a partial generation or stop.
            if n_calls - used < lambda_:
                # Do a final small batch or just stop to avoid waste?
                # Let's just stop if we can't do a full generation to keep stats clean, 
                # or do a smaller population. 
                # For simplicity and robustness, if we can't do a full pop, we stop.
                break
            
            # Update step size based on previous generation? 
            # In this loop structure, we update at the end of the generation.
            # First generation uses initial sigma.
            
            # Generate candidates
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if C becomes singular
                L = np.eye(dim)
                C = np.eye(dim)
            
            # Generate normal vectors
            N = rng.standard_normal((lambda_, dim))
            # Transform to x
            Y = N @ L.T
            X = np.array([x_mean + sigma_current * y for y in Y])
            
            # Clip to bounds
            X = np.clip(X, -5.0, 5.0)
            
            # Evaluate population
            F = np.array([func(x) for x in X])
            used += lambda_
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < f_best:
                f_best = F[idx_min]
                x_best = X[idx_min].copy()
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Weighted recombination
            z = np.zeros(dim)
            for i in range(mu):
                z += weights[i] * (X_sorted[i] - x_mean) / sigma_current
            
            # Update path p_sigma
            p_sigma = (1 - alpha_sigma) * p_sigma + np.sqrt(alpha_sigma * (2 - alpha_sigma) * mu_eff) * z
            
            # Update step size
            # h_sigma = 1 if ||p_sigma|| / sqrt(1 - (1-alpha_sigma)^{2*gen}) < (1.4 + 2/(dim+1)) * E||N(0,I)||
            # Simplified: check if ||p_sigma|| is not too large
            norm_p_sigma = np.linalg.norm(p_sigma)
            # E||N(0,I)|| = sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim^2))
            e_norm = np.sqrt(dim) * (1 - 1/(4*dim))
            # We need a generation counter for the threshold, but we can approximate
            # Using a simple threshold
            threshold = 1.4 * e_norm
            h_sigma = 1.0 if norm_p_sigma < threshold else 0.0
            
            sigma_current *= np.exp(alpha_sigma / d_sigma * (norm_p_sigma / e_norm - 1))
            
            # Update path p_c
            p_c = (1 - alpha_cs) * p_c + h_sigma * np.sqrt(alpha_cs * (2 - alpha_cs) * mu_eff) * z
            
            # Update mean
            x_mean += sigma_current * z
            
            # Update covariance matrix
            # Rank-mu update
            H = np.array([X_sorted[i] - x_mean for i in range(mu)])
            C = (1 + alpha_cs * (2 - alpha_cs) * (1 - h_sigma) * alpha_cs + mu_eff * weights[0] - 1) * C
            C += mu_eff * weights[0] * np.outer(z, z)
            for i in range(mu):
                C += (1 - weights[i]) * np.outer(H[i], H[i])
            
            # Rank-1 update (already included in the above formula if we expand, but let's do it explicitly for clarity)
            # Actually, the standard formula is:
            # C = (1 + mu_eff*weights[0] + 2*alpha_cs*(2-alpha_cs)*(1-h_sigma) - 1) * C 
            #     + mu_eff*weights[0]*z*z^T 
            #     + sum_{i=1}^{mu-1} (1-weights[i]) * (x_i - x_mean)(x_i - x_mean)^T
            # Wait, the standard CMA-ES update is:
            # C = (1 + c_sigma * (2 - c_sigma) * (1 - h_sigma) + c_1 * mu_eff * w_0 - 1) * C
            #     + c_1 * (mu_eff * w_0 * z * z^T + sum_{i=1}^{mu-1} (1 - w_i) * (y_i) * (y_i)^T)
            # where c_1 = 2 / ((dim+1.3)^2 + mu_eff)
            # Let's re-implement the update correctly.
            
            c1 = 2.0 / ((dim + 1.3)**2 + mu_eff)
            # Recalculate C update
            C_new = C.copy()
            # The term (1 + ...) * C
            factor = 1 + alpha_cs * (2 - alpha_cs) * (1 - h_sigma) + c1 * mu_eff * weights[0]
            C_new = factor * C_new
            # Add rank-1 update for z
            C_new += c1 * mu_eff * weights[0] * np.outer(z, z)
            # Add rank-(mu-1) updates
            for i in range(1, mu):
                y_i = X_sorted[i] - x_mean
                C_new += c1 * (1 - weights[i]) * np.outer(y_i, y_i)
            
            # Ensure symmetry and positive semi-definiteness
            C = (C_new + C_new.T) / 2.0
            # Regularize if needed
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except np.linalg.LinAlgError:
                C = np.eye(dim)
            
            # Update inverse square root for next iteration
            # This is expensive, but dim is small (5)
            try:
                L = np.linalg.cholesky(C)
                Cinv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                Cinv_sqrt = np.eye(dim)

        return x_best, f_best, sigma_current, used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start exploration
        # Use 20% of budget for exploration
        phase1_budget = max(10, int(budget * 0.2))
        n_starts = 5
        per_start_budget = phase1_budget // n_starts
        
        best_x = None
        best_f = np.inf
        calls = 0

        for i in range(n_starts):
            if calls >= phase1_budget or time.time() > deadline:
                break
            
            # Random start
            x0 = rng.uniform(-5.0, 5.0, dim)
            fx0 = func(x0)
            calls += 1
            
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            
            # Short CMA-ES run from this start
            # Start with a moderate sigma
            sigma0 = 1.5
            x_res, f_res, sigma_res, used = self._cma_es(
                func, x0, fx0, sigma0, per_start_budget, rng, deadline
            )
            calls += used
            
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # Phase 2: Committed CMA-ES run
        remaining = max(0, budget - calls)
        if remaining > 0 and best_x is not None:
            # Start with a slightly smaller sigma to refine
            sigma0 = 1.0
            x_res, f_res, sigma_res, used = self._cma_es(
                func, best_x, best_f, sigma0, remaining, rng, deadline
            )
            calls += used
            
            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
