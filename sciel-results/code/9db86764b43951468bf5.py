"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimized for Bent Cigar (f12) with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit

        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For small dimensions and tight budgets, a larger lambda helps exploration
        lam = 10 
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / ((n + 1.3) ** 2 + mueff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        
        # Start near center, but slightly offset to ensure movement
        # Bent Cigar is often rotated, so center is a good start
        x_mean = np.zeros(n)
        sigma = 2.0  # Initial step size, roughly half the box width
        
        # Covariance matrix
        C = np.eye(n)
        # Pre-compute Cholesky factor for sampling
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(n)

        # Evolution paths
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Best tracking
        best_x = x_mean.copy()
        best_fx = func(best_x)
        
        calls = 1
        
        # Main loop
        # We run until budget is exhausted or time limit is reached
        # Each iteration consumes 'lam' function evaluations
        while calls + lam <= budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            pop = np.empty((lam, n))
            fvals = np.empty(lam)
            
            for i in range(lam):
                z = rng.standard_normal(n)
                x_i = x_mean + sigma * (C_sqrt @ z)
                x_i = np.clip(x_i, -5, 5)
                pop[i] = x_i
                fvals[i] = func(x_i)
                calls += 1
                
                # Update best
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = x_i.copy()
            
            if calls >= budget or time.time() > deadline:
                break

            # 2. Sort population by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]
            
            # 3. Update mean (weighted sum of best mu individuals)
            x_mean_new = np.zeros(n)
            for i in range(mu):
                x_mean_new += weights[i] * pop_sorted[i]
            
            # 4. Update step-size control (path ps)
            # Note: We use the weighted mean difference
            diff = x_mean_new - x_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_sqrt @ diff)
            norm_ps = np.linalg.norm(ps)
            
            # 5. Update covariance adaptation (path pc)
            # Heuristic for h_sig (threshold for correlation)
            # Using a simplified version of the standard CMA-ES heuristic
            # The standard formula involves the expected norm of ps
            # E[||ps||] approx chi_n * (1 - (1-c_sigma)^(t/lam)) ... 
            # For simplicity and stability in small runs, we use a fixed threshold 
            # or the standard approximation.
            # Standard: h_sig = (norm_ps / sqrt(1 - (1 - c_sigma)^(2*(calls+1)/lam))) < (1.4 + 2/(n+1)) * chi_n
            # We need to be careful with the exponent term if calls is small.
            # Let's use the standard implementation's logic.
            
            # Calculate the threshold
            # The term (1 - c_sigma)^(2 * (calls + 1) / lam)
            # Since we are in a loop, calls increases by lam each time.
            # Let's just compute it.
            exp_term = 1 - (1 - c_sigma) ** (2 * (calls + 1) / lam)
            if exp_term < 1e-10:
                exp_term = 1e-10
            h_sig = (norm_ps / np.sqrt(exp_term)) < (1.4 + 2 / (n + 1)) * chi_n
            
            pc = (1 - c_c) * pc + h_sig * c_c * diff / sigma
            
            # 6. Update Covariance Matrix C
            # C <- (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * delta_hsig * pc*pc^T + c_mu * sum w_i (x_i - x_mean)*(x_i - x_mean)^T
            
            delta_hsig = (1 - h_sig) * c_c * (1 - c_c)
            
            # Rank-1 update from pc
            # C = (1 - c_1 - c_mu) * C + c_1 * delta_hsig * np.outer(pc, pc)
            # Plus rank-mu update from population
            
            # To avoid numerical issues, we can update in place or create new
            C_new = (1 + c_1 * delta_hsig - c_1 - c_mu) * C
            C_new += c_1 * delta_hsig * np.outer(pc, pc)
            
            for i in range(mu):
                diff_i = pop_sorted[i] - x_mean
                C_new += c_mu * weights[i] * np.outer(diff_i, diff_i)
            
            C = C_new
            
            # 7. Update Mean
            x_mean = x_mean_new
            
            # 8. Update Sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            
            # 9. Update Cholesky Factor
            # Regularize C if it becomes non-positive definite
            # Check for small eigenvalues or use try-except
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, regularize C
                # Add small diagonal term
                C = C + 1e-3 * np.eye(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset to identity if still failing
                    C = np.eye(n)
                    C_sqrt = np.eye(n)
                    pc = np.zeros(n)
                    ps = np.zeros(n)
                    sigma = 2.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
