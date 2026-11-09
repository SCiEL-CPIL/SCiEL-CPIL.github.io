"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with time-aware budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin well before 330s limit
        
        rng = np.random.default_rng(42)
        
        # Initial guess in center
        x = np.zeros(self.dim)
        fx = func(x)
        
        # CMA-ES parameters
        # Standard CMA-ES uses lambda = 4 + floor(3 log n)
        lam = 4 + int(3 * np.log(self.dim))
        # Step size
        sigma = 2.0
        # Covariance matrix
        C = np.eye(self.dim)
        # Inverse square root of C
        Cs = np.eye(self.dim)
        # Evolution paths
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # Adaptation parameters
        cmu = 2 / ((self.dim + 4)**2 + lam)
        ccs = (4 + lam / self.dim) / (self.dim + 4 + 2 * lam / self.dim)
        cc = (2 + 2 * min(1.0, 1 - ccs) + cmu) / (self.dim + 2 + 2 * min(1.0, 1 - ccs))
        cs = (self.dim + 1.3) / (self.dim + 5.5)
        # Step size adaptation (1/5 rule on path length)
        hsig = (1 + 2 * max(0, (1 - ccs) / (1 + ccs))) * (1 - np.trace(C) / self.dim)
        # Note: hsig calculation in standard CMA-ES is slightly different, using expected norm
        # E[||N(0,I)||]
        e_norm = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(32*self.dim**2))
        
        calls_made = 1
        
        # Pre-allocate for population
        # We will generate a population of size lam in each iteration
        # Total budget is self.budget.
        # Iterations allowed = (budget - 1) // lam
        
        while calls_made < self.budget:
            if time.time() > deadline - 5: # Stop if close to deadline
                break
                
            # Generate population
            # y = Cs @ z
            # We need to generate lam standard normal vectors
            # To save time, we can generate them all at once
            Z = rng.standard_normal((lam, self.dim))
            # Transform to X
            # X = x + sigma * Y
            # Y = Cs @ Z.T
            Y = Z @ Cs.T
            Pop = x + sigma * Y
            # Clip to box
            Pop = np.clip(Pop, -5, 5)
            
            # Evaluate population
            # We must be careful not to exceed budget
            # We have budget - calls_made left
            # If lam > remaining, we can't do a full population step?
            # The task says "More than 500 calls... makes the whole result invalid".
            # So we must strictly stay <= 500.
            
            # If remaining budget < lam, we can either stop or do a smaller population?
            # Standard CMA-ES requires full population for adaptation.
            # If we can't do a full population, we should stop or do (1+1) style?
            # Given the strict limit, let's just stop if we can't afford a full population 
            # or adjust lam? No, lam is fixed.
            # Better: Check if we can afford lam calls.
            
            if self.budget - calls_made < lam:
                break
                
            F = np.array([func(xi) for xi in Pop])
            calls_made += lam
            
            # Sort by fitness
            idx = np.argsort(F)
            # Take top mu = lam/2
            mu = lam // 2
            idx_top = idx[:mu]
            
            # Update mean
            # Weights
            # Standard CMA-ES uses w_i = log(mu+0.5) - log(i+1)
            w = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
            w /= np.sum(w)
            
            x_new = np.sum(w[:, None] * Pop[idx_top], axis=0)
            
            # Update evolution path ps
            # ps = (1-cs)ps + cs/sigma * sqrt(mu_eff) * C^-1/2 (x_new - x)
            # We need C^-1/2 which is Cs.T (since C = Cs Cs.T, so C^-1/2 = (Cs.T)^-1 ... wait)
            # If C = Cs @ Cs.T, then C^-1/2 = (Cs.T)^-1
            # Actually, we maintain Cs such that C = Cs Cs^T.
            # Then C^-1/2 is the inverse of Cs^T.
            # To avoid explicit inversion, we can solve linear systems.
            # Let's update Cs using the rank-mu update.
            
            # First, update ps
            # diff = x_new - x
            diff = x_new - x
            # We need C^-1/2 @ diff. 
            # Since C = Cs Cs^T, C^-1/2 = (Cs^T)^-1.
            # So C^-1/2 @ diff = solve(Cs^T, diff).
            # Or we can just compute it if we maintain the inverse?
            # Standard implementation updates Cs directly.
            
            # Let's use the standard update rules for C and Cs.
            
            # 1. Update ps
            # ps = (1-cs) * ps + cs/sigma * sqrt(mu_eff) * C^-1/2 (x_new - x)
            # mu_eff = 1 / sum(w_i^2)
            mu_eff = 1.0 / np.sum(w**2)
            
            # Compute C^-1/2 (x_new - x)
            # We can solve Cs.T y = diff for y
            try:
                y = np.linalg.solve(Cs.T, diff)
            except np.linalg.LinAlgError:
                y = diff / np.linalg.norm(diff) # Fallback
            
            ps = (1 - cs) * ps + cs / sigma * np.sqrt(mu_eff) * y
            
            # 2. Update pc
            # hsig = 1 if ||ps|| / sqrt(1-(1-cs)^{2it+2}) < (1.4 + 2/(dim+1)) * E[||N(0,I)||]
            # Approximation for steady state:
            # E[||N(0,I)||] approx sqrt(dim)
            # Threshold = (1.4 + 2/(dim+1)) * e_norm
            thresh = (1.4 + 2 / (self.dim + 1)) * e_norm
            if np.linalg.norm(ps) < thresh:
                hsig = 1.0
            else:
                hsig = 0.0
                
            pc = (1 - cc) * pc + hsig * cc * (x_new - x) / sigma
            
            # 3. Update C
            # C = (1 + cmu * (mu_eff + 2) - cc) C + cmu * sum(w_i (y_i y_i^T)) + (1 + cmu * (mu_eff + 2) - cc) * ... wait
            # The standard update is:
            # C = (1 + cmu (mu_eff + 2) - cc) C + cmu * sum_{i=1}^mu w_i (y_i y_i^T) + (1 + cmu (mu_eff + 2) - cc) * (pc pc^T - C) * (1 - hsig) ? No.
            
            # Correct formula:
            # C = (1 + cmu (mu_eff + 2) - cc) C + cmu * sum_{i=1}^mu w_i (y_i y_i^T)
            # Where y_i = (x_i - x_old) / sigma
            # And there is a correction term for pc if hsig=0?
            # Actually, the standard CMA-ES update for C is:
            # C = (1 + cmu (mu_eff + 2) - cc) C + cmu * sum_{i=1}^mu w_i (y_i y_i^T)
            # BUT, the covariance matrix is updated using the actual steps taken.
            # y_i = (Pop[idx_top[i]] - x) / sigma
            
            Y_top = (Pop[idx_top] - x[None, :]) / sigma
            
            # Rank-mu update
            # C_new = (1 + cmu (mu_eff + 2) - cc) * C + cmu * sum(w_i y_i y_i^T)
            # Wait, the term (1 + cmu (mu_eff + 2) - cc) ensures trace(C) is preserved if no adaptation?
            # Actually, the standard formula is:
            # C = (1 + cmu (mu_eff + 2) - cc) C + cmu * sum_{i=1}^mu w_i (y_i y_i^T)
            # This formula is correct for the rank-mu update.
            
            C = (1 + cmu * (mu_eff + 2) - cc) * C + cmu * (Y_top.T * w) @ Y_top
            
            # 4. Update sigma
            # sigma = sigma * exp(cs / d_sigma * (||ps|| / e_norm - 1))
            # d_sigma = (self.dim + 2) / 2 + min(cmu, 0.5) * (mu_eff / 2 - 1)
            d_sigma = (self.dim + 2) / 2.0 + min(cmu, 0.5) * (mu_eff / 2.0 - 1.0)
            sigma = sigma * np.exp(cs / d_sigma * (np.linalg.norm(ps) / e_norm - 1.0))
            
            # 5. Update Cs (inverse square root)
            # We need to update Cs such that Cs Cs^T = C_new.
            # This requires a matrix square root or eigen decomposition.
            # Eigen decomposition is O(n^3). For dim=5, it's fast.
            # C is symmetric positive definite.
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive
                eigvals = np.maximum(eigvals, 1e-12)
                Cs = eigvecs @ np.diag(np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                # Fallback: reset to identity if it fails (should not happen)
                Cs = np.eye(self.dim)
                C = np.eye(self.dim)
            
            # Update x
            x = x_new
            # Update fx
            # We already have F sorted. The best is F[idx[0]]
            fx = F[idx[0]]
            
        return x, fx

# EVOLVE-BLOCK-END
