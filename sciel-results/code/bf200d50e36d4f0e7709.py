import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        # Rule of thumb: lambda = 4 + floor(3 ln D)
        lam = 4 + int(3 * np.log(D)) if D > 1 else 4
        # For D=5, lam = 4 + 3*1.609 = 8.8 -> 8
        # Let's use a slightly larger population for robustness, maybe 10 or 12.
        # Standard CMA-ES often uses lambda ~ 4 + 3lnD.
        # Let's stick to standard but ensure it's at least 2*D+1? No, CMA-ES works with smaller.
        # Let's use lam = 10 for D=5 to balance exploration.
        lam = 10 
        
        # Selection size mu = lam / 2
        mu = lam // 2
        
        # Weights
        # Standard weights: w_i = log(mu + 0.5) - log(i) for i=1..mu
        # Normalized to sum to 1
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= weights.sum()
        
        # Variance-effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        # Step size sigma
        sigma = 1.0
        # Initial mean x
        x = np.zeros(D)
        # Covariance matrix C
        C = np.eye(D)
        # Evolution path for step size control
        ps = np.zeros(D)
        # Evolution path for covariance matrix adaptation
        pc = np.zeros(D)
        
        # Learning rates
        cs = (mu_eff + 2) / (D + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (D + 1)) - 1) + cs
        cmu = 2 / ((D + 2)**2 + mu_eff)
        cc = (4 - mu_eff / D) / (D + 4 - mu_eff / D)
        c1 = (2 - cmu + cc) / ((D + 1.3)**2 + mu_eff)
        
        # Cholesky decomposition for sampling
        # We need to maintain C = C. We will update C and recompute Cholesky periodically or every step?
        # For D=5, Cholesky is cheap.
        
        best_x = x.copy()
        best_f = func(x)
        calls_made = 1
        
        # Main loop
        while calls_made + lam <= budget:
            # 1. Sample new population
            # Compute Cholesky factor of C
            # Ensure C is symmetric positive definite
            C_sym = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C_sym)
            except np.linalg.LinAlgError:
                # Fallback if numerical issues arise
                C_sym = C_sym + 1e-16 * np.eye(D)
                L = np.linalg.cholesky(C_sym)
            
            Y = np.random.randn(lam, D)
            Y = Y @ L.T  # Y ~ N(0, C)
            X = np.tile(x, (lam, 1)) + sigma * Y
            
            # Evaluate
            F = np.array([func(xi) for xi in X])
            calls_made += lam
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # 2. Selection
            # Sort indices by fitness
            idx_sorted = np.argsort(F)
            selected_X = X[idx_sorted[:mu]]
            selected_F = F[idx_sorted[:mu]]
            
            # 3. Update mean
            # Weighted mean of selected individuals
            x_new = np.sum(weights[:, np.newaxis] * selected_X, axis=0)
            
            # 4. Update step size sigma
            # Compute new evolution path ps
            C_inv_sqrt = np.linalg.inv(L)  # Since L @ L.T = C, L^-1 @ (L^-1).T = C^-1
            # Actually, we need C^-1/2. L is lower triangular such that L L^T = C.
            # So C^-1/2 = (L^-1)^T ? No.
            # If C = L L^T, then C^-1 = L^-T L^-.
            # C^-1/2 is the matrix M such that M M^T = C^-1.
            # M = L^-.
            # Let's verify: L^- (L^-)^T = L^- L^-T = (L L^T)^-1 = C^-1. Yes.
            # So C_inv_sqrt = np.linalg.inv(L)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ (x_new - x) / sigma
            
            # Check if ps norm is too large
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_made / lam))) < (1.4 + 2 / (D + 1)) * np.sqrt(D):
                pass # Do nothing, just a check usually for termination or adjustment
            
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(D) - 1))
            
            # 5. Update covariance matrix
            # Compute new evolution path pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x) / sigma
            
            # Rank-1 update
            # Rank-mu update
            # Combined update
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - cc)**2) * C) + cmu * sum(w_i * (x_i - x_old) (x_i - x_old).T / sigma^2)
            
            # Note: The standard update formula uses the old mean x for the difference.
            # y_i = (x_i - x) / sigma
            Y_selected = (selected_X - x[np.newaxis, :]) / sigma
            
            # Weighted outer product sum
            # W = weights
            # YW = Y_selected.T * weights (mu x D)
            # YW @ Y_selected (D x D)
            rank_mu_update = (Y_selected.T * weights) @ Y_selected
            
            # Rank 1 update term
            # pc pc^T + (1 - (1-cc)^2) C
            # Note: The term (1 - (1-cc)^2) C is often approximated or included.
            # Standard formula:
            # C <- (1 - c1 - cmu) C + c1 (pc pc^T + (1 - (1-cc)^2) C) + cmu * sum ...
            
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * C)
            C += cmu * rank_mu_update
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update mean
            x = x_new
            
        return best_x, best_f
