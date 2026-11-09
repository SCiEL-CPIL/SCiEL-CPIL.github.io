import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(d))
        if lam % 2 == 0:
            lam += 1
        # Number of recombination weights
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        cs = (mu_eff + 2) / (d + mu_eff + 5)
        c1 = 2 / ((d + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Initialization
        # Start at origin, which is often close to the center of the search space
        m = np.zeros(d)
        C = np.eye(d)
        # Initial sigma
        sigma = 2.0
        
        # Memory for adaptation
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Pre-compute some constants
        hsig = 1.0 + 2.0 * max(0.0, (np.sqrt((1 - np.exp(-2.0 / (budget + 1))) - np.exp(-1.0 / (budget + 1)))) / np.sqrt(1.0 + 1.0 / (budget + 1)) - 1.0)
        
        # State
        best_x = m.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Loop
        while calls_made < budget:
            # Sample lambda points
            # Cholesky decomposition of C
            L = np.linalg.cholesky(C)
            
            # Generate standard normals
            Z = np.random.randn(lam, d)
            
            # Transform to search distribution
            # x = m + sigma * (L @ z.T).T
            X = m + sigma * (Z @ L.T)
            
            # Clip to bounds
            X = np.clip(X, -5.0, 5.0)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam
            
            # Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Update mean
            # Weights for the best mu
            w_mu = w[:mu]
            m_new = np.sum(X[:mu] * w_mu[:, np.newaxis], axis=0)
            
            # Update covariance
            # p_sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (L.T @ (m_new - m) / sigma)
            
            # p_c
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (m_new - m) / sigma
            
            # Update C
            # Delta term
            delta_hsig = (1 + hsig - cc) * cc * (1 - cc) * (np.dot(pc, pc) / d - 1)
            
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * np.sum(
                w_mu[:, np.newaxis, np.newaxis] * np.outer(X[:mu] - m, X[:mu] - m) / (sigma**2), axis=0
            )
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma *= np.exp(cs / d * (np.linalg.norm(ps) / np.sqrt(1 - np.exp(-2.0 * (calls_made / budget))) - np.sqrt(d - 1)))
            
            # Update state
            m = m_new
            C = C
            
            # Check if we ran out of budget mid-generation (shouldn't happen if lam divides budget, but safe)
            if calls_made >= budget:
                break
                
        return best_x, best_f
