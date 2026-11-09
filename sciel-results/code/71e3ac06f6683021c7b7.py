import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        # CMA-ES parameters
        # lambda is population size
        # For n=5, lambda=10 is reasonable
        lam = 10
        # mu is number of parents
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cs = (mueff + 2) / (lam + mueff + 5)
        c1 = 2 / (lam**2 + 18)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / (lam**2 + 18))
        cc = 4 / (lam + 4)
        
        # Initialization
        # Start at center of the box
        m = np.zeros(n)
        # Initial step size. Box is [-5, 5], so width 10.
        # Sigma = 2.5 covers roughly 2 standard deviations to the edge.
        sigma = 2.5
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Pre-compute D = sqrt(Tr(C))
        D = np.sqrt(np.trace(C))
        
        evals = 0
        best_x = m.copy()
        best_f = np.inf
        
        # We will run generations until budget is exhausted
        # Each generation uses 'lam' evaluations
        
        while evals + lam <= self.budget:
            # Sample population
            Y = np.random.multivariate_normal(np.zeros(n), C, size=lam)
            X = m + sigma * Y
            
            # Evaluate
            Fs = np.array([func(x) for x in X])
            evals += lam
            
            # Update best
            idx_min = np.argmin(Fs)
            if Fs[idx_min] < best_f:
                best_f = Fs[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort
            order = np.argsort(Fs)
            X = X[order]
            Fs = Fs[order]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
            
            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D * (C @ (m - m_old) / sigma)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (n + 1)) * np.sqrt(lam + 1 - 2):
                sigma *= np.exp(cs / (n + 2) * (np.linalg.norm(ps) / D - 1))
                
            # Update covariance
            pc = (1 - cc) * pc + cc / D * (m - m_old) / sigma
            C = (1 + c1 * (cmu + c1 - 1) - c1) * C + c1 * np.outer(pc, pc) + \
                cmu * np.sum(weights[:, np.newaxis] * np.outer(X[:mu] - m, X[:mu] - m) / sigma**2, axis=0)
                
            # Symmetrize
            C = (C + C.T) / 2
            
            # Check for convergence or if we are out of budget
            # Simple convergence check: if sigma is very small or step is small
            if sigma < 1e-8:
                break
                
        # If we have remaining budget, we can do a few more single evaluations or finish
        # CMA-ES is usually done with full populations.
        # If budget is not multiple of lam, we might leave some unused.
        # To use them, we could do a local refinement or just return.
        # Given AOCC rewards early progress, and CMA-ES converges fast, 
        # the remaining few evals (if any) are less critical.
        
        return best_x, best_f
