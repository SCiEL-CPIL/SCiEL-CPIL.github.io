import numpy as np

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = max(4, 4 + int(3 * np.log(n))) 
        if lam > 20: lam = 20
        mu = lam // 2
        
        # Weights
        weights = np.log(np.arange(mu, 0, -1)) - np.log(mu)
        weights /= np.sum(weights)
        
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4.0 + lam / mueff - 2.0) / (n + lam / mueff + 2.0)
        c1 = 2.0 / ((n + 1.3)**2 + lam / mueff)
        cmu = 2.0 * ((0.5 + lam / mueff) / ((n + 2.0)**2) - c1)
        if cmu < 0: cmu = 0
        cs = (4.0 + lam / mueff - 2.0) / (n + 6.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((lam / mueff - 1.0) / (n + 1.0)) - 1.0) + cs
        
        # Initialization
        # Start at origin, which is a good prior for [-5,5] box with shift in [-4,4]
        m = np.zeros(n)
        sigma = 2.0  # Initial step size
        
        # Covariance matrix and adaptation vectors
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Cholesky decomposition of C
        cholC = np.linalg.cholesky(C)
        
        # Track best point
        x_best = m
        f_best = func(m)
        
        # Calculate number of generations
        # We have 'budget' calls. We used 1 for initialization.
        # Each generation uses 'lam' calls.
        remaining = budget - 1
        n_gens = remaining // lam
        leftover = remaining % lam
        
        for gen in range(n_gens):
            # 1. Sample
            # Generate standard normal matrix (lam x n)
            z = np.random.randn(lam, n)
            # Transform to normal with mean m and cov sigma^2 C
            # x = m + sigma * z @ cholC.T
            x = m + sigma * (z @ cholC.T)
            
            # Clip to bounds
            x = np.clip(x, -5.0, 5.0)
            
            # 2. Evaluate
            fvals = np.array([func(xi) for xi in x])
            
            # Update global best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < f_best:
                f_best = fvals[idx_best]
                x_best = x[idx_best].copy()
            
            # 3. Sort
            idx_sorted = np.argsort(fvals)
            x_sorted = x[idx_sorted]
            fvals_sorted = fvals[idx_sorted]
            
            # 4. Update Mean
            # Weighted mean of the best mu individuals
            w = weights
            m_new = np.sum(w[:, None] * x_sorted[:mu], axis=0)
            
            # 5. Update Step Size (Sigma)
            # Calculate y_c = (m_new - m) / sigma
            yc = (m_new - m) / sigma
            
            # Update ps
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * yc
            
            # Calculate hsig (indicator for whether the step is correlated)
            # norm(ps) / sqrt(1 - (1-cs)^{2(gen+1)})
            # Approximation: norm(ps) / sqrt(n)
            # More accurate:
            # hsig = 1.0 if norm(ps) / sqrt(1 - (1-cs)^(2*(gen+1))) < (1.4 + 2/(n+1)) * sqrt((n+1)/(n-1)) else 0.0
            # For simplicity and robustness with small budget, we can use a simpler check or standard formula.
            # Let's use the standard formula.
            norm_ps = np.linalg.norm(ps)
            # We need to track the denominator term. 
            # denom = sqrt(1 - (1-cs)^(2*(gen+1)))
            # To avoid accumulating error, we can just use the standard heuristic or compute it.
            # Given the small number of generations, computing it is fine.
            denom = np.sqrt(1.0 - (1.0 - cs)**(2 * (gen + 1)))
            hsig = 1.0 if norm_ps / denom < (1.4 + 2.0 / (n + 1.0)) * np.sqrt((n + 1.0) / (n - 1.0)) else 0.0
            
            # Update pc
            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * yc
            
            # Update Sigma
            # sigma *= exp(cs / ds * (norm(ps) / sqrt(n) - 1.0))
            sigma = sigma * np.exp(cs / ds * (norm_ps / np.sqrt(n) - 1.0))
            
            # 6. Update Covariance Matrix
            # C = (1 - c1 - cmu) C + c1 pc pc^T + cmu sum w_i y_i y_i^T
            # y_i = (x_i - m) / sigma
            
            # Calculate rank-one update term: pc pc^T
            # Note: pc is already scaled in the update step? 
            # In standard CMA-ES:
            # pc = (1-cc)pc + cc * sqrt(mueff) * (m_new - m)/sigma
            # Here I used hsig inside. Let's stick to the standard update.
            # Actually, the formula is:
            # pc = (1-cc)pc + hsig * sqrt(cc(2-cc)mueff) * yc
            # Then C update uses pc.
            
            # Rank-mu update term
            y_mu = x_sorted[:mu]
            y_mu = (y_mu - m) / sigma
            # Weighted sum of outer products
            # sum_i w_i * y_i y_i^T
            # This can be computed as (y_mu.T * w) @ y_mu
            rank_mu_update = (y_mu.T * w) @ y_mu
            
            # Combine updates
            # We need to ensure positive definiteness.
            # C_new = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * rank_mu_update
            
            # To maintain positive definiteness, we often use a modification if eigenvalues become negative.
            # But for small n and stable runs, direct update is usually okay.
            # Let's compute the new C.
            C_new = (1.0 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * rank_mu_update
            
            # Check for positive definiteness and fix if necessary
            # Compute eigenvalues
            eigvals, eigvecs = np.linalg.eigh(C_new)
            # If any eigenvalue is too small or negative, clip it
            min_eig = 1e-12
            if np.any(eigvals < min_eig):
                # Reconstruct C with clipped eigenvalues
                eigvals_clipped = np.clip(eigvals, min_eig, None)
                C_new = (eigvecs * eigvals_clipped) @ eigvecs.T
                # Symmetrize
                C_new = (C_new + C_new.T) / 2.0
            
            C = C_new
            m = m_new
            
            # Update Cholesky
            try:
                cholC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add small identity
                C = C + 1e-8 * np.eye(n)
                cholC = np.linalg.cholesky(C)
        
        # Use remaining budget for a local refinement or just extra samples?
        # If leftover > 0, we can do a simple (1+1) or just random sampling around best.
        # Given AOCC rewards early progress, and we've used most budget, 
        # a few more samples might not hurt.
        if leftover > 0:
            for _ in range(leftover):
                # Sample from current distribution
                z = np.random.randn(1, n)
                x_new = m + sigma * (z @ cholC.T)
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = func(x_new[0])
                if f_new < f_best:
                    f_best = f_new
                    x_best = x_new[0].copy()
        
        return x_best, f_best
