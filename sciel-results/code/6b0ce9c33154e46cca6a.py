import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        if lam > budget // 2: # Ensure at least 2 generations
            lam = max(2, budget // 2)
            
        # Number of parents
        mu = lam // 2
        
        # Weights
        # Rank-based weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        hs = np.sqrt(1 + 1.5 / (dim + 1))
        
        # Covariance matrix adaptation parameters
        c1 = (2 - mueff / (dim + 1)**2) / ((dim + 2)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        cs = (mueff + 1.5) / (dim + mueff + 5)
        
        # Initial values
        x = np.zeros(dim) # Start at center
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-compute Cholesky of C
        L = np.linalg.cholesky(C)
        
        # Evaluation counter
        calls = 0
        best_x = x.copy()
        best_f = func(x)
        calls += 1
        
        if calls >= budget:
            return best_x, best_f

        # Main Loop
        while calls < budget:
            if time.time() >= deadline:
                break
                
            # Generate population
            population = []
            for i in range(lam):
                z = np.dot(B, D * np.dot(L, np.random.standard_normal(dim)))
                # Note: Standard CMA-ES uses z = B * (D * (L * N(0,1)))
                # Actually, the sampling is x = m + sigma * z, where z ~ N(0, C)
                # z = B * D * (L * n) where n ~ N(0, I)?
                # No, C = B * D^2 * B^T.
                # So z = B * D * y, where y ~ N(0, I).
                # Wait, the standard decomposition is C = B D^2 B^T.
                # Then z = B D y, y ~ N(0, I).
                # My previous line was mixing Cholesky and eigen.
                # Let's stick to the eigen decomposition update which is standard.
                
                # Re-calculate z correctly
                y = np.random.standard_normal(dim)
                z = np.dot(B, D * y)
                
                xi = x + sigma * z
                # Clip to box
                xi = np.clip(xi, -5, 5)
                population.append(xi)
            
            # Evaluate population
            f_values = []
            for xi in population:
                if calls >= budget:
                    break
                fi = func(xi)
                calls += 1
                f_values.append(fi)
                
            if len(f_values) < lam:
                # If we ran out of budget mid-generation, pad with inf or just break
                # It's better to break and return best so far
                break
            
            # Sort by fitness
            idx = np.argsort(f_values)
            sorted_population = [population[i] for i in idx]
            sorted_f = [f_values[i] for i in idx]
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_population[0].copy()
            
            # Update mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * sorted_population[i]
            
            # Update step size (CSA)
            # ps = (1-cs)*ps + sqrt(cs*(2-cs))*B^T*(x_new-x)/sigma
            # Note: B^T is inverse of B since B is orthogonal
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma)) * np.dot(B.T, (x_new - x) / sigma)
            
            # Update pc
            # pc = (1-cs)*pc + hsig*sqrt(cs*(2-cs))*B^T*(x_new-x)/sigma
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - lam))) < (1 + 2 / (dim + 1)) * hs
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs)) * np.dot(B.T, (x_new - x) / sigma)
            
            # Update Covariance Matrix
            # C = (1-c1-cmu)*C + c1*(pc*pc^T + (1-hsig)*cs*(2-cs)*C) + cmu*sum(wi*(yi-x)(yi-x)^T)/sigma^2
            rank_one = np.outer(pc, pc)
            if not hsig:
                rank_one += (1 - hsig) * cs * (2 - c_sigma) * C
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = (sorted_population[i] - x) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Eigen decomposition for next iteration
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid numerical issues
            eigvals = np.clip(eigvals, 1e-20, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            x = x_new
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / hs - 1))
            
            # Safety check for sigma
            if sigma < 1e-10:
                break

        return best_x, best_f
