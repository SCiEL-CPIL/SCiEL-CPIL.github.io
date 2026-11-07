"""Rewritten optimizer using a correct CMA-ES implementation."""
import numpy as np

class Optimizer:
    """CMA-ES continuous optimizer for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Number of parents
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial mean: center of the box
        mean = np.zeros(dim)
        
        # Initial standard deviation
        # BBOB functions are typically scaled. A step size of 2.0 is a reasonable start for [-5, 5]
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute Cholesky decomposition of C
        try:
            C_chol = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_chol = np.eye(dim)
            
        best_x = mean.copy()
        best_fx = float('inf')
        
        calls_made = 0
        
        # Initial evaluation
        f0 = func(mean)
        calls_made += 1
        best_fx = f0
        best_x = mean.copy()
        
        while calls_made < budget:
            # Generate population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # y = mean + sigma * (C_chol @ z)
                y = mean + sigma * (C_chol @ z)
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                pop.append(y)
                
            # Evaluate population
            fvals = []
            for y in pop:
                if calls_made >= budget:
                    break
                fy = func(y)
                fvals.append(fy)
                calls_made += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    
            if len(fvals) < lam:
                break
                
            fvals = np.array(fvals)
            order = np.argsort(fvals)
            pop_sorted = [pop[i] for i in order]
            fvals_sorted = fvals[order]
            
            # Update mean
            old_mean = mean.copy()
            mean = sum(w[i] * pop_sorted[i] for i in range(mu))
            
            # Update evolution paths
            # Inverse Cholesky of C for ps update
            try:
                C_inv_sqrt = np.linalg.inv(C_chol)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
                
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (mean - old_mean)) / sigma
            norm_ps = np.linalg.norm(ps)
            hsig = norm_ps / np.sqrt(1 - (1 - cs)**((calls_made // lam + 1) * cs)) < (1.4 + 2 / (dim + 1)) * (1 - cs)
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
            
            # Update covariance matrix
            diff = [pop_sorted[i] - old_mean for i in range(mu)]
            diff = np.array(diff)
            rank1 = np.outer(pc, pc) - C
            rankmu = sum(w[i] * np.outer(diff[i], diff[i]) for i in range(mu))
            
            C = (1 + c1 * (2 - c1) * (hsig - 1) - c1) * C + c1 * rank1 + cmu * rankmu
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            
            # Update Cholesky decomposition
            # Add a small regularization if needed
            try:
                C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                # If C becomes invalid, reset to diagonal
                C = np.diag(np.diag(C))
                C = (C + C.T) / 2
                C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                
            # Adapt step size
            sigma = sigma * np.exp(cs / (dim * sigma) * (norm_ps / np.sqrt(2 * dim) - 1))
            
            # Stop criteria
            if sigma < 1e-8 or np.allclose(mean, old_mean, atol=1e-8):
                break
                
        return best_x, best_fx
