"""Improved optimizer using CMA-ES with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with adaptive restarts for multi-modal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Initial exploration with stratified sampling
        # Use Latin Hypercube-style sampling for better coverage
        num_initial = min(20, budget // 15)
        candidates = []
        
        # Stratified sampling in each dimension
        for i in range(num_initial):
            x = np.zeros(dim)
            for d in range(dim):
                # Stratify each dimension
                stratum = (i + rng.random()) / num_initial
                x[d] = -5 + 10 * stratum
            fx = func(x)
            candidates.append((fx, x.copy()))
        
        candidates.sort(key=lambda t: t[0])
        used_budget = num_initial
        
        # Phase 2: CMA-ES from the best initial point
        if used_budget < budget:
            x0 = candidates[0][1].copy()
            f0 = candidates[0][0]
            
            # CMA-ES parameters
            lam = max(4, int(4 + 3 * np.log(dim)))  # population size
            sigma = 1.0
            mean = x0.copy()
            C = np.eye(dim)  # covariance matrix
            C_eigvals = np.ones(dim)  # eigenvalues of C
            C_eigvecs = np.eye(dim)  # eigenvectors of C
            ps = np.zeros(dim)  # evolution path for sigma
            pc = np.zeros(dim)  # evolution path for C
            cs = 1 / (lam + 3 + 2 / (dim + 1))  # step size control
            c1 = 2 / ((dim + 1.3) ** 2 + lam)  # rank-one update weight
            cmu = min(1 - c1, 2 * ((dim - 1 + 2 / (dim + 1)) ** 2 - 1) / 
                      ((dim + 3) ** 2 + lam))  # rank-mu update weight
            ds = 1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1)) - 1) + cs
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Track progress for restart detection
            best_in_run = f0
            no_improve_count = 0
            max_no_improve = lam * 5
            
            while used_budget < budget:
                # Generate offspring
                ys = []
                xs = []
                fs = []
                
                for k in range(min(lam, budget - used_budget)):
                    # Sample from N(0, C)
                    B = C_eigvecs @ np.diag(np.sqrt(C_eigvals))
                    y = rng.standard_normal(dim)
                    z = B @ y
                    x = np.clip(mean + sigma * z, -5, 5)
                    fx = func(x)
                    used_budget += 1
                    
                    ys.append(y)
                    xs.append(x.copy())
                    fs.append(fx)
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                
                if used_budget >= budget:
                    break
                
                # Sort by fitness
                order = np.argsort(fs)
                ys = [ys[i] for i in order]
                xs = [xs[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Update best in run
                if fs[0] < best_in_run - 1e-10:
                    best_in_run = fs[0]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Check for stagnation-based restart
                if no_improve_count > max_no_improve and used_budget < budget * 0.7:
                    # Restart with perturbation around best
                    sigma *= 0.5
                    mean = best_x.copy() if best_x is not None else x0.copy()
                    C = np.eye(dim)
                    C_eigvals = np.ones(dim)
                    C_eigvecs = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    no_improve_count = 0
                    continue
                
                # Update mean
                mu = lam // 2
                w = np.log(np.arange(1, mu + 1)) - np.mean(np.log(np.arange(1, mu + 1)))
                w = w / np.sum(w)
                mueff = np.sum(w) ** 2
                mean = np.sum([w[i] * xs[i] for i in range(mu)], axis=0)
                
                # Update evolution paths
                y_mean = (mean - mean) / sigma if sigma > 0 else np.zeros(dim)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_mean
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (mean - mean) / sigma if sigma > 0 else (1 - cs) * pc
                
                # Update covariance matrix
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (used_budget - num_initial + 1)))) / chi_n < 1.4 + 2 / (dim + 1)
                
                delta = (1 - hsig) * c1 * (2 - c1)
                C = (1 - c1 - cmu) * C + delta * np.outer(pc, pc)
                
                for i in range(mu):
                    y = (xs[i] - mean) / sigma if sigma > 0 else np.zeros(dim)
                    C += cmu * w[i] * np.outer(y, y)
                
                # Update sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)
                
                # Update eigendecomposition
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.maximum(C_eigvals, 1e-10)
                
                # Ensure C is positive definite
                C = C_eigvecs @ np.diag(C_eigvals) @ C_eigvecs.T
        
        # Phase 3: Final random sampling with remaining budget
        if used_budget < budget:
            # Use remaining budget for targeted random sampling around best
            while used_budget < budget:
                if best_x is not None:
                    # Sample near the best point
                    x = best_x + rng.standard_normal(dim) * 0.5
                else:
                    x = rng.uniform(-5, 5, dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                used_budget += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
