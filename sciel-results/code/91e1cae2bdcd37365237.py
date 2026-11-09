"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # CMA-ES parameters
        # For dim=5, lambda = 4 + floor(3*log(5+1)) = 4 + 5 = 9, let's use 10
        lam = 10
        mu = lam // 2  # number of parents
        
        # Initialize
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # CMA-ES parameters
        cc = (4 + dim / (dim + 4)) / ((dim + 1.7) + (4 + dim / (dim + 4)))
        cs = (1 + 2 * max(0, np.sqrt((lam / 2 - 1)) / (dim + 1) - 1)) / (dim + 3 + 2 * max(0, np.sqrt((lam / 2 - 1)) / (dim + 1) - 1))
        cmu = min(1 - cc, 2 * (lam / 2 - 2 + 1. / (lam / 2)) / ((dim + 2) ** 2 + lam / 2))
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
        
        calls = 0
        best_x = x
        best_f = float('inf')
        
        # Initial population
        pop = []
        fvals = []
        
        for i in range(lam):
            if time.time() > deadline:
                break
            xi = x + sigma * rng.multivariate_normal(np.zeros(dim), C)
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            pop.append(xi)
            fvals.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        if calls == 0:
            return best_x, best_f
        
        # Sort by fitness
        order = np.argsort(fvals)
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Update mean
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w ** 2)
            
            x_old = x.copy()
            x = sum(w[i] * pop[order[i]] for i in range(mu))
            
            # Update step size
            v = (x - x_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * v
            hsig = (np.sqrt(1 - np.exp(-2 * (calls / (lam * (dim + 1))) + 1e-10)) < (1 + cc) * chi_n) / (1 - np.exp(-2 * calls / (lam * (dim + 1))))
            
            pc = (1 - cc) * pc + hsig * cc * v
            sigma *= np.exp(cs / (dim + 1.7) * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            delta = hsig * (1 + cc) * (1 - cc)
            C = (1 + cmu * delta - cmu - cc * (1 - cc)) * C
            C += cmu * sum(w[i] * np.outer(pop[order[i]] - x_old, pop[order[i]] - x_old) for i in range(mu))
            C += cc * delta * np.outer(pc, pc)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Generate new population
            pop = []
            fvals = []
            
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                xi = x + sigma * rng.multivariate_normal(np.zeros(dim), C + 1e-10 * np.eye(dim))
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                pop.append(xi)
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
            
            if len(fvals) == 0:
                break
                
            # Sort by fitness
            order = np.argsort(fvals)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
