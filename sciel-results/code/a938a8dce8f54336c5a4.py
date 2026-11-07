"""Hypervolume-based Multi-Objective Evolutionary Algorithm for WFG7."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Multi-objective optimizer using MOEA/D (Decomposition-based) with Differential Evolution.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 50
        remaining = self.budget
        n_gen = max(1, remaining // N)
        
        # Generate weight vectors for MOEA/D
        # Random points on the simplex
        raw_weights = rng.random((N, self.n_obj))
        weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
        
        # Neighborhood structure
        K = 10 # Neighborhood size
        neighbors = np.zeros((N, K), dtype=int)
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            dists[i] = np.inf
            neighbors[i] = np.argsort(dists)[:K]
        
        # Initialize population
        X = lo + rng.random((N, self.dim)) * (hi - lo)
        F = np.array([problem(x) for x in X])
        remaining -= N
        
        # Tchebycheff aggregation function
        z_ideal = np.min(F, axis=0)
        
        def tchebycheff(f, w, z_ideal):
            return np.max(w * np.abs(f - z_ideal))
        
        # Evolutionary loop
        for gen in range(n_gen):
            if remaining <= 0:
                break
            
            # Generate offspring for each subproblem
            n_offspring = min(N, remaining)
            X_new = np.zeros((N, self.dim))
            F_new = np.zeros((N, self.n_obj))
            
            # DE parameters
            F_de = 0.8
            CR = 0.5
            
            for i in range(N):
                if remaining <= 0:
                    break
                    
                # Select a neighbor
                if rng.random() < 0.9:
                    n_idx = neighbors[i, rng.integers(K)]
                else:
                    n_idx = rng.integers(N)
                
                # Select 2 more parents from the whole population
                r2 = rng.integers(N)
                r3 = rng.integers(N)
                
                x_r1 = X[n_idx]
                x_r2 = X[r2]
                x_r3 = X[r3]
                
                # Mutation
                v = x_r1 + F_de * (x_r2 - x_r3)
                
                # Crossover
                u = np.zeros(self.dim)
                j_rand = rng.integers(self.dim)
                for d in range(self.dim):
                    if rng.random() < CR or d == j_rand:
                        u[d] = v[d]
                    else:
                        u[d] = X[i, d]
                
                # Clip
                u = np.clip(u, lo, hi)
                
                X_new[i] = u
                F_new[i] = problem(u)
                remaining -= 1
                
            # Update population using Tchebycheff
            # Update z_ideal with new points
            z_ideal = np.min(np.vstack([F, F_new]), axis=0)
            
            for i in range(N):
                if remaining < 0: break # Safety, though remaining should be >=0 here
                g_old = tchebycheff(F[i], weights[i], z_ideal)
                g_new = tchebycheff(F_new[i], weights[i], z_ideal)
                
                if g_new < g_old:
                    X[i] = X_new[i]
                    F[i] = F_new[i]
        
        # Return the non-dominated set
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        nd = np.ones(N, dtype=bool)
        for i in range(N):
            if not nd[i]:
                continue
            for j in range(N):
                if i == j or not nd[j]:
                    continue
                if dominates(F[i], F[j]):
                    nd[j] = False
        nd_idx = np.where(nd)[0]
        return [X[i] for i in nd_idx]
# EVOLVE-BLOCK-END
