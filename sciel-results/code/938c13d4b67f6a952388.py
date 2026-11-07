"""Rewritten multi-objective optimizer for WFG4 using MOEA/D."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 100
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N
        
        # Ideal point z* (lower is better)
        z_star = np.min(F, axis=0)
        
        # Generate weight vectors
        lambdas = self._generate_weight_vectors(N, n_obj, rng)
        
        # Find neighbors for each weight vector
        # Distance based on Euclidean distance between weight vectors
        distances = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                distances[i, j] = np.linalg.norm(lambdas[i] - lambdas[j])
        
        # Sort neighbors for each i
        T = 20  # Number of neighbors
        neighbors = np.zeros((N, T), dtype=int)
        for i in range(N):
            sorted_idx = np.argsort(distances[i])
            neighbors[i] = sorted_idx[:T]
        
        # Parameters
        CR = 0.5  # Crossover rate for DE
        F_mut = 0.8  # Mutation factor for DE
        
        # Main loop
        # We have budget - used evaluations left
        while used + N <= budget:
            for i in range(N):
                # Select 3 distinct neighbors for mutation
                r1, r2, r3 = rng.choice(neighbors[i], size=3, replace=False)
                
                # DE Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + F_mut * (X[r2] - X[r3])
                
                # DE Crossover
                y = np.empty_like(X[i])
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        y[j] = v[j]
                    else:
                        y[j] = X[i][j]
                    
                # Clip to bounds
                y = np.clip(y, lo, hi)
                
                # Evaluate
                f_y = np.array(problem(y), dtype=float)
                used += 1
                
                # Update z_star
                z_star = np.minimum(z_star, f_y)
                
                # Update population
                # Check if y is better than x_i for weight lambda_i
                g_y_i = self._tchebycheff(f_y, lambdas[i], z_star)
                g_x_i = self._tchebycheff(F[i], lambdas[i], z_star)
                
                if g_y_i <= g_x_i:
                    X[i] = y
                    F[i] = f_y
                else:
                    # Check neighbors
                    for j in neighbors[i]:
                        if j == i:
                            continue
                        g_y_j = self._tchebycheff(f_y, lambdas[j], z_star)
                        g_x_j = self._tchebycheff(F[j], lambdas[j], z_star)
                        if g_y_j <= g_x_j:
                            X[j] = y
                            F[j] = f_y
                            
        return X

    def _tchebycheff(self, f, lambda_vec, z_star):
        return np.max(lambda_vec * np.abs(f - z_star))

    def _generate_weight_vectors(self, N, n_obj, rng):
        # Generate N points uniformly on the simplex
        weights = np.zeros((N, n_obj))
        for _ in range(N):
            r = rng.random(n_obj)
            weights[_] = r / np.sum(r)
        return weights
# EVOLVE-BLOCK-END
