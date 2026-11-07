import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        N = 20
        if self.budget < N:
            X = lo + rng.random((self.budget, self.dim)) * span
            F = np.array([problem(x) for x in X])
            return X

        G = (self.budget - N) // N
        
        # Initialize population
        X = lo + rng.random((N, self.dim)) * span
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Ideal point estimate
        z_star = np.min(F, axis=0)
        
        # Generate weight vectors
        # Simple random weights normalized to sum to 1
        W = rng.random((N, self.n_obj))
        W = W / W.sum(axis=1, keepdims=True)
        
        # Find neighbors
        T = 5
        neighbors = np.zeros((N, T), dtype=int)
        for i in range(N):
            dists = np.linalg.norm(W - W[i], axis=1)
            # Exclude self
            dists[i] = np.inf
            neighbors[i] = np.argsort(dists)[:T]
            
        F_diff = 0.5
        CR = 0.5
        
        for gen in range(G):
            for i in range(N):
                # Select parents from neighbors
                # r1, r2, r3 distinct from neighbors[i]
                # Ensure distinct indices
                while True:
                    r1 = neighbors[i, rng.integers(T)]
                    r2 = neighbors[i, rng.integers(T)]
                    r3 = neighbors[i, rng.integers(T)]
                    if r1 != r2 and r2 != r3 and r1 != r3:
                        break
                        
                # Mutation
                v = X[r1] + F_diff * (X[r2] - X[r3])
                
                # Crossover
                j_rand = rng.integers(self.dim)
                mask = rng.random(self.dim) < CR
                mask[j_rand] = True
                
                u = np.where(mask, v, X[i])
                u = np.clip(u, lo, hi)
                
                # Evaluate
                f_u = problem(u)
                evals += 1
                
                # Update ideal point
                z_star = np.minimum(z_star, f_u)
                
                # Tchebycheff scalarization
                # g(x, lambda) = max_j (lambda_j * |f_j(x) - z_j*|)
                # Since we minimize, and f >= z_star, |f - z_star| = f - z_star
                # So g(x, lambda) = max_j (lambda_j * (f_j(x) - z_j*))
                
                g_u = np.max(W[i] * (f_u - z_star))
                g_i = np.max(W[i] * (F[i] - z_star))
                
                if g_u < g_i:
                    X[i] = u
                    F[i] = f_u
                    
        return X
