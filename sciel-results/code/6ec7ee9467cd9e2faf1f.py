import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20
        if self.budget < N:
            N = self.budget
            
        # Initialize with Latin Hypercube Sampling
        X = np.zeros((N, self.dim))
        for d in range(self.dim):
            edges = np.linspace(lo[d], hi[d], N + 1)
            for i in range(N):
                X[i, d] = np.random.uniform(edges[i], edges[i+1])
        
        F = np.array([problem(x) for x in X])
        used = N
        
        # MOEA/D Parameters
        # Generate weight vectors
        W = np.random.rand(N, self.n_obj)
        W = W / W.sum(axis=1, keepdims=True)
        
        # Find neighbors (T closest)
        T = max(2, N // 4)
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(W - W[i], axis=1)
            # Exclude self
            dists[i] = np.inf
            idxs = np.argsort(dists)
            neighbors.append(idxs[:T])
            
        # Ideal point
        ideal = F.min(axis=0)
        
        # Tchebycheff aggregation
        def tchebycheff(f, w, ideal_point):
            # f: (n_obj,)
            # w: (n_obj,)
            # ideal_point: (n_obj,)
            # g = max(w_k * |f_k - z_k|)
            return np.max(w * np.abs(f - ideal_point))
            
        # BLX-Alpha Crossover
        def blx_alpha_crossover(x1, x2, alpha=0.5):
            x1 = np.array(x1)
            x2 = np.array(x2)
            L = np.minimum(x1, x2) - alpha * (np.maximum(x1, x2) - np.minimum(x1, x2))
            U = np.maximum(x1, x2) + alpha * (np.maximum(x1, x2) - np.minimum(x1, x2))
            child = np.random.uniform(L, U)
            return child
            
        while used + N <= self.budget:
            for i in range(N):
                # Select parents from neighbors
                n_idx = np.random.choice(neighbors[i], size=2, replace=False)
                p1 = X[n_idx[0]]
                p2 = X[n_idx[1]]
                
                # Crossover
                y = blx_alpha_crossover(p1, p2, alpha=0.5)
                
                # Mutation (Gaussian)
                # Sigma should be small relative to range
                sigma = 0.1 * (hi - lo)
                y = y + np.random.normal(0, sigma)
                y = np.clip(y, lo, hi)
                
                # Evaluate
                f_y = problem(y)
                used += 1
                
                # Update ideal point
                ideal = np.minimum(ideal, f_y)
                
                # Compare with parents using Tchebycheff with weight W[i]
                # We want to replace the worse parent if y is better
                g_y = tchebycheff(f_y, W[i], ideal)
                
                # Check against p1
                f_p1 = F[n_idx[0]]
                g_p1 = tchebycheff(f_p1, W[i], ideal)
                
                # Check against p2
                f_p2 = F[n_idx[1]]
                g_p2 = tchebycheff(f_p2, W[i], ideal)
                
                # Replace strategy:
                # If y is better than p1, replace p1
                if g_y < g_p1:
                    X[n_idx[0]] = y
                    F[n_idx[0]] = f_y
                    # Also update ideal if it changed (already done above, but f_p1 might have been the min)
                    # Recalculate ideal? No, we updated ideal with f_y. 
                    # But if p1 was removed, ideal might need recalc if p1 was the unique min.
                    # However, ideal is non-increasing. If we remove a point that was the min, 
                    # the new min is >= old min. So ideal remains valid as a lower bound, 
                    # but might be too optimistic (lower than actual min of current pop).
                    # Tchebycheff with an optimistic ideal point is still a valid metric for comparison 
                    # as long as it's consistent.
                    
                # If y is better than p2, replace p2
                elif g_y < g_p2:
                    X[n_idx[1]] = y
                    F[n_idx[1]] = f_y
                    
        return X
