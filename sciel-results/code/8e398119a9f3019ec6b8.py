import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        P = 50
        P = min(P, budget)
        
        # Weight vectors for MOEA/D (Tchebycheff)
        weights = np.zeros((P, n_obj))
        for i in range(P):
            w1 = i / (P - 1) if P > 1 else 0.5
            weights[i, 0] = w1
            weights[i, 1] = 1 - w1
        
        # Initialize population
        X = lo + range_ * np.random.rand(P, dim)
        F = np.array([problem(x) for x in X])
        evals = P
        
        # Ideal point z*
        z_star = np.min(F, axis=0)
        
        # Neighborhood size
        T = min(10, P)
        # Precompute neighbors
        neighbors = []
        for i in range(P):
            idxs = np.argsort(np.abs(np.arange(P) - i))
            neighbors.append(idxs[:T])
        
        # Main Loop
        while evals < budget:
            if time.time() - start_time > 290:
                break
                
            # Pick a weight vector index to update
            i = np.random.randint(P)
            
            # Select parents from neighbors of i
            n_idx = neighbors[i]
            p1_idx = n_idx[np.random.randint(T)]
            p2_idx = n_idx[np.random.randint(T)]
            if p1_idx == p2_idx:
                p2_idx = n_idx[(np.random.randint(T) + 1) % T]
            
            x1 = X[p1_idx]
            x2 = X[p2_idx]
            
            # Vectorized SBX Crossover
            eta = 20
            mask = np.random.rand(dim) < 0.5
            # For indices where mask is True, apply SBX
            # For indices where mask is False, take from x1 (or x2, standard SBX swaps)
            # Let's use standard SBX:
            # if rand < 0.5:
            #   if x1[d] == x2[d]: child[d] = x1[d]
            #   else:
            #     u = rand
            #     if u <= 0.5:
            #       alpha = 2*u
            #       beta = (alpha+1)^(eta+1)
            #       c1 = (x1*(beta+1) - x2*beta) / (2*beta)
            #     else:
            #       alpha = 2*(1-u)
            #       beta = (alpha+1)^(eta+1)
            #       c1 = (x2*(beta+1) - x1*beta) / (2*beta)
            
            child = x1.copy()
            if np.any(mask):
                x1_m = x1[mask]
                x2_m = x2[mask]
                diff = np.abs(x1_m - x2_m)
                # Handle case where x1 == x2
                valid = diff > 1e-14
                child_m = x1_m.copy()
                if np.any(valid):
                    x1_v = x1_m[valid]
                    x2_v = x2_m[valid]
                    u = np.random.rand(np.sum(valid))
                    alpha = np.where(u <= 0.5, 2*u, 2*(1-u))
                    beta = (alpha + 1) ** (eta + 1)
                    c1 = np.where(u <= 0.5, 
                                  (x1_v * (beta + 1) - x2_v * beta) / (2 * beta),
                                  (x2_v * (beta + 1) - x1_v * beta) / (2 * beta))
                    child_m[valid] = c1
                child[mask] = child_m
            
            # Polynomial Mutation
            eta_m = 20
            mut_mask = np.random.rand(dim) < 1.0 / dim
            if np.any(mut_mask):
                u = np.random.rand(np.sum(mut_mask))
                delta = np.where(u < 0.5,
                                 (2 * u) ** (1 / (eta_m + 1)) - 1,
                                 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
                child[mut_mask] += delta * range_[mut_mask]
                
            child = np.clip(child, lo, hi)
            
            # Evaluate
            f_child = problem(child)
            evals += 1
            
            # Update z_star
            z_star = np.minimum(z_star, f_child)
            
            # Update neighborhood
            for j in neighbors[i]:
                # Calculate Tchebycheff value for child
                diff_c = np.abs(f_child - z_star)
                g_child = np.max(weights[j] * diff_c)
                
                # Calculate Tchebycheff value for current X[j]
                diff_x = np.abs(F[j] - z_star)
                g_x = np.max(weights[j] * diff_x)
                
                if g_child < g_x:
                    X[j] = child
                    F[j] = f_child
                        
        return X
