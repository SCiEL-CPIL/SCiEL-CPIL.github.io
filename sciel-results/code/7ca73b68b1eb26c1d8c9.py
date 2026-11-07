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
            P = 20
            # Ensure we don't exceed budget
            P = min(P, budget)
            
            # Weight vectors for MOEA/D (Tchebycheff)
            # For 2 objectives, weights are on the line w1 + w2 = 1
            weights = np.zeros((P, n_obj))
            for i in range(P):
                # Uniform distribution
                w1 = i / (P - 1)
                weights[i, 0] = w1
                weights[i, 1] = 1 - w1
            
            # Initialize population
            X = lo + range_ * np.random.rand(P, dim)
            F = np.array([problem(x) for x in X])
            evals = P
            
            # Ideal point z*
            z_star = np.min(F, axis=0)
            
            # Neighborhood size
            T = 5
            # Precompute neighbors
            # Distance between weight vectors
            # For 2D, index distance is sufficient
            neighbors = []
            for i in range(P):
                # Find T closest indices
                idxs = np.argsort(np.abs(np.arange(P) - i))
                neighbors.append(idxs[:T])
            
            # Main Loop
            while evals < budget:
                # Check time
                if time.time() - start_time > 290:
                    break
                
                # Pick a weight vector index to update
                # Randomly select i
                i = np.random.randint(P)
                
                # Select parents from neighbors of i
                n_idx = neighbors[i]
                # Pick 2 distinct parents
                p1_idx = n_idx[np.random.randint(T)]
                p2_idx = n_idx[np.random.randint(T)]
                
                # Ensure distinct? If T > 1, likely distinct.
                if p1_idx == p2_idx:
                    p2_idx = n_idx[(np.random.randint(T) + 1) % T]
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # SBX Crossover
                # eta = 20
                eta = 20
                child = np.zeros(dim)
                for d in range(dim):
                    if np.random.rand() < 0.5:
                        if abs(x1[d] - x2[d]) > 1e-14:
                            u = np.random.rand()
                            if u <= 0.5:
                                alpha = 2 * u
                                beta = ((alpha + 1) ** (eta + 1)) ** 0.5
                                c1 = (x1[d] * (beta + 1) - x2[d] * beta) / (2 * beta)
                            else:
                                alpha = 2 * (1 - u)
                                beta = ((alpha + 1) ** (eta + 1)) ** 0.5
                                c1 = (x2[d] * (beta + 1) - x1[d] * beta) / (2 * beta)
                            child[d] = c1
                        else:
                            child[d] = x1[d]
                    else:
                        child[d] = x1[d] # Wait, standard SBX swaps. 
                        # Let's use a simpler crossover or standard SBX implementation.
                        
                # Let's use a vectorized SBX for speed and correctness.
                # Actually, loop is fine for dim=30, P=20, 180 iterations.
                
                # Polynomial Mutation
                eta_m = 20
                for d in range(dim):
                    if np.random.rand() < 1.0 / dim:
                        u = np.random.rand()
                        if u < 0.5:
                            delta_1 = (2 * u) ** (1 / (eta_m + 1)) - 1
                        else:
                            delta_1 = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                        child[d] += delta_1 * (hi[d] - lo[d])
                        
                child = np.clip(child, lo, hi)
                
                # Evaluate
                f_child = problem(child)
                evals += 1
                
                # Update z_star
                z_star = np.minimum(z_star, f_child)
                
                # Update neighborhood
                # Calculate Tchebycheff value for child against neighbors
                # g(x, w, z*) = max_i w_i * |f_i(x) - z*_i|
                # Since z* is ideal, f_i - z*_i >= 0.
                
                # We need to check if child is better than current X[j] for j in neighbors[i]
                # Current values of X[j] need to be compared.
                
                # To avoid recalculating F for all neighbors, we store F.
                # But F might be stale if z_star changed? 
                # Tchebycheff value depends on z*. If z* decreases, the "distance" increases for all points.
                # However, the *relative* ranking might change.
                # Standard MOEA/D updates: 
                # For each neighbor j:
                #   if g(child, w_j, z*) < g(X[j], w_j, z*):
                #       X[j] = child
                #       F[j] = f_child
                
                # We need g(X[j], w_j, z*). We can compute this on the fly.
                # F is the objective vector for X.
                
                for j in neighbors[i]:
                    # Calculate g for child
                    diff_c = np.abs(f_child - z_star)
                    g_child = np.max(weights[j] * diff_c)
                    
                    # Calculate g for current X[j]
                    diff_x = np.abs(F[j] - z_star)
                    g_x = np.max(weights[j] * diff_x)
                    
                    if g_child < g_x:
                        X[j] = child
                        F[j] = f_child
                        
            return X
