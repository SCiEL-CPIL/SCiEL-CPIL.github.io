# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size for MOEA/D
        # With 200 budget, N=50 allows 4 generations. N=100 allows 2.
        # Let's try N=50 for more iterations.
        N = 50 
        
        # Generate weight vectors on the simplex
        # For 3 objectives, we can use a grid.
        # Or random uniform points on the simplex.
        # Systematic grid:
        # Number of divisions per dimension. 
        # For N=50, we can approximate a grid.
        # Simpler: Random uniform on simplex, then normalize.
        # But random might cluster.
        # Let's use a systematic approach for 3D.
        # We want ~50 points.
        # Layers: 
        # L=1: 1 point (1,0,0), (0,1,0), (0,0,1) -> 3
        # L=2: 6 points
        # ...
        # This is complex. Let's just use random uniform on simplex and deduplicate?
        # Or use a fixed set of directions.
        
        # Let's use a simpler decomposition: 
        # Assign each individual a random weight vector, but ensure diversity.
        # Actually, standard MOEA/D uses a systematic set.
        # Let's generate N random points on the simplex and use them.
        
        rng = np.random.RandomState()
        
        # Generate N points on the (m-1) simplex
        weights = np.zeros((N, self.n_obj))
        for i in range(N):
            # Generate random points and normalize
            while True:
                p = rng.rand(self.n_obj)
                s = p.sum()
                if s > 1e-6:
                    p = p / s
                    # Ensure they are distinct enough? 
                    # For now, just use them.
                    weights[i] = p
                    break
        
        # Initialize population
        X = lo + (hi - lo) * rng.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Ideal point (minimum of each objective)
        ideal = F.min(axis=0)
        
        # Neighbor list: find K nearest neighbors for each weight vector
        # K=20 is standard
        K = min(20, N-1)
        neighbors = []
        for i in range(N):
            dists = np.sum((weights - weights[i])**2, axis=1)
            # Exclude self
            dists[i] = np.inf
            neighbors.append(np.argsort(dists)[:K])
        
        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # For each individual, try to improve it
            # We process in random order to avoid bias
            order = rng.permutation(N)
            
            for i in order:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select 2 neighbors for DE
                # Pick 2 distinct neighbors from the neighbor list
                # Ensure they are different from each other and i
                # Simple: pick 2 random from neighbors[i]
                n1 = neighbors[i][rng.randint(K)]
                n2 = neighbors[i][rng.randint(K)]
                if n1 == n2:
                    n2 = neighbors[i][(rng.randint(K)+1)%K]
                    
                # Differential Evolution
                # v = x[n1] + F * (x[n2] - x[i])
                F_de = 0.5 + 0.5 * rng.rand()
                v = X[n1] + F_de * (X[n2] - X[i])
                v = np.clip(v, lo, hi)
                
                # Evaluate
                f_v = problem(v)
                used += 1
                
                # Update ideal point
                ideal = np.minimum(ideal, f_v)
                
                # Calculate Tchebycheff values for v
                # g(x) = max(w_i * |f_i(x) - ideal_i|)
                # Note: ideal changes, so we might need to recompute?
                # Standard MOEA/D updates ideal and uses it.
                
                # Check if v is better than x[i]
                # We need to compare g(v) and g(x[i]) using the CURRENT ideal?
                # Or the ideal at the time of creation?
                # Usually, we compute g for v and compare with g(x[i]) computed with same ideal.
                
                # Compute g for v
                g_v = np.max(weights[i] * np.abs(f_v - ideal))
                
                # Compute g for x[i] (using current ideal)
                # We don't store g for all, so we recompute or store.
                # Let's store F and recompute g when needed.
                g_i = np.max(weights[i] * np.abs(F[i] - ideal))
                
                if g_v < g_i:
                    X[i] = v
                    F[i] = f_v
                    
                    # Update neighbors if v is better for them
                    for j in neighbors[i]:
                        if j == i:
                            continue
                        g_j = np.max(weights[j] * np.abs(F[j] - ideal))
                        g_v_for_j = np.max(weights[j] * np.abs(f_v - ideal))
                        if g_v_for_j < g_j:
                            X[j] = v
                            F[j] = f_v

        # Return something (ignored)
        return F
# EVOLVE-BLOCK-END
