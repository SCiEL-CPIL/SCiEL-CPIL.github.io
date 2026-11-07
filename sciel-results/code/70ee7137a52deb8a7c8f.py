import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # MOEA/D parameters
        N = 20  # Population size
        n_gen = self.budget // N
        if n_gen < 1:
            N = self.budget
            n_gen = 1
            
        # Generate weight vectors
        # For 2 objectives, lambda1 from 0 to 1
        if self.n_obj == 2:
            lambdas = np.linspace(0, 1, N)
            lambda_vectors = np.column_stack([lambdas, 1 - lambdas])
        else:
            # Fallback for other n_obj (though task says 2)
            lambda_vectors = np.random.dirichlet(np.ones(self.n_obj), size=N)

        # Neighborhood size
        T = 5
        # Build neighborhood based on weight vector distance
        neighbors = []
        for i in range(N):
            # Euclidean distance in weight space
            dists = np.linalg.norm(lambda_vectors - lambda_vectors[i], axis=1)
            sorted_idx = np.argsort(dists)
            nbrs = sorted_idx[:T]
            neighbors.append(nbrs)

        # Initialize population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        
        # Track min and max for normalization
        min_f = np.min(F, axis=0)
        max_f = np.max(F, axis=0)
        
        # Tchebycheff distance function
        def tcheb_dist(f, w, min_f, max_f):
            range_f = max_f - min_f
            range_f[range_f == 0] = 1.0
            f_norm = (f - min_f) / range_f
            return np.max(w * f_norm)

        # Main loop
        for g in range(n_gen):
            # Update normalization bounds
            min_f = np.min(F, axis=0)
            max_f = np.max(F, axis=0)
            
            for i in range(N):
                # Select parents from neighborhood of i
                nbrs = neighbors[i]
                # Pick 2 distinct parents
                p1_idx, p2_idx = np.random.choice(nbrs, 2, replace=False)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover: Blending
                alpha = np.random.rand(self.dim)
                child = alpha * p1 + (1 - alpha) * p2
                
                # Mutation: Gaussian
                sigma = 0.1
                child = child + sigma * (hi - lo) * np.random.randn(self.dim)
                child = np.clip(child, lo, hi)
                
                # Evaluate
                f_child = problem(child)
                
                # Update individual i
                w_i = lambda_vectors[i]
                g_child = tcheb_dist(f_child, w_i, min_f, max_f)
                g_i = tcheb_dist(F[i], w_i, min_f, max_f)
                
                if g_child < g_i:
                    F[i] = f_child
                    X[i] = child
                    # Update neighbors
                    for j in nbrs:
                        if j == i:
                            continue
                        w_j = lambda_vectors[j]
                        g_child_j = tcheb_dist(f_child, w_j, min_f, max_f)
                        g_j = tcheb_dist(F[j], w_j, min_f, max_f)
                        if g_child_j < g_j:
                            F[j] = f_child
                            X[j] = child

        return X
