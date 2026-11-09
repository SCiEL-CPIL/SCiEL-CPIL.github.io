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
        N = 50
        T = 5 # Neighborhood size
        
        # 1. Generate Weight Vectors on Simplex
        weights = np.random.dirichlet(np.ones(self.n_obj), size=N)
        
        # 2. Precompute Neighborhoods
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        neighbors = []
        for i in range(N):
            sorted_indices = np.argsort(dists[i])
            neigh_list = [idx for idx in sorted_indices if idx != i][:T]
            neighbors.append(neigh_list)
        
        # 3. Initialize Population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # 4. Initialize Ideal Point
        z_star = np.min(F, axis=0)
        
        # Helper: Tchebycheff Scalarization
        def tchebycheff(f, w, z):
            return np.max(w * np.abs(f - z))
        
        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=20.0):
            child1 = np.copy(x1)
            child2 = np.copy(x2)
            mask = np.random.rand(self.dim) < 0.5
            for i in range(self.dim):
                if mask[i]:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1.0))
                    child1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    child2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta=20.0):
            x_mut = np.copy(x)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1.0))
                    x_mut[i] += delta * (hi[i] - lo[i])
            return np.clip(x_mut, lo, hi)

        # Main Loop
        while used < self.budget:
            for i in range(N):
                if used >= self.budget:
                    break
                
                # Select 2 parents from neighborhood
                neigh = neighbors[i]
                p1_idx = neigh[np.random.randint(len(neigh))]
                p2_idx = neigh[np.random.randint(len(neigh))]
                
                # Crossover and Mutation
                c1, _ = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                
                # Update Ideal Point
                z_star = np.minimum(z_star, f1)
                
                # Update Population
                # Check if offspring is better for subproblem i
                g_off_i = tchebycheff(f1, weights[i], z_star)
                g_curr_i = tchebycheff(F[i], weights[i], z_star)
                
                if g_off_i < g_curr_i:
                    X[i] = c1
                    F[i] = f1
                    
                # Check neighbors
                for j in neigh:
                    g_off_j = tchebycheff(f1, weights[j], z_star)
                    g_curr_j = tchebycheff(F[j], weights[j], z_star)
                    
                    if g_off_j < g_curr_j:
                        X[j] = c1
                        F[j] = f1
                        
        return X
