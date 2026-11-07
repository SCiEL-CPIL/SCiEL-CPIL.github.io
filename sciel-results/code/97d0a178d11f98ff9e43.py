import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        N = 20
        
        # Generate weight vectors
        # Using a grid on the simplex
        H = 5
        weights = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                weights.append([i / H, j / H, k / H])
        # We need N weights. If we have more, take first N. If less, repeat or adjust.
        # For H=5, we have 21 weights. We take first 20.
        if len(weights) < N:
            # Pad with random weights if necessary
            while len(weights) < N:
                w = np.random.dirichlet(np.ones(n_obj))
                weights.append(w)
        else:
            weights = weights[:N]
        weights = np.array(weights)
        
        # Initialize population
        X = lo + range_ * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Ideal point
        z_star = np.min(F, axis=0)
        
        # Define Tchebycheff function
        def tchebycheff(f, w, z):
            return np.max(w * np.abs(f - z))
        
        # Neighborhood definition
        # For simplicity, use a fixed neighborhood size or all if small
        # Standard MOEA/D uses a neighborhood T(i).
        # Let's define neighbors based on weight vector distance.
        # Or simply use a fixed index neighborhood if weights are ordered.
        # With grid weights, ordering is not trivial for neighborhood.
        # Let's use random selection of parents from the whole population for simplicity,
        # or define neighbors by index if we sort weights.
        # Actually, standard MOEA/D defines neighbors based on the weight vector index in a specific ordering.
        # Let's stick to a simpler variant: for each subproblem i, pick parents randomly from the population.
        # This is essentially a global search per subproblem, which might be less efficient but simpler.
        # A better approach: define neighbors as the K closest weight vectors.
        
        # Calculate pairwise distances between weights to find neighbors
        # K = 10 neighbors?
        K = 10
        neighbors = []
        for i in range(N):
            dists = np.sum((weights - weights[i]) ** 2, axis=1)
            # Exclude self
            dists[i] = np.inf
            # Get K closest
            n_idx = np.argsort(dists)[:K]
            neighbors.append(n_idx)
        
        # DE Parameters
        F_mut = 0.8
        CR = 0.5
        
        while used + N <= budget:
            # Generate trial solutions
            for i in range(N):
                # Select parents from neighbors of i
                # Ensure we pick 3 distinct parents from neighbors + i
                # If neighbors are not enough, add random
                cand_idx = list(neighbors[i])
                if i not in cand_idx:
                    cand_idx.append(i)
                
                # Shuffle and pick 3
                np.random.shuffle(cand_idx)
                # If we don't have 3 distinct, fill with random
                while len(cand_idx) < 3:
                    cand_idx.append(np.random.randint(N))
                
                r1, r2, r3 = cand_idx[0], cand_idx[1], cand_idx[2]
                
                # Mutation
                v = X[r1] + F_mut * (X[r2] - X[r3])
                
                # Crossover
                j_rand = np.random.randint(dim)
                trial = X[i].copy()
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                trial_f = problem(trial)
                used += 1
                
                # Update ideal point
                z_star = np.minimum(z_star, trial_f)
                
                # Update population
                # 1. Update subproblem i
                if tchebycheff(trial_f, weights[i], z_star) < tchebycheff(F[i], weights[i], z_star):
                    X[i] = trial
                    F[i] = trial_f
                
                # 2. Update neighbors
                for k in neighbors[i]:
                    if tchebycheff(trial_f, weights[k], z_star) < tchebycheff(F[k], weights[k], z_star):
                        X[k] = trial
                        F[k] = trial_f
        
        return X
