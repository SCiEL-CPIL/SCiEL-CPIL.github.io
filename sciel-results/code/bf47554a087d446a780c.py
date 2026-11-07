import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # MOEA/D parameters
        # Number of subproblems
        # With budget 200, we can afford ~100 subproblems if we do 1 update each? 
        # Or 50 subproblems and 2-3 updates each?
        # Let's use N=60 subproblems.
        N = 60
        
        # Generate weight vectors
        # Simplex-lattice method or random
        # For m=3, we can use a grid on the simplex.
        # Or just random Dirichlet(1,1,1)
        weights = rng.dirichlet(np.ones(m), size=N)
        
        # Initialize population
        X = lo + span * rng.random((N, n))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N
        
        # Tchebycheff distance
        def tchebycheff(f, w):
            return np.max(w * np.abs(f))
            
        # Initial Tchebycheff values
        Z = np.zeros(m)
        # Ideal point approximation (min of F)
        Z = F.min(axis=0)
        
        # Calculate initial distances
        dists = np.zeros(N)
        for i in range(N):
            dists[i] = tchebycheff(F[i], weights[i])
            
        # Find neighbors for each subproblem
        # Distance between weight vectors
        w_dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                w_dists[i, j] = np.linalg.norm(weights[i] - weights[j])
                
        # Sort neighbors for each i
        neighbor_indices = np.zeros((N, N), dtype=int)
        for i in range(N):
            neighbor_indices[i] = np.argsort(w_dists[i])
            
        T = 10 # Neighborhood size
        
        # Evolution loop
        # We have budget - N evaluations left.
        # Each evaluation produces 1 offspring (if we do 1:1 replacement) or 2?
        # Standard MOEA/D: 1 offspring per subproblem update.
        # If we update all N subproblems, that's N evaluations per generation.
        # Remaining budget: 200 - 60 = 140.
        # Generations: 140 // 60 = 2 full generations + 20 partial.
        
        while used < budget:
            # Shuffle order of subproblems to update
            order = rng.permutation(N)
            
            for i in order:
                if used >= budget:
                    break
                    
                # Select parents from neighbors
                # Pick 2 distinct parents from the T nearest neighbors
                # Ensure we don't pick the same one twice
                p1_idx = i
                # Pick p2 from neighbors
                # Simple: pick random from neighbors[0:T]
                # But we need 2 parents.
                # Let's pick p1 from neighbors, p2 from neighbors.
                # If p1 == p2, pick another.
                
                # To ensure diversity, maybe pick p1 = i (current best) and p2 = random neighbor?
                # Or random 2 from neighbors.
                cand1 = neighbor_indices[i, rng.integers(0, T)]
                cand2 = neighbor_indices[i, rng.integers(0, T)]
                if cand1 == cand2:
                    cand2 = neighbor_indices[i, (rng.integers(0, T) + 1) % T]
                    
                x1 = X[cand1]
                x2 = X[cand2]
                
                # Crossover: SBX or BLX
                # Let's use BLX-Alpha for simplicity and robustness in bounded spaces
                # Or SBX.
                # SBX is better.
                beta = 1.0
                # Simplified SBX
                # We can just use arithmetic crossover with noise for speed?
                # No, let's do proper SBX.
                
                # To save code size and complexity, let's use a simple effective crossover
                # Uniform crossover + Gaussian mutation?
                # Or BLX-Alpha:
                # a = min(x1, x2), b = max(x1, x2)
                # child = a + rng.uniform(-alpha, 1+alpha) * (b-a)
                
                alpha = 0.5
                child = np.zeros(n)
                for k in range(n):
                    a = min(x1[k], x2[k])
                    b = max(x1[k], x2[k])
                    if a == b:
                        child[k] = a
                    else:
                        child[k] = a + rng.uniform(-alpha, 1 + alpha) * (b - a)
                        
                # Mutation: Gaussian
                # sigma = 0.1
                child += rng.normal(0, 0.1, n)
                child = np.clip(child, lo, hi)
                
                # Evaluate
                f_child = problem(child)
                used += 1
                
                # Update
                # 1. Check against current subproblem i
                d_child = tchebycheff(f_child, weights[i])
                if d_child < dists[i]:
                    X[i] = child
                    F[i] = f_child
                    dists[i] = d_child
                    
                # 2. Update neighbors if better
                # Check T nearest neighbors (excluding i? or including?)
                # Standard: if it improves the fitness of a neighbor j, replace X[j]
                for j_idx in range(T):
                    j = neighbor_indices[i, j_idx]
                    if j == i:
                        continue
                    d_j = tchebycheff(f_child, weights[j])
                    if d_j < dists[j]:
                        X[j] = child
                        F[j] = f_child
                        dists[j] = d_j
                        
        return X
