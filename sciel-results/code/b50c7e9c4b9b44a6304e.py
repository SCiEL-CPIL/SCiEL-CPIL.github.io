import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # MOEA/D parameters
        M = 10  # Number of subproblems
        if self.budget < 2 * M:
            M = max(2, self.budget // 2)
            
        # Generate weight vectors (simplex-lattice or random on simplex)
        # For 3 objectives, we can use a grid or random points on the simplex
        # Simple random points on simplex
        weights = np.random.dirichlet(np.ones(n_obj), size=M)
        
        # Initialize population with LHS
        X = np.zeros((M, dim))
        for d in range(dim):
            bin_width = (hi[d] - lo[d]) / M
            perm = np.random.permutation(M)
            centers = lo[d] + (perm + 0.5) * bin_width
            X[:, d] = centers
            
        F = np.array([problem(x) for x in X])
        
        # Maintain a global archive of non-dominated points
        # We will store (x, f) pairs
        archive = []
        for i in range(M):
            x = X[i]
            f = F[i]
            # Check if dominated by archive
            dominated = False
            for _, f_arch in archive:
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    dominated = True
                    break
            if not dominated:
                # Remove dominated from archive
                new_archive = []
                for x_arch, f_arch in archive:
                    if not (np.all(f <= f_arch) and np.any(f < f_arch)):
                        new_archive.append((x_arch, f_arch))
                new_archive.append((x, f))
                archive = new_archive

        # MOEA/D Evolution
        # Calculate ideal point for Tchebycheff (approximate with min seen)
        ideal = np.min(F, axis=0)
        
        # Neighborhood
        # Simple: each subproblem is neighbor to all others? Or K nearest?
        # With M=10, let's just use all others as neighbors for simplicity, or K=4.
        # Let's use K=4 nearest neighbors based on weight vector distance.
        # Distance between weight vectors
        dist_matrix = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                dist_matrix[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        neighbors = []
        for i in range(M):
            # Get indices of 4 nearest neighbors (excluding self)
            sorted_idx = np.argsort(dist_matrix[i])
            # Exclude self (which is 0)
            nbrs = [idx for idx in sorted_idx if idx != i][:4]
            neighbors.append(nbrs)

        eta = 20.0  # Distribution index for SBX
        mu = 0.5    # Probability of SBX
        pm = 1.0 / dim  # Mutation probability
        dm = 20.0  # Distribution index for mutation

        def tchebycheff(f, w, ideal):
            # min max (w_i * (f_i - ideal_i))
            # Assuming ideal is the minimum found so far (lower bound)
            # If ideal is not known, use 0 or min of population.
            # Here we use dynamic ideal.
            return np.max(w * (f - ideal))

        def sbx_crossover(x1, x2):
            if np.random.rand() < mu:
                c1 = x1.copy()
                c2 = x2.copy()
                for d in range(dim):
                    if np.random.rand() < 0.5:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        elif u >= 0.5:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        else:
                            beta = 1.0
                        c1[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                        c2[d] = 0.5 * ((1 + 1 / beta) * x2[d] + (1 - 1 / beta) * x1[d])
                # Clip to bounds
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                return c1, c2
            else:
                return x1.copy(), x2.copy()

        def polynomial_mutate(x):
            for d in range(dim):
                if np.random.rand() < pm:
                    u = np.random.rand()
                    range_d = hi[d] - lo[d]
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (dm + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (dm + 1))
                    x[d] = x[d] + delta * range_d
            return np.clip(x, lo, hi)

        # Main loop
        while self.budget > 0 and time.time() < deadline:
            # Generate one offspring per subproblem
            for i in range(M):
                if self.budget <= 0:
                    break
                
                # Select parents from neighborhood
                nbrs = neighbors[i]
                if len(nbrs) == 0:
                    continue
                p1_idx = np.random.choice(nbrs)
                p2_idx = np.random.choice(nbrs)
                
                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                c1, c2 = sbx_crossover(x1, x2)
                c1 = polynomial_mutate(c1)
                
                f1 = problem(c1)
                self.budget -= 1
                
                # Update ideal point
                ideal = np.minimum(ideal, f1)
                
                # Update archive
                dominated = False
                for _, f_arch in archive:
                    if np.all(f_arch <= f1) and np.any(f_arch < f1):
                        dominated = True
                        break
                if not dominated:
                    new_archive = []
                    for x_arch, f_arch in archive:
                        if not (np.all(f1 <= f_arch) and np.any(f1 < f_arch)):
                            new_archive.append((x_arch, f_arch))
                    new_archive.append((c1, f1))
                    archive = new_archive

                # Update subproblem i if c1 is better
                # Check Tchebycheff value
                val_new = tchebycheff(f1, weights[i], ideal)
                val_old = tchebycheff(F[i], weights[i], ideal)
                
                # Also check domination
                if val_new < val_old or (np.all(f1 <= F[i]) and np.any(f1 < F[i])):
                    X[i] = c1
                    F[i] = f1
                    
                    # Update neighbors if better for them
                    for j in neighbors[i]:
                        val_new_j = tchebycheff(f1, weights[j], ideal)
                        val_old_j = tchebycheff(F[j], weights[j], ideal)
                        if val_new_j < val_old_j or (np.all(f1 <= F[j]) and np.any(f1 < F[j])):
                            X[j] = c1
                            F[j] = f1

        return archive
