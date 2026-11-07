"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using Differential Evolution with an Archive.
    
    Strategy:
    - Initialize a population using Latin Hypercube Sampling (LHS) for good initial spread.
    - Maintain a non-dominated archive of the best solutions found so far.
    - Iteratively generate new candidates using DE mutation (differential evolution) 
      where parents are selected from the current population and/or archive.
    - This allows the algorithm to exploit promising regions while the archive 
      maintains diversity, aiming to approximate the Pareto front more closely 
      than static sampling.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # 1. Initialize with LHS
        n_init = 50
        # Simple LHS: divide each dimension into n_init intervals, pick one random point per interval, permute
        indices = np.arange(n_init)
        # For each dimension, generate random points within its specific interval
        # x_j = (rand + j) / n_init
        u = np.random.rand(n_init, self.dim)
        # Shift each column by its index
        # We want the i-th point in the i-th bin for each dimension, but permuted across dimensions?
        # Standard LHS: for dim d, sort the random numbers? No, usually:
        # Generate (n, d) random numbers. For each dimension, assign the i-th smallest value to the i-th sample?
        # Or simply: x[i, d] = (rand[i, d] + i) / n
        # Let's use the simple version:
        unit_init = (np.random.rand(n_init, self.dim) + np.arange(n_init).reshape(-1, 1)) / n_init
        # This ensures each bin in each dimension is hit exactly once if we sort, but without sorting it's just stratified.
        # Actually, standard LHS permutes the bins. Let's do proper LHS.
        
        # Proper LHS
        u_lhs = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            # Create bins: 0/n, 1/n, ..., (n-1)/n
            bins = np.arange(n_init) / n_init
            # Random offset within bin
            offsets = np.random.rand(n_init) / n_init
            # Permute the bins to ensure coverage across the population
            perm = np.random.permutation(n_init)
            u_lhs[:, d] = bins[perm] + offsets[perm]
        
        X = lo + range_ * u_lhs
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X

        # Initialize Archive
        # Archive stores (x, f) pairs
        archive_X = []
        archive_F = []
        
        def is_dominated(a, b):
            # Returns True if a dominates b
            # a is better than b if a[i] <= b[i] for all i and a[j] < b[j] for some j
            if np.all(a <= b) and np.any(a < b):
                return True
            return False

        def update_archive(x, f):
            # Check if f is dominated by any in archive
            for af in archive_F:
                if is_dominated(af, f):
                    return # f is dominated, do not add
            
            # Remove any archive members dominated by f
            i = 0
            while i < len(archive_F):
                if is_dominated(f, archive_F[i]):
                    archive_X.pop(i)
                    archive_F.pop(i)
                else:
                    i += 1
            
            # Add f
            archive_X.append(x)
            archive_F.append(f)

        # Add initial points to archive
        for i in range(n_init):
            update_archive(X[i], F[i])
            
        # If archive is empty (should not happen), add best point
        if not archive_F:
             update_archive(X[0], F[0])

        # 2. Iterative DE
        # Population size for DE
        pop_size = 20
        # Select initial population from archive or random? 
        # Let's start with a mix of archive points and random to maintain diversity
        # But we must be careful with budget.
        
        # We have used n_init evaluations.
        # Remaining budget: self.budget - n_init
        remaining_budget = self.budget - n_init
        
        # DE parameters
        CR = 0.5 # Crossover rate
        F = 0.8  # Differential weight (using F as variable name conflicts with objective F, rename to F_de)
        F_de = 0.8
        
        # Current population for DE
        # Initialize DE population from archive (sampled) and some random points
        # To ensure we use budget effectively, we evaluate new points.
        
        # Let's maintain a 'population' of size pop_size.
        # Initially, populate it with points from the archive (randomly sampled with replacement if archive is small)
        # and some random points to ensure exploration.
        
        # However, the simplest robust approach:
        # Each iteration:
        # 1. Select parents from current pool (archive + recent non-dominated?)
        # 2. Generate offspring via DE mutation
        # 3. Evaluate offspring
        # 4. Update archive
        # 5. If time/budget runs out, stop.
        
        # To keep it simple and fast, we will just use the Archive as the source of parents.
        # This is similar to a greedy approach but with DE mutation.
        
        # Let's create a helper to get a random point from archive
        def get_random_archive_point():
            if not archive_X:
                return None
            idx = np.random.randint(len(archive_X))
            return archive_X[idx]
        
        # Main loop
        evals_made = n_init
        while evals_made < self.budget:
            if time.time() > deadline:
                break
            
            # Select 3 distinct parents from archive
            # If archive is small, we might pick same point. 
            # To avoid degenerate mutations, we can add random noise or use random points if archive is too small.
            if len(archive_X) < 3:
                # Fallback to random sampling if archive is too small to support DE
                x_new = lo + range_ * np.random.rand(self.dim)
            else:
                # Select 3 distinct indices
                # If len < 3, we can't pick 3 distinct. But we checked <3 above.
                # Wait, if len is 3, we pick 0,1,2.
                # If len > 3, we pick 3 random distinct.
                idxs = np.random.choice(len(archive_X), size=3, replace=False)
                x1 = archive_X[idxs[0]]
                x2 = archive_X[idxs[1]]
                x3 = archive_X[idxs[2]]
                
                # DE Mutation: v = x1 + F_de * (x2 - x3)
                v = x1 + F_de * (x2 - x3)
                
                # Crossover: uniform crossover between v and x1 (or a random base?)
                # Standard DE: base is x1? Or random? 
                # Let's use x1 as the base for the trial vector.
                # j_rand = random integer in [0, dim-1]
                j_rand = np.random.randint(self.dim)
                
                x_new = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        x_new[j] = v[j]
                    else:
                        x_new[j] = x1[j]
                
                # Clip to bounds
                x_new = np.clip(x_new, lo, hi)
            
            # Evaluate
            f_new = problem(x_new)
            evals_made += 1
            
            # Update Archive
            update_archive(x_new, f_new)
            
            # Optional: Periodically refresh the archive source? 
            # The archive itself is the source. As it grows, it represents the best front.
            # This is a form of "Archive-based DE".
            
            # To prevent stagnation, we can occasionally inject a random point.
            # Let's inject a random point every 10 iterations if archive hasn't grown significantly?
            # Or just rely on DE exploration.
            # Given the small budget (200), DE might converge too fast to a local region.
            # Let's mix in some random exploration.
            # If evals_made % 10 == 0 and len(archive_X) < 20:
            #    ...
            
            # Actually, a better strategy for WFG4 (which has a curved front) is to ensure diversity.
            # The archive does this.
            
            # To ensure we don't waste time, we break if we hit deadline.
            
        return archive_X if archive_X else X

# EVOLVE-BLOCK-END
