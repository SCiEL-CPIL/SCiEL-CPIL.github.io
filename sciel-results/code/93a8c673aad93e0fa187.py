"""Multi-objective optimizer for RE21 using Adaptive Random Search with Archive."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = [] # List of (x, f) tuples
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 300.0 # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Helper to check non-dominance
        def is_non_dominated(f_new, archive_f):
            if len(archive_f) == 0:
                return True
            # Check if any point in archive dominates f_new
            # a dominates b if a <= b and a != b
            for f_old in archive_f:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return False
            return True

        def add_to_archive(x, f):
            # Remove points dominated by f
            new_archive = []
            for x_old, f_old in self.archive:
                # If f_old is dominated by f, discard f_old
                if np.all(f <= f_old) and np.any(f < f_old):
                    continue
                new_archive.append((x_old, f_old))
            new_archive.append((x, f))
            self.archive = new_archive

        # Phase 1: Initialization
        # Use a mix of random and boundary-focused points
        n_init = min(20, self.budget)
        X_init = np.random.random((n_init, self.dim))
        # Add some boundary points
        if n_init < self.budget:
            # Sample a few points on the boundaries
            n_bound = min(5, self.budget - n_init)
            X_bound = np.zeros((n_bound, self.dim))
            for i in range(n_bound):
                # Randomly pick a dimension to be at a boundary
                dim_idx = np.random.randint(self.dim)
                X_bound[i] = np.random.random(self.dim)
                if np.random.rand() < 0.5:
                    X_bound[i, dim_idx] = 0.0
                else:
                    X_bound[i, dim_idx] = 1.0
            X_init = np.vstack([X_init, X_bound])
            n_init = len(X_init)
            
        # Clip to [0,1] then scale to bounds
        X_init = lo + span * X_init
        
        for x in X_init:
            if self.eval_count >= self.budget:
                break
            f = problem(x)
            self.eval_count += 1
            add_to_archive(x, f)
            
        # Phase 2: Adaptive Search
        sigma = 0.1 # Initial step size relative to bounds
        stagnation = 0
        max_stagnation = 10
        
        while self.eval_count < self.budget:
            # Time check
            if time.time() > deadline - 5.0:
                break
                
            # Select a parent from the archive
            # Prefer points that are on the Pareto front (all in archive are non-dominated)
            # To ensure diversity, we can pick randomly, or based on "potential"
            # Let's pick randomly from the archive to cover the front
            idx = np.random.randint(len(self.archive))
            parent_x, parent_f = self.archive[idx]
            
            # Generate child
            # Use adaptive sigma
            # If stagnation, increase sigma to explore more
            if stagnation > max_stagnation:
                sigma = min(0.5, sigma * 1.5)
                stagnation = 0
            # If we found a new point recently, keep sigma or slightly decrease
            # Actually, for hypervolume, maintaining a moderate sigma is key.
            
            child_x = parent_x + sigma * span * np.random.randn(self.dim)
            child_x = np.clip(child_x, lo, hi)
            
            f_child = problem(child_x)
            self.eval_count += 1
            
            # Check if child is non-dominated
            # We need to check against the current archive
            # Optimization: Only check against points that are close? 
            # For small archives, full check is fine.
            
            # Check if child is dominated by any point in archive
            dominated = False
            for _, f_old in self.archive:
                if np.all(f_old <= f_child) and np.any(f_old < f_child):
                    dominated = True
                    break
            
            if not dominated:
                # Add to archive (and remove dominated points)
                add_to_archive(child_x, f_child)
                stagnation = 0
            else:
                stagnation += 1
                # If dominated, maybe try a smaller step next time? 
                # Or just keep going.
                
        return self.archive
# EVOLVE-BLOCK-END
