import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize population
        pop_size = 50
        X = []
        F = []
        
        # Hybrid Init
        # 1/3 Random
        for _ in range(pop_size // 3):
            x = lo + (hi - lo) * np.random.random(self.dim)
            X.append(x)
            F.append(problem(x))
            
        # 1/3 Near 0 for tail
        for _ in range(pop_size // 3):
            x = np.random.random(self.dim)
            x[3:] = 0.0
            X.append(x)
            F.append(problem(x))
            
        # 1/3 Near 0.1 for tail (guess)
        for _ in range(pop_size - 2 * (pop_size // 3)):
            x = np.random.random(self.dim)
            x[3:] = 0.1
            X.append(x)
            F.append(problem(x))
            
        X = np.array(X)
        F = np.array(F)
        
        evals = len(X)
        
        # Archive of non-dominated points
        # We will maintain a list of (x, f)
        archive_x = []
        archive_f = []
        
        def is_dominated(f_new, archive_f):
            # f_new is dominated if exists f_a in archive such that f_a <= f_new and f_a < f_new
            if len(archive_f) == 0:
                return False
            # Vectorized check
            # archive_f shape (N, M), f_new shape (M,)
            # dominated if any row is <= f_new AND any element is < f_new
            leq = np.all(archive_f <= f_new, axis=1)
            lt = np.any(archive_f < f_new, axis=1)
            return np.any(leq & lt)
            
        def add_to_archive(x, f):
            # Check if x is dominated by archive
            if is_dominated(f, np.array(archive_f)):
                return False
            # Remove points in archive dominated by x
            new_archive_x = []
            new_archive_f = []
            for ax, af in zip(archive_x, archive_f):
                # Check if af is dominated by f
                # af dominated if f <= af and f < af
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_x.append(ax)
                    new_archive_f.append(af)
            new_archive_x.append(x)
            new_archive_f.append(f)
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        # Populate archive
        for x, f in zip(X, F):
            add_to_archive(x, f)
            
        # Evolution
        # Adaptive sigma
        max_sigma = 0.3
        min_sigma = 0.02
        
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Pick a parent from archive
            if not archive_x:
                # Fallback to random
                x_parent = lo + (hi - lo) * np.random.random(self.dim)
            else:
                idx = np.random.randint(len(archive_x))
                x_parent = archive_x[idx]
                
            # Adaptive sigma based on progress
            progress = evals / self.budget
            sigma = max_sigma * (1 - progress) + min_sigma * progress
            
            # Perturb
            x_child = x_parent + sigma * np.random.normal(size=self.dim)
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            evals += 1
            
            # Add to archive if non-dominated
            add_to_archive(x_child, f_child)
            
            # Optional: If archive is large, we could prune, but 200 is small.
            
        return archive_x
