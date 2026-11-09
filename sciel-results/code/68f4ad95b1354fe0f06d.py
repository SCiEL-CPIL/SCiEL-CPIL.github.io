import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Initial population size
        init_size = 20
        X = []
        F = []
        
        # Latin Hypercube Sampling for initialization
        for i in range(init_size):
            x = np.array([rng.uniform(lo[j], hi[j]) for j in range(dim)])
            X.append(x)
            F.append(np.array(problem(x)))
            
        used = init_size
        
        # Archive of non-dominated points
        archive = list(zip(X, F))
        
        # Function to check if a point is dominated by any in a set
        def is_dominated(point, set_points):
            for p in set_points:
                if np.all(p <= point) and np.any(p < point):
                    return True
            return False
            
        # Function to calculate crowding distance for a point in a set
        def crowding_distance(point, set_points):
            if len(set_points) <= 2:
                return float('inf')
            dist = 0
            for k in range(n_obj):
                min_val = min(p[k] for p in set_points)
                max_val = max(p[k] for p in set_points)
                if max_val - min_val == 0:
                    continue
                dist += (point[k] - min_val) / (max_val - min_val)
            return dist / n_obj

        # Adaptive step size
        step_size = 0.5 * np.max(hi - lo)
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select a parent from the archive
            # Use tournament selection based on crowding distance
            if len(archive) > 1:
                # Select 2 random points from archive
                p1, p2 = rng.choice(len(archive), size=2, replace=False)
                # Calculate crowding distance for both
                cd1 = crowding_distance(archive[p1][1], [a[1] for a in archive])
                cd2 = crowding_distance(archive[p2][1], [a[1] for a in archive])
                parent_idx = p1 if cd1 > cd2 else p2
            else:
                parent_idx = 0
                
            parent = archive[parent_idx][0]
            
            # Mutate
            child = parent + step_size * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate
            child_f = np.array(problem(child))
            used += 1
            
            # Add to main lists
            X.append(child)
            F.append(child_f)
            
            # Update archive
            # Check if child is dominated by any in archive
            dominated = is_dominated(child_f, [a[1] for a in archive])
            if not dominated:
                # Remove any points in archive dominated by child
                new_archive = []
                for a in archive:
                    if not (np.all(child_f <= a[1]) and np.any(child_f < a[1])):
                        new_archive.append(a)
                archive = new_archive
                archive.append((child, child_f))
                
                # If archive is too large, prune
                max_archive_size = 50
                if len(archive) > max_archive_size:
                    # Calculate crowding distance for all points in archive
                    cds = [crowding_distance(a[1], [b[1] for b in archive]) for a in archive]
                    # Remove the point with the lowest crowding distance
                    min_cd_idx = np.argmin(cds)
                    archive.pop(min_cd_idx)
            
            # Adapt step size
            # Decrease step size linearly
            progress = used / self.budget
            step_size = 0.5 * np.max(hi - lo) * (1 - progress) + 0.05 * np.max(hi - lo)
            
        return X
