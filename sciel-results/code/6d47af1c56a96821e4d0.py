import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin: evaluator kills at 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        archive_x = []
        archive_f = []
        evals = 0
        
        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            af = np.array(archive_f_list)
            leq = np.all(af <= f_new, axis=1)
            lt = np.any(af < f_new, axis=1)
            return np.any(leq & lt)
            
        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def add_to_archive(x, f):
            nonlocal evals
            if is_dominated(f, archive_f):
                return False
            
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)
                
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        def time_ok():
            return time.time() < deadline

        # 1. Initialization: use 30 points for better coverage
        init_size = 30
        X_init = lo + range_ * np.random.random((init_size, self.dim))
        for x in X_init:
            if not time_ok():
                break
            f = problem(x)
            evals += 1
            add_to_archive(x, f)
        
        if not time_ok():
            return archive_x
            
        # 2. Main search: use remaining budget for mutation-based search
        # Reserve 20 evals for final local polish
        main_budget = self.budget - 20
        
        # Adaptive step size
        base_step = 0.15 * range_
        step_sizes = base_step.copy()
        
        while evals < main_budget and time_ok():
            if len(archive_x) == 0:
                break
                
            # Select parent: prefer points with good hypervolume contribution
            # Simple approach: random selection from archive
            idx = np.random.randint(len(archive_x))
            parent = archive_x[idx]
            
            # Gaussian mutation with adaptive step
            child = parent + step_sizes * np.random.normal(size=self.dim)
            child = np.clip(child, lo, hi)
            
            f_child = problem(child)
            evals += 1
            
            added = add_to_archive(child, f_child)
            if added:
                # Reduce step size when finding new points (exploit)
                step_sizes *= 0.97
            else:
                # Occasionally increase step size to explore
                if np.random.random() < 0.1:
                    step_sizes *= 1.05
                    step_sizes = np.minimum(step_sizes, 0.3 * range_)
        
        # 3. Polish phase: local search on best archive points
        if not time_ok():
            return archive_x
            
        # Sort archive by "quality" - use simple heuristic: sum of objectives
        if len(archive_x) > 0:
            # Pick up to 10 points from archive for local polish
            k = min(10, len(archive_x))
            # Use random selection for diversity
            indices = np.random.choice(len(archive_x), k, replace=False)
            
            for idx in indices:
                if evals >= self.budget or not time_ok():
                    break
                    
                x_curr = archive_x[idx].copy()
                f_curr = archive_f[idx].copy()
                
                # Local search with decreasing step
                local_step = 0.03 * range_
                improved = True
                max_local_iters = 5
                
                for _ in range(max_local_iters):
                    if evals >= self.budget or not time_ok():
                        break
                    if not improved:
                        break
                        
                    improved = False
                    # Try multiple random directions
                    for _ in range(8):
                        if evals >= self.budget or not time_ok():
                            break
                            
                        direction = np.random.normal(size=self.dim)
                        norm = np.linalg.norm(direction)
                        if norm < 1e-10:
                            continue
                        direction /= norm
                        
                        x_try = x_curr + local_step * direction
                        x_try = np.clip(x_try, lo, hi)
                        
                        f_try = problem(x_try)
                        evals += 1
                        
                        # Accept if it dominates current
                        if np.all(f_try <= f_curr) and np.any(f_try < f_curr):
                            x_curr = x_try
                            f_curr = f_try
                            improved = True
                            local_step *= 0.7  # Shrink step on improvement
                            break
                        # Also accept if it's non-dominated w.r.t current (for front diversity)
                        elif not is_dominated(f_try, [f_curr]) and not is_dominated(f_curr, [f_try]):
                            # This point is on the front between them
                            # Add to archive but don't update x_curr
                            add_to_archive(x_try, f_try)
                            
                # Add polished point to archive
                if time_ok() and evals < self.budget:
                    add_to_archive(x_curr, f_curr)
                    
                # Reduce local step for next point
                local_step *= 0.8

        return archive_x
