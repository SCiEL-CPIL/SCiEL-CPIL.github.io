import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Archive of non-dominated points
        archive_x = []
        archive_f = []
        evals = 0
        
        def is_dominated(f_new, archive_f):
            # f_new is dominated if there exists f in archive such that f <= f_new and f != f_new
            if len(archive_f) == 0:
                return False
            # Vectorized check
            # f_new shape (n_obj,), archive_f shape (n, n_obj)
            # dominated if any row in archive_f is <= f_new and at least one <
            # Note: problem minimizes
            leq = np.all(archive_f <= f_new, axis=1)
            lt = np.any(archive_f < f_new, axis=1)
            return np.any(leq & lt)
            
        def dominates(f_new, f_old):
            # f_new dominates f_old if f_new <= f_old and f_new != f_old
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def add_to_archive(x, f):
            nonlocal evals
            # Check if dominated by existing
            if is_dominated(f, archive_f):
                return False
            
            # Remove points dominated by new
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)
                
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            # Limit archive size to prevent explosion? 
            # With 200 evals, max size is 200. 
            # But we want to keep the best spread.
            # If size > 50, we might prune.
            
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        # 1. Initialization
        init_size = 20
        X_init = lo + (hi - lo) * np.random.random((init_size, self.dim))
        for x in X_init:
            if time.time() > deadline: return archive_x
            f = problem(x)
            evals += 1
            add_to_archive(x, f)
            
        # 2. Main Search
        # Use a step size that adapts
        step_size = 0.2 * (hi - lo)
        
        while evals < self.budget - 50: # Reserve 50 for polish
            if time.time() > deadline: return archive_x
            
            # Select parent from archive
            # Prefer points that are "central" or have good spread?
            # Random selection is okay, but maybe tournament on hypervolume contribution?
            # Simple: Random parent
            idx = np.random.randint(len(archive_x))
            parent = archive_x[idx]
            
            # Mutate
            # Adaptive step: shrink if no improvement? 
            # Or just fixed small step for exploitation?
            # Let's try Gaussian mutation
            child = parent + step_size * np.random.normal(size=self.dim)
            child = np.clip(child, lo, hi)
            
            f_child = problem(child)
            evals += 1
            
            # If it's non-dominated, add it
            added = add_to_archive(child, f_child)
            if added:
                # Maybe reduce step size slightly?
                step_size *= 0.95
            else:
                # If not added, maybe increase step? Or keep?
                pass
                
        # 3. Polish Phase
        # Take top K points from archive
        # Sort by something? 
        # We want to improve the Pareto front.
        # Let's pick 10 points evenly spaced or just the last 10 non-dominated ones.
        # To be safe, pick 10 random non-dominated points from the archive.
        if len(archive_x) > 0:
            k = min(10, len(archive_x))
            indices = np.random.choice(len(archive_x), k, replace=False)
            
            for idx in indices:
                if evals >= self.budget: break
                if time.time() > deadline: break
                
                x_curr = archive_x[idx].copy()
                f_curr = archive_f[idx].copy()
                
                # Local search: try to find a dominating neighbor
                # Use a small step
                local_step = 0.05 * (hi - lo)
                improved = True
                while improved and evals < self.budget:
                    improved = False
                    # Try a few random directions
                    for _ in range(5):
                        if evals >= self.budget: break
                        direction = np.random.normal(size=self.dim)
                        direction /= np.linalg.norm(direction) + 1e-9
                        x_try = x_curr + local_step * direction
                        x_try = np.clip(x_try, lo, hi)
                        
                        f_try = problem(x_try)
                        evals += 1
                        
                        # Check if f_try dominates f_curr
                        if np.all(f_try <= f_curr) and np.any(f_try < f_curr):
                            x_curr = x_try
                            f_curr = f_try
                            improved = True
                            break # Restart local search from new point?
                            # Or continue from here.
                        elif not is_dominated(f_try, [f_curr]):
                            # If it's non-dominated w.r.t current, but doesn't dominate,
                            # it might be a better point on the front?
                            # For hypervolume, any non-dominated point is good.
                            # But we are polishing a specific point.
                            # Let's only accept if it dominates or is very close?
                            # Actually, if it doesn't dominate, it might be worse in one obj, better in other.
                            # We want to tighten the front.
                            # Let's stick to domination for polish to be safe.
                            pass

                # Update archive with polished point
                # It might dominate old points
                # We can just add it to archive logic
                # But we need to replace the old one if it's dominated?
                # The add_to_archive handles domination.
                # However, if x_curr is the same as old, and f_curr is better, 
                # add_to_archive will remove the old one if dominated.
                
                # Check if we should replace in archive
                # Simple: just call add_to_archive
                # But if it's the same point, it won't be added if it's dominated by itself?
                # is_dominated checks against archive_f.
                # If x_curr is in archive, f_curr might be in archive_f.
                # If f_curr < f_old, it dominates.
                
                # To ensure we don't duplicate, we can remove the old index first?
                # Or just let add_to_archive handle it.
                # If f_curr is not dominated by archive, it will be added.
                # If it dominates old points, they are removed.
                
                add_to_archive(x_curr, f_curr)
                
                # Reduce local step for next point?
                local_step *= 0.8

        return archive_x
