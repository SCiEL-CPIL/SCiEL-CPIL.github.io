import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        bounds_range = hi - lo
        
        n_obj = self.n_obj
        dim = self.dim
        
        # Helper: Check if f1 dominates f2
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # Archive stores (x, f) for non-dominated points
        archive_x = []
        archive_f = []
        
        # Helper: Update archive with new point
        def update_archive(x_new, f_new):
            nonlocal archive_x, archive_f
            
            # If new point is dominated by any existing, discard
            for f_old in archive_f:
                if dominates(f_old, f_new):
                    return False
            
            # Remove existing points dominated by new point
            new_x = []
            new_f = []
            for i in range(len(archive_f)):
                if not dominates(f_new, archive_f[i]):
                    new_x.append(archive_x[i])
                    new_f.append(archive_f[i])
            
            new_x.append(x_new)
            new_f.append(f_new)
            archive_x = new_x
            archive_f = new_f
            return True

        evals = 0
        
        # 1. Initialization: Space-filling Halton sequence for broad coverage
        # This is better than random initialization for hypervolume as it ensures diversity
        def halton(dim, n, skip=1):
            points = np.zeros((n, dim))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for d in range(dim):
                p = primes[d % len(primes)]
                for i in range(n):
                    x = 0.0
                    f = 1.0 / p
                    idx = i + skip
                    while idx > 0:
                        idx, rem = divmod(idx, p)
                        x += f * rem
                        f /= p
                    points[i, d] = x
            return points

        n_init = min(40, self.budget // 4)
        H = halton(dim, n_init)
        X_init = lo + bounds_range * H
        
        for i in range(n_init):
            if time.time() > deadline or evals >= self.budget:
                break
            x = X_init[i]
            f = np.array(problem(x))
            evals += 1
            update_archive(x, f)
            
        if not archive_x:
            # Fallback if something went wrong (shouldn't happen)
            x = lo + bounds_range * 0.5
            f = np.array(problem(x))
            evals += 1
            update_archive(x, f)

        # 2. Weighted-sum scalarization with local search
        # This structural change (from greedy coverage to scalarization) is the key.
        # We will generate multiple weighted-sum objectives and optimize each using a simple local search.
        # This creates a "structured" coverage of the objective space.
        
        # Generate weight vectors. For 3 objectives, we can use a simplex lattice or random points on the simplex.
        # Let's use random points on the simplex for flexibility.
        rng = np.random.default_rng(42)
        
        n_weights = 20 # Number of sub-problems to solve
        weights = rng.random((n_weights, n_obj))
        weights /= weights.sum(axis=1, keepdims=True)
        
        # For each weight vector, we start from the best point in the current archive
        # that minimizes the weighted sum, and perform a few steps of local search.
        
        # Helper: Find best point for a weight vector
        def best_for_weight(F, w):
            if len(F) == 0:
                return None
            return np.argmin(F @ w)
        
        # Helper: Local search step (simple random perturbation)
        def local_search_step(x, F, w, bounds, rng, step_size=0.1):
            # Generate a few candidates
            best_x = x
            best_f = F
            for _ in range(5): # Try 5 candidates
                if time.time() > deadline or evals >= self.budget:
                    break
                noise = step_size * bounds_range * rng.standard_normal(dim)
                x_new = np.clip(x + noise, bounds[0], bounds[1])
                f_new = np.array(problem(x_new))
                evals += 1
                if f_new @ w < best_f @ w:
                    best_x = x_new
                    best_f = f_new
            return best_x, best_f

        # Main loop: iterate over weights, refining each
        # We have a budget, so we need to be careful how many steps we take per weight
        # Total budget is 200. Init took ~40. Remaining ~160.
        # 20 weights * 5 steps * 80% success rate? 
        # Let's just run until budget is exhausted or time is up.
        
        # Sort weights by some heuristic? Or just random order.
        # Let's shuffle weights to explore different regions
        perm = rng.permutation(n_weights)
        
        for w_idx in perm:
            if evals >= self.budget or time.time() > deadline:
                break
            
            w = weights[w_idx]
            
            # Find starting point
            if len(archive_f) > 0:
                best_idx = best_for_weight(np.array(archive_f), w)
                start_x = archive_x[best_idx]
                start_f = np.array(archive_f[best_idx])
            else:
                start_x = lo + bounds_range * 0.5
                start_f = np.array(problem(start_x))
                evals += 1
                update_archive(start_x, start_f)
            
            # Perform local search
            # Step size can be adaptive or fixed. Fixed 0.1 is good for [0,1] box.
            # We try a few iterations of "hill climbing" on the scalarized objective
            current_x = start_x
            current_f = start_f
            
            # Try to improve
            for _ in range(10): # Max 10 steps per weight
                if evals >= self.budget or time.time() > deadline:
                    break
                
                # Generate candidate
                noise = 0.1 * bounds_range * rng.standard_normal(dim)
                x_new = np.clip(current_x + noise, lo, hi)
                f_new = np.array(problem(x_new))
                evals += 1
                
                # Check if it's better in scalarized sense
                if f_new @ w < current_f @ w:
                    # It's an improvement for this weight.
                    # Update local best
                    current_x = x_new
                    current_f = f_new
                    
                    # Try to add to archive
                    update_archive(current_x, current_f)
                else:
                    # If it's not better, maybe we should try a different direction or stop?
                    # For now, just continue trying random perturbations around current best
                    # But to save budget, maybe break if we've tried several and none worked?
                    # Let's just keep trying for a few more steps to escape local minima
                    pass

        return archive_x
