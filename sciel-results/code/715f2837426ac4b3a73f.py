"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with adaptive mutation and tournament selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # Initialize population
        pop_size = min(40, self.budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            # A point dominates f if it's <= in all objectives and < in at least one
            for f_other in F_set:
                if np.all(f_other <= f) and np.any(f_other < f):
                    return True
            return False
        
        while used < self.budget:
            # Check time budget
            if time.time() > deadline:
                break
            
            # Compute non-dominated front
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated[i] = True
                            break
            
            # Get non-dominated parents
            non_dom_idx = np.where(~dominated)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))
            
            # Select parent from non-dominated set
            parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            
            # Adaptive mutation: decrease step size over time
            progress = used / self.budget
            step_size = 0.3 * (1 - 0.5 * progress)
            
            # Create child with mutation
            child = X[parent_idx].copy()
            # Mutate a random subset of dimensions
            n_mutate = max(1, int(dim * rng.uniform(0.3, 0.8)))
            mut_dims = rng.choice(dim, size=n_mutate, replace=False)
            child[mut_dims] += step_size * (hi[mut_dims] - lo[mut_dims]) * rng.standard_normal(n_mutate)
            child = np.clip(child, lo, hi)
            
            # Evaluate child
            fc = problem(child)
            used += 1
            
            # Replace worst point in population
            # Use a simple heuristic: maximize sum of objectives
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc
            
            # Track all evaluated points
            all_X.append(child.copy())
            all_F.append(fc.copy())
            
            # Every 10 evaluations, do a local refinement around best point
            if used % 10 == 0 and used < self.budget:
                # Find point with minimum sum of objectives
                best_idx = int(np.argmin(np.sum(F, axis=1)))
                # Do a few small steps around it
                for _ in range(2):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    local_child = X[best_idx] + 0.05 * (hi - lo) * rng.standard_normal(dim)
                    local_child = np.clip(local_child, lo, hi)
                    local_fc = problem(local_child)
                    used += 1
                    # Replace worst if local point is better
                    if np.sum(local_fc) < np.sum(F[worst_idx]):
                        X[worst_idx] = local_child
                        F[worst_idx] = local_fc
                    all_X.append(local_child.copy())
                    all_F.append(local_fc.copy())
        
        return X
# EVOLVE-BLOCK-END
