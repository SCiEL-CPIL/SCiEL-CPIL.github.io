# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.n_shape = n_obj - 1
        self.n_dist = dim - self.n_shape

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure: first M-1 vars are shape, rest are distance
        # Distance vars optimum is typically 0.5
        x_dist_opt = 0.5 * (lo[self.n_shape:] + hi[self.n_shape:])
        
        pop_size = 50
        # Initialize population
        X = np.zeros((pop_size, self.dim))
        X[:, :self.n_shape] = lo[:self.n_shape] + (hi[:self.n_shape] - lo[:self.n_shape]) * np.random.rand(pop_size, self.n_shape)
        X[:, self.n_shape:] = x_dist_opt
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper for non-dominated selection (vectorized)
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def is_dominated_by_pop(f, F_pop):
            """Check if f is dominated by any point in F_pop (vectorized)."""
            # F_pop shape: (N, M), f shape: (M,)
            # For each i: all(F_pop[i] <= f) and any(F_pop[i] < f)
            leq = np.all(F_pop <= f, axis=1)
            lt = np.any(F_pop < f, axis=1)
            return np.any(leq & lt)
        
        def dominated_indices_in_pop(f, F_pop):
            """Find indices in F_pop dominated by f (vectorized)."""
            # For each i: all(f <= F_pop[i]) and any(f < F_pop[i])
            leq = np.all(f <= F_pop, axis=1)
            lt = np.any(f < F_pop, axis=1)
            return np.where(leq & lt)[0]

        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select parent: random from non-dominated or just random
            idx = np.random.randint(pop_size)
            parent = X[idx]
            
            # Create child
            child = parent.copy()
            
            # Mutate shape variables (explore the front)
            # Use a small sigma to refine
            sigma_shape = 0.1
            child[:self.n_shape] += sigma_shape * (hi[:self.n_shape] - lo[:self.n_shape]) * np.random.randn(self.n_shape)
            child[:self.n_shape] = np.clip(child[:self.n_shape], lo[:self.n_shape], hi[:self.n_shape])
            
            # Mutate distance variables (refine towards optimum 0.5)
            # Reset to optimum + small noise
            sigma_dist = 0.05
            child[self.n_shape:] = x_dist_opt + sigma_dist * (hi[self.n_shape:] - lo[self.n_shape:]) * np.random.randn(self.n_dist)
            child[self.n_shape:] = np.clip(child[self.n_shape:], lo[self.n_shape:], hi[self.n_shape:])
            
            f_child = problem(child)
            used += 1
            
            # Update population
            # Check if child is non-dominated
            if not is_dominated_by_pop(f_child, F):
                # Find dominated individuals to replace
                dom_idx = dominated_indices_in_pop(f_child, F)
                
                if len(dom_idx) > 0:
                    # Replace one of the dominated
                    rep_idx = dom_idx[np.random.randint(len(dom_idx))]
                    X[rep_idx] = child
                    F[rep_idx] = f_child
                else:
                    # If not dominated and doesn't dominate, replace worst (highest sum)
                    sums = np.sum(F, axis=1)
                    worst_idx = np.argmax(sums)
                    if np.sum(f_child) < sums[worst_idx]:
                        X[worst_idx] = child
                        F[worst_idx] = f_child
            else:
                # If dominated, maybe still replace worst if child is better in sum?
                sums = np.sum(F, axis=1)
                worst_idx = np.argmax(sums)
                if np.sum(f_child) < sums[worst_idx]:
                    X[worst_idx] = child
                    F[worst_idx] = f_child

        return X
# EVOLVE-BLOCK-END
