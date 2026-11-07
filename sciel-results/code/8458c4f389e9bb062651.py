import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        pop_size = 20
        
        # Initialize population
        X = lo + range_ * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Maintain a list of non-dominated points (archive)
        # We will record all points that are non-dominated in the population
        # But for efficiency, we just keep the population and filter at the end.
        # However, to maximize hypervolume, we want to keep diverse points.
        # Let's use a simple strategy: 
        # 1. Start with random points.
        # 2. Generate children via mutation of non-dominated parents.
        # 3. Replace the worst point in the population if the child is better 
        #    in terms of some metric, or if it dominates an existing point.
        
        # For WFG7, a simple random search with local refinement might be good.
        # Let's try a hybrid: 
        # - Keep a population of size N.
        # - Each step: 
        #   - Pick a parent from the current non-dominated set (if available, else random).
        #   - Mutate it.
        #   - If the child dominates any point in the population, replace the dominated one.
        #   - Else, if the child is non-dominated w.r.t population, replace the worst point 
        #     (e.g., largest distance to reference point, or smallest hypervolume contribution).
        
        ref_point = np.array([3.0, 5.0, 7.0])
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def is_non_dominated(f, f_set):
            # f: (n_obj,)
            # f_set: (N, n_obj)
            for j in range(len(f_set)):
                if dominates(f_set[j], f):
                    return False
            return True
        
        # Helper to find non-dominated indices in F
        def find_non_dominated_indices(F):
            n = len(F)
            nd_indices = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and dominates(F[j], F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated set
            nd_indices = find_non_dominated_indices(F)
            
            # If no non-dominated points (should not happen), pick random
            if len(nd_indices) == 0:
                parent_idx = np.random.randint(pop_size)
            else:
                parent_idx = nd_indices[np.random.randint(len(nd_indices))]
            
            parent = X[parent_idx]
            
            # Mutation: Gaussian
            # Adaptive sigma? Start with 0.1, maybe reduce over time
            # For simplicity, fixed sigma
            sigma = 0.05
            child = parent + sigma * range_ * np.random.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Check if child dominates any point in population
            replaced = False
            for i in range(pop_size):
                if dominates(fc, F[i]):
                    X[i] = child
                    F[i] = fc
                    replaced = True
                    break
            
            # If not replaced by domination, check if child is non-dominated
            if not replaced:
                # Check if child is non-dominated w.r.t current population
                # We already know it doesn't dominate any, so we just need to check 
                # if it is dominated by any.
                dominated_by_any = False
                for i in range(pop_size):
                    if dominates(F[i], fc):
                        dominated_by_any = True
                        break
                
                if not dominated_by_any:
                    # Child is non-dominated. Replace the "worst" point.
                    # Define worst as the one with the largest distance to the reference point
                    # in the objective space (normalized).
                    # Or simply replace a random point to maintain diversity.
                    # Let's replace the point with the largest sum of normalized objectives.
                    # Normalize by ref_point
                    scores = np.sum(F / ref_point, axis=1)
                    worst_idx = np.argmax(scores)
                    X[worst_idx] = child
                    F[worst_idx] = fc
        
        # Return the non-dominated set
        nd_indices = find_non_dominated_indices(F)
        return X[nd_indices]
