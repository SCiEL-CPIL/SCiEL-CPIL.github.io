import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initialization with Sobol
        n_init = 50
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points_01 = sampler.random(n_init)
        X = lo + (hi - lo) * points_01
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        eval_count = n_init
        
        # Maintain list of indices of non-dominated points
        # Since we start with random-ish points, most are non-dominated
        # We will maintain a list of all points and their fitness, and prune
        
        # To speed up, we'll keep a list of active points
        # But for 200 evals, keeping all and sorting at the end is also viable
        # However, we need to select parents from the non-dominated set
        
        def is_dominated(f, F_set):
            # Returns True if f is dominated by any point in F_set
            # Vectorized check
            # F_set shape (N, 3)
            # f shape (3,)
            return np.any(np.all(F_set <= f, axis=1) & np.any(F_set < f, axis=1))

        def dominates(f, F_set):
            # Returns indices of points in F_set dominated by f
            return np.where(np.all(f <= F_set, axis=1) & np.any(f < F_set, axis=1))[0]

        # Active population: list of indices into X, F
        # Actually, let's just use X and F as the population, and prune it.
        
        pop_size = 50
        
        while eval_count < self.budget:
            if time.time() > deadline:
                break
                
            # Identify non-dominated points in current population
            # This is O(N^2)
            m = F.shape[0]
            non_dom_mask = np.ones(m, dtype=bool)
            for i in range(m):
                if not non_dom_mask[i]: continue
                for j in range(m):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        non_dom_mask[j] = False
            
            non_dom_indices = np.where(non_dom_mask)[0]
            
            if len(non_dom_indices) == 0:
                # Fallback, should not happen
                non_dom_indices = [0]
                
            # Select parent
            # Use crowding distance on non-dominated points
            F_nd = F[non_dom_indices]
            cd = np.zeros(len(non_dom_indices))
            for k in range(self.n_obj):
                sorted_idx = np.argsort(F_nd[:, k])
                if len(sorted_idx) > 2:
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    F_min = F_nd[sorted_idx[0], k]
                    F_max = F_nd[sorted_idx[-1], k]
                    if F_max > F_min:
                        for i in range(1, len(sorted_idx)-1):
                            idx = sorted_idx[i]
                            cd[idx] += (F_nd[sorted_idx[i+1], k] - F_nd[sorted_idx[i-1], k]) / (F_max - F_min)
            
            # Pick parent with max CD
            parent_local_idx = np.argmax(cd)
            parent_idx = non_dom_indices[parent_local_idx]
            
            # Mutation
            progress = eval_count / self.budget
            sigma = 0.3 * (1 - progress) + 0.05
            mutation = np.random.randn(self.dim) * sigma * (hi - lo)
            child = X[parent_idx] + mutation
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            eval_count += 1
            
            # Check if child is dominated by current non-dominated set
            if is_dominated(fc, F_nd):
                # If dominated, we might still keep it if it's better than the worst dominated point
                # But for simplicity, discard dominated children in early stages?
                # No, let's just add it and let pruning handle it
                pass
            
            # Add child to population
            X = np.vstack([X, child])
            F = np.vstack([F, fc])
            
            # Prune population to pop_size
            if F.shape[0] > pop_size:
                # Find non-dominated
                m = F.shape[0]
                non_dom_mask = np.ones(m, dtype=bool)
                for i in range(m):
                    if not non_dom_mask[i]: continue
                    for j in range(m):
                        if i == j: continue
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            non_dom_mask[j] = False
                
                non_dom_indices = np.where(non_dom_mask)[0]
                dominated_indices = np.where(~non_dom_mask)[0]
                
                # We want to keep all non-dominated if possible
                if len(non_dom_indices) <= pop_size:
                    # Keep all non-dominated, fill rest with best dominated
                    # "Best" dominated: closest to non-dominated front?
                    # Just keep random dominated for now, or the ones with best objective sum
                    sum_obj = np.sum(F, axis=1)
                    # Sort dominated by sum_obj (lower is better)
                    sorted_dom = dominated_indices[np.argsort(sum_obj[dominated_indices])]
                    needed = pop_size - len(non_dom_indices)
                    keep_indices = list(non_dom_indices) + list(sorted_dom[:needed])
                else:
                    # Too many non-dominated, prune by crowding
                    F_nd = F[non_dom_indices]
                    cd = np.zeros(len(non_dom_indices))
                    for k in range(self.n_obj):
                        sorted_idx = np.argsort(F_nd[:, k])
                        if len(sorted_idx) > 2:
                            cd[sorted_idx[0]] = np.inf
                            cd[sorted_idx[-1]] = np.inf
                            F_min = F_nd[sorted_idx[0], k]
                            F_max = F_nd[sorted_idx[-1], k]
                            if F_max > F_min:
                                for i in range(1, len(sorted_idx)-1):
                                    idx = sorted_idx[i]
                                    cd[idx] += (F_nd[sorted_idx[i+1], k] - F_nd[sorted_idx[i-1], k]) / (F_max - F_min)
                    
                    # Sort non-dominated by CD descending
                    sorted_nd = non_dom_indices[np.argsort(-cd)]
                    keep_indices = list(sorted_nd[:pop_size])
                
                X = X[keep_indices]
                F = F[keep_indices]
        
        return X
