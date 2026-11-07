import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Adaptive Random Search with Non-Dominated Set Maintenance
        # This is a structural change from deterministic grid to stochastic adaptive search.
        
        # State
        evaluated_points = []
        evaluated_objs = []
        nd_set_indices = [] # Indices into evaluated_points that are non-dominated
        
        def is_dominated(a, b):
            # a dominates b if a <= b in all objectives and a < b in at least one
            return np.all(a <= b) and np.any(a < b)
            
        def is_non_dominated(obj, nd_objs):
            # Check if obj is not dominated by any in nd_objs
            # and does not dominate any in nd_objs (if it does, it should replace them, 
            # but for simplicity we just add to ND set and prune)
            for o in nd_objs:
                if is_dominated(o, obj):
                    return False
            return True

        def prune_nd_set(nd_indices, all_objs):
            # Remove dominated points from nd_indices
            new_nd = []
            nd_objs = [all_objs[i] for i in nd_indices]
            for i, idx in enumerate(nd_indices):
                dominated = False
                for j, jdx in enumerate(nd_indices):
                    if i == j: continue
                    if is_dominated(all_objs[jdx], all_objs[idx]):
                        dominated = True
                        break
                if not dominated:
                    new_nd.append(idx)
            return new_nd

        n_evals = 0
        
        # Phase 1: Initial Random Sampling (20% of budget)
        n_init = max(10, budget // 5)
        
        for _ in range(n_init):
            if n_evals >= budget or time.time() > deadline:
                break
            x = np.random.uniform(lo, hi)
            f = problem(x)
            evaluated_points.append(x)
            evaluated_objs.append(f)
            n_evals += 1
            
        # Initialize ND set
        nd_set_indices = list(range(len(evaluated_points)))
        nd_set_indices = prune_nd_set(nd_set_indices, evaluated_objs)
        
        # Phase 2: Adaptive Search (80% of budget)
        # Mix of random and mutation of ND set points
        while n_evals < budget and time.time() < deadline:
            # Decide exploration vs exploitation
            if np.random.rand() < 0.5:
                # Random
                x = np.random.uniform(lo, hi)
            else:
                # Mutate a point from ND set
                if not nd_set_indices:
                    x = np.random.uniform(lo, hi)
                else:
                    idx = np.random.choice(nd_set_indices)
                    x = evaluated_points[idx]
                    # Gaussian mutation
                    sigma = 0.1
                    x = x + np.random.normal(0, sigma, dim)
                    x = np.clip(x, lo, hi)
            
            f = problem(x)
            evaluated_points.append(x)
            evaluated_objs.append(f)
            n_evals += 1
            
            # Check if new point is non-dominated
            if is_non_dominated(f, [evaluated_objs[i] for i in nd_set_indices]):
                nd_set_indices.append(n_evals - 1)
                # Prune the set to keep only non-dominated
                nd_set_indices = prune_nd_set(nd_set_indices, evaluated_objs)
        
        return evaluated_points
