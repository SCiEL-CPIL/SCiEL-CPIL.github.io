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
        max_time = 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use Sobol for better initial spread
        # Budget 200. Let's use 100 for init, 100 for refinement.
        n_init = 100
        n_refine = self.budget - n_init
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        # Sobol requires power of 2 for best efficiency, but works for others.
        # 128 is a good power of 2 close to 100.
        n_sobol = 128
        if n_sobol > self.budget:
             n_sobol = 64
             n_init = n_sobol
             n_refine = self.budget - n_init
             
        sobol_points = sampler.random(n_sobol)
        # Scale to [lo, hi]
        X_init = lo + (hi - lo) * sobol_points
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used = n_sobol
        
        # Store results
        X_all = X_init
        F_all = F_init
        
        # Helper: Check if a point is dominated by any in F_all
        def is_dominated(f_new, F_archive):
            # f_new is shape (n_obj,)
            # F_archive is shape (N, n_obj)
            # Returns True if any point in F_archive dominates f_new
            # Dominance: a <= b and a < b
            # Vectorized: 
            # a <= b for all objectives -> np.all(F_archive <= f_new, axis=1)
            # a < b for at least one -> np.any(F_archive < f_new, axis=1)
            # We need (all <=) AND (any <)
            leq = np.all(F_archive <= f_new, axis=1)
            lt = np.any(F_archive < f_new, axis=1)
            return np.any(leq & lt)
        
        # Helper: Remove points dominated by a new point
        def remove_dominated(F_archive, f_new):
            # f_new dominates a point if f_new <= point and f_new < point
            leq = np.all(f_new <= F_archive, axis=1)
            lt = np.any(f_new < F_archive, axis=1)
            mask = leq & lt
            return ~mask # Keep points NOT dominated by f_new
        
        # Refinement phase
        # We will do local search around the best points.
        # To be efficient, we'll just pick random points from the current archive and perturb them.
        # Or better: pick the "most central" or "best" points.
        # Simple strategy: Randomly select from archive, perturb, evaluate.
        
        # Adaptive step size
        base_step = 0.1
        
        while used < self.budget:
            if time.time() - start_time > max_time:
                break
            
            # Select a parent from the current archive
            # To encourage exploration of the front, we can pick randomly.
            # Or pick from the non-dominated set.
            # Let's maintain a list of non-dominated points indices.
            # For simplicity, let's just pick a random index from F_all.
            # This might re-explore dominated areas, but with small budget, it's okay.
            # Better: Maintain a list of non-dominated indices.
            
            # Recompute non-dominated set? Expensive. 
            # Let's just keep a list of "promising" points.
            # Initially, all are promising.
            # If a new point is dominated, we don't add it.
            # If a new point dominates others, we remove them.
            
            # Let's maintain `X_active` and `F_active` as the current Pareto front approximation.
            # Actually, the code above `X_all` and `F_all` contains everything.
            # Let's filter `F_all` to get the non-dominated set for parent selection.
            
            # Efficiently finding non-dominated set in O(N^2) is fine for N=200.
            nd_indices = []
            for i in range(len(F_all)):
                dominated = False
                for j in range(len(F_all)):
                    if i == j: continue
                    if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            
            if not nd_indices:
                 # Fallback if something went wrong
                 nd_indices = [0]
                 
            # Pick a random parent from the non-dominated set
            parent_idx = nd_indices[np.random.randint(len(nd_indices))]
            parent = X_all[parent_idx]
            
            # Perturb
            # Use a Gaussian perturbation with decaying step size?
            # Or fixed small step.
            step = base_step * (hi - lo)
            child = parent + np.random.normal(0, step, self.dim)
            child = np.clip(child, lo, hi)
            
            f_child = problem(child)
            used += 1
            
            # Check if child is useful
            # 1. Is it dominated by current F_all?
            if is_dominated(f_child, F_all):
                continue
            
            # 2. If not dominated, add it and remove any points it dominates.
            X_all = np.vstack([X_all, child])
            F_all = np.vstack([F_all, f_child])
            
            # Remove dominated points from F_all
            # We need to identify which rows in F_all are dominated by f_child.
            # f_child is the last row.
            mask = remove_dominated(F_all[:-1], f_child)
            # Keep only non-dominated by f_child from the previous set
            F_all = F_all[mask]
            X_all = X_all[mask]
            # Add f_child back
            F_all = np.vstack([F_all, f_child])
            X_all = np.vstack([X_all, child])

        return X_all
