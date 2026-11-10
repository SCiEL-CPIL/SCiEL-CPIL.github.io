import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 typically has the first M variables controlling the objectives directly.
        # The rest contribute to the 'distance' or 'shape' function which is minimized at 0.
        # We assume the first 3 variables are the primary decision variables for the 3 objectives.
        n_primary = min(3, self.dim)
        
        # Strategy:
        # 1. Dense sampling of the primary variables (first 3) to cover the disconnected front.
        # 2. Set non-primary variables to 0 (or lower bound if 0 is not in range) to minimize auxiliary terms.
        # 3. Use remaining budget for local refinement of non-dominated points.
        
        # Calculate grid points for primary variables
        # We want to use about 60% of the budget for the initial sweep.
        sweep_budget = int(self.budget * 0.6)
        
        # Determine grid resolution
        # If n_primary is 3, we want k^3 <= sweep_budget
        k = int(np.floor(sweep_budget ** (1.0 / n_primary)))
        if k < 2: k = 2
        # Adjust k to not exceed sweep_budget
        while k**n_primary > sweep_budget:
            k -= 1
        
        # Generate grid points for primary variables
        # Use linspace to cover the range
        grid_axes = [np.linspace(lo[i], hi[i], k) for i in range(n_primary)]
        
        # Non-primary variables: set to 0 if possible, else lower bound
        # In WFG, G(x) is minimized when x_i = 0.
        non_primary_defaults = np.zeros(self.dim - n_primary)
        for i in range(n_primary, self.dim):
            if lo[i] <= 0 <= hi[i]:
                non_primary_defaults[i - n_primary] = 0.0
            else:
                non_primary_defaults[i - n_primary] = lo[i] # Fallback to bound
        
        # Create initial population
        X_list = []
        F_list = []
        
        # Iterate over grid
        indices = np.indices((k,) * n_primary)
        indices = indices.reshape(n_primary, -1).T
        
        for idx in indices:
            x = np.zeros(self.dim)
            for i in range(n_primary):
                x[i] = grid_axes[i][idx[i]]
            x[n_primary:] = non_primary_defaults
            X_list.append(x)
            
        # Evaluate initial population
        for x in X_list:
            if time.time() > deadline:
                break
            F_list.append(problem(x))
        
        used = len(X_list)
        
        if not F_list:
            # Fallback if we didn't even get one eval
            x = (lo + hi) / 2
            F_list = [problem(x)]
            X_list = [x]
            used = 1

        F = np.array(F_list)
        X = np.array(X_list)
        
        # Find non-dominated points
        def is_non_dominated(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return nd
        
        nd_mask = is_non_dominated(F)
        nd_indices = np.where(nd_mask)[0]
        
        # If we have budget left, perform local search on non-dominated points
        # We will perturb the non-dominated points slightly.
        # We prioritize points that are "central" in the objective space or just pick random ones from the ND set.
        
        if used < self.budget:
            # Shuffle ND indices to ensure diversity in refinement
            np.random.shuffle(nd_indices)
            
            # We will refine the top 'refine_budget' points
            # But we must be careful not to evaluate too many times.
            # Let's just iterate through the ND set and try to improve them.
            
            # Simple local search: Gaussian perturbation
            # Step size decreases over time
            current_step = 0.1
            
            # We will try to improve each ND point once or twice
            # Let's allocate remaining budget evenly among ND points
            n_nd = len(nd_indices)
            if n_nd == 0:
                nd_indices = [0] # Fallback to first point
                n_nd = 1
                
            evals_per_point = (self.budget - used) // n_nd
            if evals_per_point < 1:
                evals_per_point = 1
                # If we can't even do 1 per point, we just stop
                if self.budget - used < n_nd:
                    # Prioritize
                    nd_indices = nd_indices[:self.budget - used]
                    n_nd = len(nd_indices)
                    evals_per_point = 1

            for idx in nd_indices:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                base_x = X[idx].copy()
                base_f = F[idx].copy()
                
                for _ in range(evals_per_point):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Perturb all variables, but maybe less for non-primary?
                    # Actually, perturbing all is safer.
                    noise = np.random.normal(0, current_step, self.dim)
                    new_x = base_x + noise
                    new_x = np.clip(new_x, lo, hi)
                    
                    new_f = problem(new_x)
                    used += 1
                    
                    # If new point is better than base point (in any objective, and not worse in others? No, just keep if non-dominated relative to base)
                    # Simpler: If new_f is better than base_f in at least one objective and not worse in any, replace base.
                    if np.any(new_f < base_f) and np.all(new_f <= base_f):
                        base_x = new_x
                        base_f = new_f
                        X[idx] = base_x
                        F[idx] = base_f
                    else:
                        # Even if not strictly better, we might want to add it to the population if it's non-dominated?
                        # For simplicity in this budget, we just try to improve the existing slot.
                        pass
                    
                    # Decay step size
                    current_step *= 0.8

        return X
