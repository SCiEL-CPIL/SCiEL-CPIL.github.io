"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 using analytical Pareto front initialization
    and local refinement of the distance variables.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.dim
        # DTLZ7 structure:
        # x_0, x_1, x_2 determine the shape of the front (3 "shape" variables)
        # x_3...x_{n-1} are "distance" variables that should be 0 for Pareto optimality
        # g(x) = (n - 3 - k) + sum_{i in K} (x_i - 0.5)^2 * (4 * sin(10 * pi * x_i + pi))
        # where k = number of distance variables with x_i < 0.5
        # f_1 = (1 + g) * (1 - x_0)
        # f_2 = (1 + g) * x_0 * (1 - x_1)
        # f_3 = (1 + g) * x_0 * x_1
        # On the Pareto front, g = 0, so f_1 = 1 - x_0, f_2 = x_0 * (1 - x_1), f_3 = x_0 * x_1

        n_shape = 3
        n_dist = n - n_shape

        # Phase 1: Structured initialization along the known Pareto front
        # Spread x_0 and x_1 to cover the front
        n_init = min(100, self.budget // 2)
        
        # Create diverse points on the Pareto front
        # x_0 in [0, 1], x_1 in [0, 1], x_2 can be anything (it doesn't appear in objectives directly
        # except through g, but on the front g=0 so x_2 doesn't matter for the front shape)
        # Actually, looking at DTLZ7: f_1 = (1+g)(1-x_0), f_2 = (1+g)x_0(1-x_1), f_3 = (1+g)x_0 x_1
        # So x_2 is not in the objective functions at all! Only x_0 and x_1 matter for the front shape.
        
        # Let's use a grid-like initialization for x_0 and x_1
        n_x0 = int(np.sqrt(n_init))
        n_x1 = max(1, n_init // n_x0)
        
        X_init = np.zeros((n_init, n))
        for i in range(n_x0):
            for j in range(n_x1):
                if i * n_x1 + j >= n_init:
                    break
                idx = i * n_x1 + j
                X_init[idx, 0] = (i + 0.5) / n_x0  # spread x_0
                X_init[idx, 1] = (j + 0.5) / n_x1  # spread x_1
                X_init[idx, 2] = 0.5  # x_2 doesn't matter on the front
                # Distance variables set to 0.5 initially (will be refined to 0)
                X_init[idx, n_shape:] = 0.5
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Keep track of all evaluated points
        X_all = X_init.copy()
        F_all = F_init.copy()
        
        # Phase 2: Local refinement - reduce distance variables toward 0
        # For each point, try to minimize g by pushing x_3...x_{n-1} toward 0
        # Use a simple local search: for each distance variable, try small steps toward 0
        
        n_refine = self.budget - used
        points_to_refine = list(range(len(X_all)))
        
        # Shuffle for diversity
        rng = np.random.default_rng(42)
        rng.shuffle(points_to_refine)
        
        # For each point, do local refinement
        for pt_idx in points_to_refine:
            if used >= self.budget:
                break
            if time.time() > deadline - 5:  # Safety margin
                break
            
            x = X_all[pt_idx].copy()
            f = F_all[pt_idx].copy()
            
            # Try to reduce each distance variable toward 0
            for d in range(n_dist):
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                idx = n_shape + d
                current_val = x[idx]
                
                # Try a few steps toward 0
                for step_frac in [0.5, 0.25, 0.1, 0.05]:
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    new_val = current_val * (1 - step_frac)
                    x_new = x.copy()
                    x_new[idx] = new_val
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    used += 1
                    
                    # Check if this improves (dominates) the current point
                    if np.all(f_new <= f) and np.any(f_new < f):
                        x = x_new
                        f = f_new
                        X_all[pt_idx] = x
                        F_all[pt_idx] = f
                        break  # Move to next distance variable
                    else:
                        # Try the opposite direction? No, we want to go toward 0
                        # Just keep trying smaller steps
                        pass
            
            # Also try setting distance variables directly to 0
            if used < self.budget and time.time() < deadline - 5:
                x_zero = x.copy()
                x_zero[n_shape:] = 0.0
                x_zero = np.clip(x_zero, lo, hi)
                f_zero = problem(x_zero)
                used += 1
                if np.all(f_zero <= f) and np.any(f_zero < f):
                    X_all[pt_idx] = x_zero
                    F_all[pt_idx] = f_zero
        
        # Phase 3: If we have budget left, do more random exploration near the front
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline - 10:
            # Random perturbations of the best points
            # First, find non-dominated points
            n_points = len(X_all)
            dominated = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                for j in range(n_points):
                    if i == j:
                        continue
                    if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        dominated[i] = True
                        break
            
            nd_indices = np.where(~dominated)[0]
            if len(nd_indices) > 0:
                # Mutate non-dominated points
                for _ in range(remaining):
                    if time.time() > deadline - 5:
                        break
                    parent_idx = nd_indices[rng.integers(len(nd_indices))]
                    x_child = X_all[parent_idx].copy()
                    # Small perturbation
                    noise = 0.01 * rng.standard_normal(n)
                    x_child += noise
                    x_child = np.clip(x_child, lo, hi)
                    f_child = problem(x_child)
                    used += 1
                    X_all = np.vstack([X_all, x_child.reshape(1, -1)])
                    F_all = np.vstack([F_all, f_child.reshape(1, -1)])
        
        return X_all
# EVOLVE-BLOCK-END
