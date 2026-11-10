# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Initialization: Latin Hypercube Sampling (LHS)
        # Use a population size that balances exploration and exploitation.
        # With 200 budget, a pop of 20-30 allows for ~5-10 generations of DE.
        pop_size = 30
        if budget < pop_size:
            pop_size = max(1, budget)
            
        # Generate LHS points
        # For each dimension, we divide the range into pop_size bins and pick one point per bin.
        # This ensures that for each dimension, the population covers the entire range.
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            # Permutation ensures one point in each bin
            perm = np.random.permutation(pop_size)
            # Random offset within the bin
            u = (perm + np.random.random(pop_size)) / pop_size
            X[:, d] = lo[d] + (hi[d] - lo[d]) * u
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size

        # 2. Global Search: Differential Evolution (DE/rand/1/bin)
        # Parameters for DE
        CR = 0.8  # Crossover rate
        F_factor = 0.5  # Differential weight
        
        # We will iterate until budget is exhausted or time runs out
        # Each generation requires pop_size evaluations (one per individual)
        # So we can do roughly (budget - pop_size) / pop_size generations
        
        def dominates(a, b):
            """True if a dominates b (a <= b in all objectives and a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F_arr):
            """Return indices of non-dominated solutions."""
            n = len(F_arr)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(i + 1, n):
                    if not nd[j]:
                        continue
                    if dominates(F_arr[i], F_arr[j]):
                        nd[j] = False
                    elif dominates(F_arr[j], F_arr[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        # DE Loop
        while used < budget - 20: # Leave 20 budget for local refinement
            if time.time() > deadline - 5:
                break
                
            # Generate trial vectors for the population
            trial_X = np.zeros_like(X)
            trial_F = np.zeros_like(F)
            
            for i in range(pop_size):
                if time.time() > deadline - 5:
                    break
                
                # Select 3 distinct random indices
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Differential mutation
                v = X[r1] + F_factor * (X[r2] - X[r3])
                
                # Binary Crossover
                j_rand = np.random.randint(dim)
                trial = X[i].copy()
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                f_trial = problem(trial)
                used += 1
                trial_X[i] = trial
                trial_F[i] = f_trial
                
                # Selection: Replace if trial dominates or is not dominated by current
                # In multi-objective, we usually keep both if non-dominated, but for simplicity
                # in a population-based DE without explicit archive, we can use a tournament
                # or simple replacement if it's better in a scalar sense or non-dominated.
                # A common strategy is to keep the non-dominated set in the population.
                # Here, we use a simple rule: if trial is better than current in all objectives, replace.
                # If they are non-dominated, we might keep both, but since we have fixed pop size,
                # we need a selection pressure.
                
                # Let's use a slightly more robust selection:
                # If trial dominates current, replace.
                # If current dominates trial, keep current.
                # If non-dominated, replace with some probability (e.g., 0.5) to maintain diversity.
                
                if dominates(trial_F[i], F[i]):
                    X[i] = trial_X[i]
                    F[i] = trial_F[i]
                elif not dominates(F[i], trial_F[i]):
                    # Non-dominated: random choice to maintain diversity
                    if np.random.random() < 0.5:
                        X[i] = trial_X[i]
                        F[i] = trial_F[i]

        # 3. Local Refinement: Nelder-Mead on non-dominated points
        # Use the remaining budget to polish the best points.
        remaining_budget = budget - used
        if remaining_budget > 0 and time.time() < deadline - 10:
            nd_idx = non_dominated_sort(F)
            
            # If we have many non-dominated points, select a subset to refine
            # Nelder-Mead is expensive, so we limit the number of runs.
            # Each run takes roughly dim * 10 evaluations.
            nm_cost_per_run = dim * 10
            max_nm_runs = max(1, remaining_budget // nm_cost_per_run)
            
            if len(nd_idx) > max_nm_runs:
                # Select random subset of non-dominated points
                refine_idx = np.random.choice(nd_idx, size=max_nm_runs, replace=False)
            else:
                refine_idx = nd_idx
                
            for idx in refine_idx:
                if used >= budget or time.time() > deadline - 5:
                    break
                
                # Scalarize objectives using random weights
                weights = np.random.random(n_obj)
                weights /= np.sum(weights)
                
                def objective(x):
                    if used >= budget:
                        return np.inf
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=float)
                    used += 1
                    return np.dot(f, weights)
                
                x0 = X[idx].copy()
                
                try:
                    result = minimize(objective, x0, method='Nelder-Mead', 
                                      options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4})
                    if result.success and used < budget:
                        x_new = np.clip(result.x, lo, hi)
                        f_new = np.array(problem(x_new), dtype=float)
                        used += 1
                        
                        # Update population if the new point is better
                        # Check if it dominates the current point
                        if dominates(f_new, F[idx]):
                            X[idx] = x_new
                            F[idx] = f_new
                        else:
                            # If non-dominated, we might want to add it to the archive,
                            # but since we return the whole population, we can just
                            # replace if it's "better" in some scalar sense or keep it.
                            # For simplicity, if it doesn't dominate, we might still keep it
                            # if it's in a different region of the Pareto front.
                            # However, replacing a non-dominated point with another non-dominated
                            # point is tricky without an archive.
                            # Let's just keep the better one in terms of sum of objectives
                            # if they are non-dominated, to ensure some progress.
                            if np.sum(f_new) < np.sum(F[idx]):
                                X[idx] = x_new
                                F[idx] = f_new
                except Exception:
                    pass

        return X
# EVOLVE-BLOCK-END
