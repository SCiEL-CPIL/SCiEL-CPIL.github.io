"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DE + Nelder-Mead Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300
        
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # --- Phase 1: Differential Evolution ---
        # Use 60% of budget for global search
        phase1_budget = int(budget * 0.6)
        np_size = 10  # Population size
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = np.random.uniform(-5, 5, (np_size, dim))
        pop_f = np.array([evaluate(x) for x in pop])
        
        best_idx = np.argmin(pop_f)
        
        generations = phase1_budget // np_size
        for g in range(generations):
            if time.time() > deadline:
                break
            if calls >= phase1_budget:
                break
                
            for i in range(np_size):
                if calls >= phase1_budget:
                    break
                    
                # Select 3 distinct random individuals
                r1, r2, r3 = np.random.choice([j for j in range(np_size) if j != i], 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, v, pop[i])
                trial[j_rand] = v[j_rand]
                
                # Selection
                trial_f = evaluate(trial)
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < pop_f[best_idx]:
                        best_idx = i

        # --- Phase 2: Nelder-Mead ---
        # Use remaining budget for local refinement
        remaining_budget = budget - calls
        
        if remaining_budget > 0 and best_x is not None:
            # Initialize simplex
            # Start with best point and small perturbations
            alpha = 0.1 * 10  # Initial step size relative to box size? 
            # Better: use a small fraction of the box size, e.g., 0.5
            alpha = 0.5
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x
            for i in range(1, dim + 1):
                simplex[i] = best_x.copy()
                simplex[i][i-1] += alpha
                # Ensure within bounds
                simplex[i] = np.clip(simplex[i], -5, 5)
            
            simplex_f = np.array([evaluate(x) for x in simplex])
            
            # Nelder-Mead Parameters
            alpha_nm = 1.0  # Reflection
            gamma_nm = 2.0  # Expansion
            rho_nm = 0.5    # Contraction
            sigma_nm = 0.5  # Shrink
            
            while calls < budget:
                if time.time() > deadline:
                    break
                
                # Sort simplex by fitness
                sort_idx = np.argsort(simplex_f)
                simplex = simplex[sort_idx]
                simplex_f = simplex_f[sort_idx]
                
                # Check for convergence (optional, but helps stop early if stuck)
                if np.std(simplex_f) < 1e-10:
                    break
                    
                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = x_centroid + alpha_nm * (x_centroid - x_worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = evaluate(x_ref)
                
                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = x_centroid + gamma_nm * (x_ref - x_centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = evaluate(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    x_con = x_centroid + rho_nm * (x_worst - x_centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = evaluate(x_con)
                    if f_con < simplex_f[-1]:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma_nm * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = evaluate(simplex[i])
                        # Re-sort after shrink
                        sort_idx = np.argsort(simplex_f)
                        simplex = simplex[sort_idx]
                        simplex_f = simplex_f[sort_idx]

        return best_x, best_f

# EVOLVE-BLOCK-END
