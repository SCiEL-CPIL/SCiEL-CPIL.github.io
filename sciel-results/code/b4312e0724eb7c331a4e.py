"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid DE + Nelder-Mead optimizer for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        budget = self.budget
        dim = self.dim
        
        # Count function calls
        call_count = [0]
        
        def wrapped_func(x):
            call_count[0] += 1
            x = np.clip(x, -5, 5)
            return func(x)
        
        # Phase 1: Differential Evolution for global exploration
        # Use about 60% of budget for DE
        de_budget = int(budget * 0.6)
        pop_size = max(10, dim * 4)  # Population size
        
        # Initialize population
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5, 5, (pop_size, dim))
        pop_f = np.array([wrapped_func(ind) for ind in pop])
        
        # DE parameters
        F = 0.8  # Mutation scale
        CR = 0.9  # Crossover rate
        
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        
        # DE loop
        while call_count[0] < de_budget:
            if time.time() > deadline - 50:  # Leave time for NM
                break
                
            for i in range(pop_size):
                if call_count[0] >= de_budget:
                    break
                    
                # Select three distinct random individuals
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation: DE/rand/1
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover: Binomial
                j_rand = np.random.randint(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                trial = np.clip(trial, -5, 5)
                trial_f = wrapped_func(trial)
                
                # Selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
        
        # Phase 2: Nelder-Mead local refinement
        nm_budget = budget - call_count[0]
        if nm_budget > 10 and time.time() < deadline - 20:
            # Initialize NM simplex around best point
            simplex = [best_x.copy()]
            for j in range(dim):
                point = best_x.copy()
                step = 0.1 * (best_x[j] + 0.1) if best_x[j] >= 0 else 0.1 * (best_x[j] - 0.1)
                point[j] += step
                point = np.clip(point, -5, 5)
                simplex.append(point)
            
            simplex_f = [wrapped_func(pt) for pt in simplex]
            
            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            max_nm_iter = min(nm_budget, 500)
            for _ in range(max_nm_iter):
                if call_count[0] >= budget or time.time() > deadline - 10:
                    break
                
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = wrapped_func(x_c)
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = wrapped_func(simplex[i])
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = wrapped_func(x_c)
                        if f_c < worst_f:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex_f[i] = wrapped_func(simplex[i])
                
                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()
        
        # Final check
        if call_count[0] < budget and time.time() < deadline - 5:
            # Try a few more random restarts around best
            for _ in range(min(20, budget - call_count[0])):
                if time.time() > deadline - 2:
                    break
                x_try = best_x + 0.01 * rng.standard_normal(dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = wrapped_func(x_try)
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try
        
        return best_x, best_f
# EVOLVE-BLOCK-END
