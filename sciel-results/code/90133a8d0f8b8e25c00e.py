import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        remaining_budget = self.budget
        dim = self.dim
        
        # Helper to check time and budget
        def check_stop():
            return remaining_budget <= 0 or time.time() > deadline

        # Phase 1: Differential Evolution
        # Population size
        pop_size = max(10, 2 * dim) 
        # Initial population
        pop = np.random.uniform(-5, 5, (pop_size, dim))
        fvals = np.array([func(x) for x in pop])
        remaining_budget -= pop_size
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # DE parameters
        F = 0.5
        CR = 0.9
        
        # Reserve budget for local refinement
        local_budget = 100
        de_budget = max(0, self.budget - local_budget)
        
        while remaining_budget > local_budget and not check_stop():
            for i in range(pop_size):
                if remaining_budget <= local_budget or check_stop():
                    break
                
                # Select two distinct indices
                r1, r2 = np.random.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover
                trial = np.random.uniform(0, 1, dim) < CR
                if not np.any(trial):
                    trial[np.random.randint(dim)] = True
                u = np.where(trial, v, pop[i])
                
                # Bounds
                u = np.clip(u, -5, 5)
                
                # Evaluation
                fu = func(u)
                remaining_budget -= 1
                
                # Selection
                if fu <= fvals[i]:
                    pop[i] = u
                    fvals[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()

        # Phase 2: Nelder-Mead Refinement
        # Initialize simplex around best_x
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = best_x
        for i in range(dim):
            simplex[i+1] = best_x.copy()
            # Small step in each dimension
            step = 0.1
            simplex[i+1][i] += step
            simplex[i+1] = np.clip(simplex[i+1], -5, 5)
            
        fvals_nm = np.array([func(x) for x in simplex])
        remaining_budget -= (dim + 1)
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while remaining_budget > 0 and not check_stop():
            # Sort simplex
            order = np.argsort(fvals_nm)
            simplex = simplex[order]
            fvals_nm = fvals_nm[order]
            
            # Check convergence
            if np.ptp(fvals_nm) < 1e-12:
                break
                
            x_worst = simplex[-1]
            x_second = simplex[-2]
            x_best = simplex[0]
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            remaining_budget -= 1
            
            if f_r < fvals_nm[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                remaining_budget -= 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals_nm[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals_nm[-1] = f_r
            elif f_r < fvals_nm[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals_nm[-1] = f_r
            else:
                # Contraction
                if f_r < fvals_nm[-1]:
                    # Outer contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    remaining_budget -= 1
                    
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        fvals_nm[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals_nm[i] = func(simplex[i])
                            remaining_budget -= 1
                else:
                    # Inner contraction
                    x_c = centroid - rho * (centroid - x_worst)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    remaining_budget -= 1
                    
                    if f_c < fvals_nm[-1]:
                        simplex[-1] = x_c
                        fvals_nm[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals_nm[i] = func(simplex[i])
                            remaining_budget -= 1
            
            # Update best
            current_best_idx = np.argmin(fvals_nm)
            if fvals_nm[current_best_idx] < best_f:
                best_f = fvals_nm[current_best_idx]
                best_x = simplex[current_best_idx].copy()

        return best_x, best_f
