import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        rng = np.random.default_rng(42)
        
        # Phase 1: Differential Evolution for global exploration
        pop_size = 12
        F = 0.8  # Mutation scale
        CR = 0.7  # Crossover rate
        
        # Initialize population uniformly in the box
        pop = rng.uniform(-5.0, 5.0, (pop_size, dim))
        pop_f = np.array([func(x) for x in pop])
        calls_made += pop_size
        
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_fx = pop_f[best_idx]
        
        # DE parameters
        de_budget = int(budget * 0.85)
        no_improve_count = 0
        max_no_improve = 50  # Generations without improvement before adapting
        
        while calls_made < de_budget:
            if time.time() > deadline - 15:
                break
            
            # Check for stagnation
            if pop_f[best_idx] < best_fx - 1e-10:
                best_fx = pop_f[best_idx]
                best_x = pop[best_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Adapt F and CR based on stagnation
            if no_improve_count > max_no_improve:
                F = np.clip(F * 0.8, 0.3, 1.5)
                CR = np.clip(CR * 0.9, 0.3, 0.9)
                no_improve_count = 0
            
            # Generate trial vectors
            for i in range(pop_size):
                if calls_made >= de_budget:
                    break
                
                # Select two distinct random indices different from i
                while True:
                    r1 = rng.integers(0, pop_size)
                    r2 = rng.integers(0, pop_size)
                    if r1 != i and r2 != i and r1 != r2:
                        break
                
                # Mutation: DE/rand/1
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover: binomial
                j_rand = rng.integers(0, dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial
                    if f_trial < best_fx:
                        best_fx = f_trial
                        best_x = trial.copy()
        
        # Phase 2: Nelder-Mead Polish
        # Use remaining budget for local refinement
        nm_budget = budget - calls_made
        
        if nm_budget > 0 and time.time() < deadline - 5:
            # Initialize simplex around best_x
            simplex = [best_x.copy()]
            simplex_f = [best_fx]
            
            # Create vertices by perturbing each dimension
            for i in range(dim):
                v = best_x.copy()
                # Perturbation size relative to current sigma or fixed small value
                pert = 0.1
                if v[i] + pert > 5.0:
                    v[i] -= pert
                else:
                    v[i] += pert
                simplex.append(v)
                fv = func(v)
                calls_made += 1
                simplex_f.append(fv)
                
                if calls_made >= budget:
                    break
            
            if len(simplex) == dim + 1:
                # Run Nelder-Mead
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5 # Shrink factor for contraction
                
                while calls_made < budget:
                    if time.time() > deadline - 2:
                        break
                    
                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Centroid of all but the worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflect
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5.0, 5.0)
                    f_r = func(x_r)
                    calls_made += 1
                    
                    if f_r < simplex_f[0]:
                        # Expand
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5.0, 5.0)
                        if calls_made < budget:
                            f_e = func(x_e)
                            calls_made += 1
                            if f_e < f_r:
                                simplex[-1] = x_e
                                simplex_f[-1] = f_e
                            else:
                                simplex[-1] = x_r
                                simplex_f[-1] = f_r
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Contract
                        if f_r < simplex_f[-1]:
                            # Inner contraction
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            # Outer contraction
                            x_c = centroid - rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        
                        if calls_made < budget:
                            f_c = func(x_c)
                            calls_made += 1
                            if f_c < simplex_f[-1]:
                                simplex[-1] = x_c
                                simplex_f[-1] = f_c
                            else:
                                # Shrink
                                for i in range(1, len(simplex)):
                                    simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                    if calls_made < budget:
                                        simplex_f[i] = func(simplex[i])
                                        calls_made += 1
                                    else:
                                        break
                                if calls_made >= budget:
                                    break
                    
                    # Update best
                    current_best_idx = np.argmin(simplex_f)
                    if simplex_f[current_best_idx] < best_fx:
                        best_fx = simplex_f[current_best_idx]
                        best_x = simplex[current_best_idx].copy()
        
        return best_x, best_fx
