"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        n_calls = 0
        best_x = np.random.uniform(-5, 5, dim)
        best_fx = func(best_x)
        n_calls += 1
        
        def time_left():
            return deadline - time.time()
        
        # Phase 1: Differential Evolution with adaptive restarts for plateau handling
        # f7 has a step/plateau structure, so we need to detect convergence and expand search
        
        pop_size = 20
        F = 0.8  # Mutation scale
        CR = 0.5  # Crossover rate
        
        # Initialize population
        pop = np.random.uniform(-5, 5, (pop_size, dim))
        fvals = np.empty(pop_size)
        for i in range(pop_size):
            if n_calls >= budget or time_left() < 5:
                break
            fvals[i] = func(pop[i])
            n_calls += 1
            if fvals[i] < best_fx:
                best_fx = fvals[i]
                best_x = pop[i].copy()
        
        # Track convergence
        no_improve = 0
        best_fx_prev = best_fx
        sigma_expand = 1.0
        
        while n_calls < budget and time_left() > 5:
            # Check for improvement
            if best_fx < best_fx_prev - 1e-10:
                no_improve = 0
                best_fx_prev = best_fx
                sigma_expand = max(1.0, sigma_expand * 0.9)
            else:
                no_improve += 1
            
            # If stuck, expand search range (key for step functions)
            if no_improve > 30:
                # Expand the population around best_x with larger steps
                expand_scale = min(4.0, 0.5 * (1 + no_improve / 30.0))
                pop = best_x + np.random.randn(pop_size, dim) * expand_scale
                pop = np.clip(pop, -5, 5)
                no_improve = 0
                sigma_expand = 1.0
            
            # Differential Evolution step
            for i in range(pop_size):
                if n_calls >= budget or time_left() < 5:
                    break
                
                # Select two distinct indices different from i
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = np.random.choice(pop_size, 2, replace=False)
                
                # Mutation
                v = pop[i] + F * (pop[r1] - pop[r2])
                
                # Crossover
                j_rand = np.random.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                n_calls += 1
                
                # Selection
                if f_trial <= fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    if f_trial < best_fx:
                        best_fx = f_trial
                        best_x = trial.copy()
        
        # Phase 2: Local polish with Nelder-Mead if budget remains
        remaining = budget - n_calls
        if remaining > 20 and time_left() > 10:
            try:
                # Simple Nelder-Mead implementation to avoid scipy overhead
                x0 = best_x.copy()
                f0 = best_fx
                simplex = [x0]
                for j in range(dim):
                    xj = x0.copy()
                    xj[j] += 0.5 if x0[j] < 0 else -0.5
                    simplex.append(xj)
                    n_calls += 1
                    if n_calls >= budget:
                        break
                    simplex_f = [f0]
                    simplex_f.append(func(xj))
                    if simplex_f[-1] < f0:
                        f0 = simplex_f[-1]
                        x0 = xj
                        best_fx = f0
                        best_x = xj.copy()
                    else:
                        simplex_f.append(f0)
                
                # Simple iterations
                for _ in range(min(remaining - 2 * dim, 50)):
                    if n_calls >= budget or time_left() < 5:
                        break
                    # Sort simplex
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    alpha = 1.0
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = func(x_r)
                    n_calls += 1
                    if f_r < best_fx:
                        best_fx = f_r
                        best_x = x_r.copy()
                    
                    if simplex_f[0] <= f_r < simplex_f[-1]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    elif f_r < simplex_f[0]:
                        # Expansion
                        gamma = 2.0
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        n_calls += 1
                        if f_e < best_fx:
                            best_fx = f_e
                            best_x = x_e.copy()
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    else:
                        # Contraction
                        rho = 0.5
                        x_c = centroid + rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        n_calls += 1
                        if f_c < best_fx:
                            best_fx = f_c
                            best_x = x_c.copy()
                        if f_c < simplex_f[-1]:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for k in range(1, len(simplex)):
                                simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                                if n_calls < budget:
                                    simplex_f[k] = func(simplex[k])
                                    n_calls += 1
                                    if simplex_f[k] < best_fx:
                                        best_fx = simplex_f[k]
                                        best_x = simplex[k].copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
