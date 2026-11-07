import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        budget = self.budget
        dim = self.dim
        
        call_count = 0
        best_x = None
        best_f = np.inf
        
        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            try:
                val = func(x)
                if np.isfinite(val) and val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                return np.inf

        # 1. Differential Evolution (Global)
        # NP = 10, dim = 5
        # Budget for DE: 350 calls
        # Generations: 35
        np_pop = 10
        max_gen_de = 35
        F = 0.8
        CR = 0.9
        
        # Initialize population
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5, 5, (np_pop, dim))
        fit = np.array([safe_call(x) for x in pop])
        
        # Track best index
        best_idx = np.argmin(fit)
        best_x_de = pop[best_idx].copy()
        best_f_de = fit[best_idx]
        
        no_improve_gen = 0
        threshold = 5 # generations without improvement
        
        for g in range(max_gen_de):
            if time.time() > deadline or call_count >= budget:
                break
            
            improved_this_gen = False
            for i in range(np_pop):
                if time.time() > deadline or call_count >= budget:
                    break
                
                # Mutation: DE/rand/1
                r1, r2 = np.random.choice(np_pop, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = np.random.choice(np_pop, 2, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover: Binomial
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                f_trial = safe_call(trial)
                
                if f_trial < fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    improved_this_gen = True
                    if f_trial < best_f_de:
                        best_f_de = f_trial
                        best_x_de = trial.copy()
            
            if not improved_this_gen:
                no_improve_gen += 1
                if no_improve_gen > threshold:
                    break
            else:
                no_improve_gen = 0

        # 2. Nelder-Mead (Local)
        # Use best_x_de as start
        # Budget remaining: budget - call_count
        # NM typically needs ~10 * dim iterations for good convergence
        # We have ~150 calls left.
        
        x0 = best_x_de.copy()
        f0 = best_f_de
        
        # Initialize Simplex
        # Standard NM simplex
        simplex = [x0]
        for i in range(dim):
            x_temp = x0.copy()
            # Step size: 0.05 * (bound - x0[i]) or just 0.05 if x0 is not near bound
            step = 0.05
            if x0[i] > 4.95: step = -0.05
            elif x0[i] < -4.95: step = 0.05
            x_temp[i] += step
            simplex.append(np.clip(x_temp, -5, 5))
        
        # Evaluate initial simplex
        simplex_fits = [safe_call(s) for s in simplex]
        
        # NM Parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        max_iter_nm = 50 # Iterations, not calls. Each iter uses ~1-3 calls.
        
        for _ in range(max_iter_nm):
            if time.time() > deadline or call_count >= budget:
                break
            
            # Sort simplex
            order = np.argsort(simplex_fits)
            simplex = [simplex[i] for i in order]
            simplex_fits = [simplex_fits[i] for i in order]
            
            best_s = simplex[0]
            worst_s = simplex[-1]
            second_worst_s = simplex[-2]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst_s)
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_call(x_r)
            
            if f_r < simplex_fits[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_call(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_fits[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_fits[-1] = f_r
            elif f_r < simplex_fits[-2]:
                # Accept Reflection
                simplex[-1] = x_r
                simplex_fits[-1] = f_r
            else:
                # Contraction
                if f_r < simplex_fits[-1]:
                    # Outside Contraction
                    x_c = centroid + 0.5 * (x_r - centroid)
                else:
                    # Inside Contraction
                    x_c = centroid - 0.5 * (worst_s - centroid)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_call(x_c)
                
                if f_c < min(f_r, simplex_fits[-1]):
                    simplex[-1] = x_c
                    simplex_fits[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = best_s + sigma * (simplex[i] - best_s)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_fits[i] = safe_call(simplex[i])
            
            # Update global best
            current_best_idx = np.argmin(simplex_fits)
            if simplex_fits[current_best_idx] < best_f:
                best_f = simplex_fits[current_best_idx]
                best_x = simplex[current_best_idx].copy()

        return best_x, best_f
