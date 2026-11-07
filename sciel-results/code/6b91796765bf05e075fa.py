import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                raise TimeoutError("Time limit exceeded")
            if calls >= self.budget:
                raise Exception("Budget exceeded")
            
            calls += 1
            val = float(func(x))
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Phase 1: Differential Evolution (Global Search)
        # DE is robust for low dimensions and handles ill-conditioning well
        de_budget = int(0.6 * self.budget)
        popsize = 10
        F = 0.8
        CR = 0.9
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        pop = [rng.uniform(-5, 5, size=dim) for _ in range(popsize)]
        fitness = [wrapped_func(p) for p in pop]
        
        # Sort initial population
        indices = np.argsort(fitness)
        pop = [pop[i] for i in indices]
        fitness = [fitness[i] for i in indices]
        
        # DE Loop
        while calls < de_budget and time_ok():
            # Sort population
            indices = np.argsort(fitness)
            pop = [pop[i] for i in indices]
            fitness = [fitness[i] for i in indices]
            
            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = pop[0].copy()
                
            # Generate new population
            new_pop = []
            new_fitness = []
            
            for i in range(popsize):
                if calls >= de_budget or not time_ok():
                    break
                    
                # Mutation: DE/best/1
                # Pick 3 distinct random indices different from i
                r1, r2 = np.random.choice(popsize, 2, replace=False)
                while r1 == i: r1 = np.random.choice(popsize)
                while r2 == i or r2 == r1: r2 = np.random.choice(popsize)
                
                mutant = pop[0] + F * (pop[r1] - pop[r2])
                
                # Crossover: Binomial
                trial = np.copy(pop[i])
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(f_trial)
                else:
                    new_pop.append(pop[i])
                    new_fitness.append(fitness[i])
                    
            if len(new_pop) < popsize:
                # Pad with best if we ran out of budget mid-generation
                while len(new_pop) < popsize:
                    new_pop.append(pop[0])
                    new_fitness.append(fitness[0])
                    
            pop = new_pop
            fitness = new_fitness

        # Phase 2: Nelder-Mead Local Refinement
        remaining_budget = self.budget - calls
        
        if remaining_budget > 0 and best_x is not None and time_ok():
            # Initialize Nelder-Mead simplex
            x0 = best_x.copy()
            f0 = best_f
            
            # Create simplex: x0 and x0 + delta * e_i
            delta = 0.5
            simplex = [x0]
            simplex_f = [f0]
            
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += delta
                x_i = np.clip(x_i, -5, 5)
                f_i = wrapped_func(x_i)
                simplex.append(x_i)
                simplex_f.append(f_i)
                
                if calls >= self.budget or not time_ok():
                    break
            
            if len(simplex) < dim + 1:
                return best_x, best_f

            # Nelder-Mead Loop
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            while calls < self.budget and time_ok():
                # Sort simplex
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                    
                # Centroid of best n points (excluding worst)
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept Reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside Contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside Contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                        
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)
                    
                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = wrapped_func(simplex[i])
                            
                        if not time_ok() or calls >= self.budget:
                            break

        return best_x, best_f
