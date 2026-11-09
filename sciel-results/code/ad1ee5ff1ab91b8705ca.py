import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 280  # 20s margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        def eval_point(x):
            nonlocal best_x, best_f, evals_used
            if evals_used >= budget:
                return best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Initialization (10 points)
        # Mix of center and random points to quickly identify basin
        init_candidates = []
        
        # Center
        x_center = np.zeros(dim)
        f_center = eval_point(x_center)
        init_candidates.append((f_center, x_center.copy()))
        
        # Random points
        for _ in range(9):
            x = rng.uniform(-5, 5, dim)
            f = eval_point(x)
            init_candidates.append((f, x.copy()))
        
        # Sort by fitness
        init_candidates.sort(key=lambda k: k[0])
        best_x = init_candidates[0][1].copy()
        best_f = init_candidates[0][0]
        
        # Phase 2: Differential Evolution (DE)
        # DE is robust for multimodal and step-like functions
        pop_size = 10
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population around best found
        # Use a mix of random and perturbed best
        population = []
        for i in range(pop_size):
            if i == 0:
                x = best_x.copy()
            else:
                # Spread out
                x = best_x + rng.uniform(-2, 2, dim)
                x = np.clip(x, -5, 5)
            f = eval_point(x)
            population.append((f, x.copy()))
        
        de_budget = 300
        de_evals = 0
        
        while de_evals < de_budget and evals_used < budget:
            if time.time() > deadline:
                break
                
            if budget - evals_used < pop_size:
                break
            
            new_population = []
            # Keep best from previous generation
            population.sort(key=lambda k: k[0])
            new_population.append(population[0])
            
            for i in range(1, pop_size):
                if evals_used >= budget or time.time() > deadline:
                    break
                    
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Differential mutation
                mutant = population[r1][1] + F * (population[r2][1] - population[r3][1])
                
                # Binary crossover
                trial = population[i][1].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                trial = np.clip(trial, -5, 5)
                f_trial = eval_point(trial)
                
                # Selection
                if f_trial <= population[i][0]:
                    new_population.append((f_trial, trial.copy()))
                else:
                    new_population.append(population[i])
                
                de_evals += 1
            
            # Update population
            population = new_population[:pop_size]
            
            # Update global best
            pop_best = min(population, key=lambda k: k[0])
            if pop_best[0] < best_f:
                best_f = pop_best[0]
                best_x = pop_best[1].copy()

        # Phase 3: Nelder-Mead Refinement
        # Use remaining budget for local refinement
        remaining = budget - evals_used
        if remaining > 5 and time.time() < deadline:
            # Initialize simplex
            x0 = best_x.copy()
            f0 = best_f
            
            # Simplex size
            delta = 0.1
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            # Evaluate simplex
            simplex_f = [f0]
            for i in range(1, dim + 1):
                if evals_used >= budget:
                    break
                f_i = eval_point(simplex[i])
                simplex_f.append(f_i)
            
            # Nelder-Mead iterations
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink
            
            max_nm_iters = 50
            for _ in range(max_nm_iters):
                if evals_used >= budget or time.time() > deadline:
                    break
                    
                # Sort simplex by fitness
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Check convergence
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                    
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-8:
                    break
                
                # Centroid of all but worst
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_centroid + alpha * (x_centroid - simplex[-1])
                f_r = eval_point(x_r)
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = x_centroid + gamma * (x_r - x_centroid)
                    f_e = eval_point(x_e)
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
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = x_centroid + rho * (x_r - x_centroid)
                    else:
                        # Inside contraction
                        x_c = x_centroid - rho * (x_centroid - simplex[-1])
                    
                    f_c = eval_point(x_c)
                    
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if evals_used < budget:
                                simplex_f[i] = eval_point(simplex[i])

        return best_x, best_f
