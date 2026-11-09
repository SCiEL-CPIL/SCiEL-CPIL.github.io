"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Hybrid DE + Nelder-Mead optimizer for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Differential Evolution (Global Search)
        # Use 50% of budget for global exploration
        de_budget = budget // 2
        pop_size = 10  # Small population to allow more generations
        F = 0.8
        CR = 0.9
        
        # Initialize population
        population = rng.uniform(-5, 5, size=(pop_size, dim))
        fitness = np.array([safe_func(ind) for ind in population])
        
        if calls >= budget:
            return best_x, best_f
            
        best_idx = np.argmin(fitness)
        
        while calls < de_budget:
            # Select best individual
            x_best = population[best_idx]
            
            # Update population
            new_population = population.copy()
            new_fitness = fitness.copy()
            
            for i in range(pop_size):
                if calls >= de_budget:
                    break
                
                # Select three distinct random individuals
                while True:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    if r1 != i and r2 != i and r3 != i:
                        break
                
                # Mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                j_rand = rng.integers(dim)
                u = rng.random(dim)
                trial = np.where(u < CR, v, population[i])
                trial[j_rand] = v[j_rand]
                
                # Boundary handling (clip)
                trial = np.clip(trial, -5, 5)
                
                # Evaluation
                f_trial = safe_func(trial)
                
                if f_trial <= fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        # Update global best if needed, safe_func handles it
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
            
            population = new_population
            fitness = new_fitness
            
        # Phase 2: Nelder-Mead (Local Search)
        # Start from the best point found in DE
        x0 = best_x.copy()
        
        # Initialize Nelder-Mead simplex
        # Standard simplex: x0, x0 + delta*e_i
        delta = 0.5
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0
            simplex[i + 1][i] += delta
        
        # Evaluate simplex
        simplex_f = np.array([safe_func(s) for s in simplex])
        
        if calls >= budget:
            return best_x, best_f
            
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5   # Contraction
        sigma = 0.5 # Shrink
        
        while calls < budget:
            # Sort simplex by fitness
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            x_worst = simplex[-1]
            x_best_nm = simplex[0]
            x_second_worst = simplex[-2]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_func(x_r)
            
            if calls >= budget:
                break
                
            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_func(x_e)
                
                if calls >= budget:
                    break
                    
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
                # Outside contraction if f_r < f_worst, else inside
                if f_r < simplex_f[-1]:
                    x_c = centroid + 0.5 * (x_r - centroid)
                else:
                    x_c = centroid - 0.5 * (x_worst - centroid)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_func(x_c)
                
                if calls >= budget:
                    break
                
                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = x_best_nm + sigma * (simplex[i] - x_best_nm)
                        simplex_f[i] = safe_func(simplex[i])
                        
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
