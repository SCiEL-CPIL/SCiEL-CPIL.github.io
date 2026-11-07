"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        pop_size = max(10, 4 * dim)  # At least 10 or 4*dim
        F = 0.8  # Mutation strength
        CR = 0.5  # Crossover rate
        
        # Initialize population
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitness = np.zeros(pop_size)
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            calls_made += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation
        for i in range(pop_size):
            fitness[i] = safe_eval(population[i])
        
        # Sort initial population by fitness (optional for DE, but good for tracking)
        order = np.argsort(fitness)
        population = population[order]
        fitness = fitness[order]
        
        # Main DE loop
        while calls_made < budget:
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                
                # Select three distinct random indices different from i
                if pop_size < 4:
                    break
                    
                # Randomly select 3 distinct indices excluding i
                others = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(others, size=3, replace=False)
                
                # Mutation: v = population[a] + F * (population[b] - population[c])
                v = population[a] + F * (population[b] - population[c])
                
                # Crossover
                u = np.empty_like(v)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = population[i][j]
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                f_u = safe_eval(u)
                
                # Selection
                if f_u <= fitness[i]:
                    population[i] = u
                    fitness[i] = f_u
        
        return best_x, best_f
# EVOLVE-BLOCK-END
