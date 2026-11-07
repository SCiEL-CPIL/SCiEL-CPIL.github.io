"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with random restarts and adaptive exploration."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # We'll use a combination of global random search to find good basins
        # and then local refinement with L-BFGS-B.
        
        # Strategy:
        # 1. Use a portion of the budget for random sampling to find good starting points.
        # 2. Use the remaining budget for local optimization from the best points.
        
        # Since we don't know the cost of func calls exactly in terms of L-BFGS-B steps,
        # we need to be careful. L-BFGS-B can use many function evaluations.
        
        # Alternative: Use a derivative-free method like Powell or Nelder-Mead which
        # has more predictable function evaluation counts, or limit the iterations.
        
        # Let's try a hybrid approach:
        # - Phase 1: Random search with a small number of points to get an initial guess.
        # - Phase 2: Use Nelder-Mead or Powell for local refinement.
        
        # Actually, since the problem is smooth (Schwefel-like but potentially with hidden transformations),
        # gradient-based methods might struggle if the landscape is noisy or non-convex.
        # However, the task says "Schwefel" which is typically non-convex and multimodal.
        
        # A robust approach for multimodal continuous optimization with a fixed budget:
        # 1. Use a population-based method like CMA-ES or a simple evolutionary strategy.
        # 2. Or use multi-start local search.
        
        # Given the constraints (500 calls, 5D), let's try a multi-start approach with
        # Nelder-Mead, which is derivative-free and robust.
        
        # We'll allocate some calls to random sampling and the rest to local optimization.
        
        # Let's say we use 100 calls for random sampling to find the best point,
        # then use the remaining 400 calls for local optimization from that point.
        
        # But Nelder-Mead can also be expensive. Let's limit the number of iterations.
        
        # Better idea: Use a simple evolutionary strategy with a population size of 10
        # and 50 generations (10*50 = 500 calls). This is a standard approach for
        # black-box optimization.
        
        # Let's implement a simple (10+1)-ES or (10,10)-ES.
        
        # Actually, let's use a more sophisticated approach:
        # 1. Start with a random population of size 10.
        # 2. For each generation, evaluate the population.
        # 3. Keep the best 10 individuals.
        # 4. Generate offspring by adding Gaussian noise to the best individuals.
        # 5. Repeat until budget is exhausted.
        
        # This is a simple evolutionary strategy that should work well for this problem.
        
        # Let's implement this.
        
        pop_size = 10
        num_generations = budget // pop_size  # 50 generations
        
        # Initialize population
        population = rng.uniform(-5, 5, (pop_size, dim))
        fitnesses = np.array([func(x) for x in population])
        
        best_x = population[np.argmin(fitnesses)]
        best_f = np.min(fitnesses)
        
        # Adaptive step size
        sigma = 1.0
        
        for gen in range(num_generations):
            # Sort by fitness
            order = np.argsort(fitnesses)
            population = population[order]
            fitnesses = fitnesses[order]
            
            # Keep the best half
            n_keep = pop_size // 2
            parents = population[:n_keep]
            
            # Generate offspring
            offspring = []
            for _ in range(pop_size):
                parent = parents[rng.integers(n_keep)]
                child = parent + sigma * rng.standard_normal(dim)
                child = np.clip(child, -5, 5)
                offspring.append(child)
            
            offspring = np.array(offspring)
            offspring_fitnesses = np.array([func(x) for x in offspring])
            
            # Combine parents and offspring
            population = np.vstack([population, offspring])
            fitnesses = np.concatenate([fitnesses, offspring_fitnesses])
            
            # Select best pop_size individuals
            order = np.argsort(fitnesses)
            population = population[order][:pop_size]
            fitnesses = fitnesses[order][:pop_size]
            
            # Update best
            if fitnesses[0] < best_f:
                best_f = fitnesses[0]
                best_x = population[0]
            
            # Adapt step size based on success rate
            # Simple adaptation: if we made progress, increase sigma; else decrease
            if fitnesses[0] < fitnesses[-1]:
                sigma *= 1.1
            else:
                sigma *= 0.9
            
            sigma = np.clip(sigma, 0.1, 5.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
