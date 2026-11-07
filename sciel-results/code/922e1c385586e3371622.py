"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Stochastic Multi-Objective Evolutionary Strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Parameters
        pop_size = 10
        # Calculate generations to fit budget
        # Initial population: pop_size evals
        # Each generation: pop_size offspring evals
        # Total = pop_size + generations * pop_size <= budget
        generations = (self.budget - pop_size) // pop_size
        
        # Initialize population uniformly in bounds
        low = np.array([b[0] for b in bounds])
        high = np.array([b[1] for b in bounds])
        
        # Random initialization
        pop = np.random.uniform(low, high, size=(pop_size, self.dim))
        
        # Evaluate initial population
        F = np.array([problem(x) for x in pop])
        
        all_points = list(pop)
        all_f = list(F)
        
        for g in range(generations):
            # Create offspring via mutation (simple random perturbation)
            # We use a simple mutation: pick a random parent, add noise
            offspring = np.zeros_like(pop)
            for i in range(pop_size):
                # Pick a random parent
                parent_idx = np.random.randint(pop_size)
                # Simple mutation: perturb each dimension with some probability
                # Or just uniform recombination? Let's use simple random replacement for unexplored space
                # Better: Gaussian mutation around parent
                noise = np.random.normal(0, 0.1, size=self.dim)
                child = pop[parent_idx] + noise
                # Clip to bounds
                child = np.clip(child, low, high)
                offspring[i] = child
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            
            all_points.extend(offspring)
            all_f.extend(F_off)
            
            # Selection: Combine parents and offspring, keep best pop_size non-dominated
            # For simplicity in this "new approach" demo, we can just keep the union 
            # and select based on hypervolume contribution or simple dominance.
            # Given the budget constraints and the nature of ZDT1, a simple 
            # "keep best" based on distance to origin (proxy) or random retention 
            # is weak. 
            # Let's implement a simple non-dominated sort (NSGA-II lite) or just 
            # random selection to keep diversity?
            # Actually, to maximize hypervolume, we want points spread out.
            # A simple strategy: Keep the pop_size points that are non-dominated 
            # in the combined set (parents + offspring). If more than pop_size, 
            # pick randomly from the non-dominated set to maintain diversity.
            
            combined_pop = np.vstack([pop, offspring])
            combined_F = np.vstack([F, F_off])
            
            # Find non-dominated indices
            non_dominated_idx = []
            for i in range(len(combined_F)):
                dominated = False
                for j in range(len(combined_F)):
                    if i == j:
                        continue
                    # j dominates i if j <= i in all objectives and < in at least one
                    if np.all(combined_F[j] <= combined_F[i]) and np.any(combined_F[j] < combined_F[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dominated_idx.append(i)
            
            # If we have more non-dominated points than pop_size, select randomly
            if len(non_dominated_idx) > pop_size:
                selected_idx = np.random.choice(non_dominated_idx, size=pop_size, replace=False)
            else:
                selected_idx = non_dominated_idx
            
            # If we have fewer (shouldn't happen often with random init), fill with random new points?
            # Or just keep as is? The problem requires exactly budget evals.
            # My loop structure adds pop_size evals per generation.
            # I need to ensure the next 'pop' is formed.
            
            pop = combined_pop[selected_idx]
            F = combined_F[selected_idx]
            
        return None
# EVOLVE-BLOCK-END
