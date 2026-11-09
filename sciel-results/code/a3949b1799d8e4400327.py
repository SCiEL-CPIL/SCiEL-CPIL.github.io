"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        # Standard DE uses NP ~ 5*dim to 10*dim. For dim=5, NP=25 is a good balance.
        # We need to fit within budget. 
        # Let's aim for roughly 20 generations if possible, or just fill budget.
        # If NP=25, 500/25 = 20 generations. This is reasonable.
        pop_size = 25
        
        # Ensure we don't exceed budget with initial population
        if pop_size >= budget:
            pop_size = max(4, budget // 2) # Fallback for small budgets

        # Initialize population uniformly in [-5, 5]
        population = rng.uniform(-5, 5, size=(pop_size, dim))
        fitnesses = np.array([func(x) for x in population])
        calls_made = pop_size
        
        # Track best
        best_idx = np.argmin(fitnesses)
        best_x = population[best_idx].copy()
        best_f = fitnesses[best_idx]
        
        # DE Parameters
        F = 0.8  # Scale factor
        CR = 0.9 # Crossover rate
        
        # Main loop
        while calls_made < budget:
            # Generate offspring for each individual
            new_population = []
            new_fitnesses = []
            
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                    
                # Select 3 distinct random indices different from i
                # For small pop_size, this might be tricky, but pop_size >= 4 usually
                while True:
                    j = rng.integers(0, pop_size)
                    k = rng.integers(0, pop_size)
                    l = rng.integers(0, pop_size)
                    if j != i and k != i and l != i and j != k and k != l and j != l:
                        break
                
                # Differential mutation
                v = population[i] + F * (population[j] - population[k])
                
                # Binomial Crossover
                trial = population[i].copy()
                r = rng.random()
                # Random dimension to ensure at least one gene is taken from v
                j_rand = rng.integers(0, dim)
                
                for d in range(dim):
                    if rng.random() < CR or d == j_rand:
                        trial[d] = v[d]
                    
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection (Greedy)
                if f_trial <= fitnesses[i]:
                    new_population.append(trial)
                    new_fitnesses.append(f_trial)
                    # Update global best
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_population.append(population[i])
                    new_fitnesses.append(fitnesses[i])
            
            if not new_population:
                break
                
            # Update population
            population = np.array(new_population)
            fitnesses = np.array(new_fitnesses)
            
            # Optional: Restart if stuck? 
            # For this simple task, standard DE should suffice. 
            # But if we detect stagnation, we might increase F or reseed.
            # Given the budget is small (500) and dim=5, standard DE is likely enough.
            # Let's add a simple stagnation check: if no improvement in last 2 generations, increase F slightly.
            # Actually, simpler is better for robustness. Stick to standard DE.

        return best_x, best_f
# EVOLVE-BLOCK-END
