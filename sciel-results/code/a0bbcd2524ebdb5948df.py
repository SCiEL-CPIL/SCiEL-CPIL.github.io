# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for BBOB f7 (Step ellipsoidal).
    
    DE is well-suited for non-smooth, non-convex landscapes like the step
    ellipsoidal function. It uses population-based search with mutation,
    crossover, and selection to explore the search space effectively.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Reset state
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def count_eval(x):
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # DE parameters
        # Population size: 5*dim is standard for DE
        pop_size = max(5 * dim, 10)
        # Mutation factor F: 0.5 is standard for DE/rand/1
        F = 0.5
        # Crossover rate CR: 0.5 is standard
        CR = 0.5

        # Initialize population uniformly in [-5, 5]^dim
        population = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.array([count_eval(x) for x in population])
        
        # Track best
        best_idx = np.argmin(fvals)
        best_x = population[best_idx].copy()
        best_f = fvals[best_idx]

        # DE loop
        while self._call_count + pop_size <= budget and time.time() < deadline:
            # Generate trial population
            trial_population = []
            for i in range(pop_size):
                # Select three distinct random indices
                r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                
                # Mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover: binomial crossover with rate CR
                crossover_mask = rng.random(dim) < CR
                # Ensure at least one dimension is taken from v
                if not np.any(crossover_mask):
                    j = rng.randint(dim)
                    crossover_mask[j] = True
                
                # Trial individual
                u = np.where(crossover_mask, v, population[i])
                u = np.clip(u, -5, 5)
                trial_population.append(u)
            
            # Evaluate trial population
            trial_fvals = np.array([count_eval(x) for x in trial_population])
            
            # Selection: keep the better of trial and original
            for i in range(pop_size):
                if trial_fvals[i] < fvals[i]:
                    population[i] = trial_population[i]
                    fvals[i] = trial_fvals[i]
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = population[i].copy()

            # Check for stagnation: if no improvement in last several generations,
            # restart with perturbation around best
            # We can track this by checking if best_f hasn't changed
            # For simplicity, we'll rely on DE's exploration capability

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
