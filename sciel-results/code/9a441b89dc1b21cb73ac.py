"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution (DE) optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s/330s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # DE Parameters
        pop_size = 20  # Small population to fit within 500 calls while maintaining diversity
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate

        # Initialize population
        # Uniform initialization in [-5, 5]
        pop = rng.uniform(-5, 5, size=(pop_size, n))
        fit = np.full(pop_size, np.inf)
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = func(x)
            calls_made += 1
            return fx

        # Initial evaluation of population
        for i in range(pop_size):
            fit[i] = evaluate(pop[i])
            if fit[i] < best_f:
                best_f = fit[i]
                best_x = pop[i].copy()

        # Main DE Loop
        # We will run generations until budget or time is exhausted.
        # Each generation costs pop_size calls.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            # If not, we might do partial updates or just stop. 
            # For simplicity and robustness, we stop if we can't afford a full gen 
            # OR we can just let the inner loop handle the call limit.
            # The inner loop will break if calls_made >= budget.
            
            # Generate offspring
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select two distinct random indices different from i
                # Simple approach: sample without replacement
                r1, r2 = rng.choice(pop_size, size=2, replace=False)
                while r1 == i:
                    r1 = rng.choice(pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.choice(pop_size)

                # Mutation: v = x[r1] + F * (x[r2] - x[i])
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)

                # Crossover
                # Binary crossover with CR
                # jrand: random index where we force x[i] to be taken
                jrand = rng.integers(0, n)
                trial = np.empty_like(pop[i])
                for j in range(n):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Ensure trial is within bounds (though v was clipped, pop[i] is too, so trial is too)
                # But just in case of any floating point weirdness or if we changed logic later
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                f_trial = evaluate(trial)
                
                # Selection
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
