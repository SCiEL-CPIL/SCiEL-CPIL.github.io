"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution (DE/rand/1/bin) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit, leave 10s
        
        dim = self.dim
        budget = self.budget
        
        # Population size heuristic
        pop_size = max(10, 10 * dim)
        
        # Initialize population uniformly in [-5, 5]
        rng = np.random.default_rng(42)
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fit = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        calls_made = pop_size
        F = 0.8  # Scaling factor
        CR = 0.9 # Crossover rate
        
        # Adaptive parameters logic:
        # If no improvement for a while, increase F to explore more, decrease CR to exploit?
        # Standard DE is quite robust. Let's add a simple stagnation check.
        stagnation_counter = 0
        no_improve_threshold = 50 # generations
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Determine how many full generations we can afford
            # We evaluate 1 offspring per individual in the population
            # So 1 generation costs pop_size evaluations
            # If remaining budget < pop_size, we do partial generation or stop
            if budget - calls_made < pop_size:
                # Do partial generation if we have enough for at least 1 eval
                # For simplicity, if we can't do a full gen, we might just stop 
                # or do a few random restarts if budget allows.
                # Given DE needs population context, stopping is safer for consistency.
                # But we can squeeze out remaining calls as random perturbations of best?
                # Let's just stop to avoid breaking DE logic.
                break

            # Generate offspring
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Differential mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Binary crossover
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate
                trial_f = func(trial)
                calls_made += 1
                
                # Selection
                if trial_f < fit[i]:
                    pop[i] = trial
                    fit[i] = trial_f
                    
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

            # Simple adaptation: If stagnating, increase F
            if stagnation_counter > no_improve_threshold:
                F = min(1.0, F + 0.1)
                stagnation_counter = 0
            else:
                # Slowly reset F towards 0.8 if improving? Or keep constant.
                # Keeping F constant is usually fine for 5D.
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
