"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution (DE) for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        # DE parameters
        # Population size: small enough to fit in budget, large enough for diversity
        # dim=5, budget=500. 
        # If pop_size=10, we can do ~40 generations.
        # If pop_size=20, we can do ~20 generations.
        # Rastrigin has many local minima. DE is good at jumping between them.
        pop_size = 15
        
        # Initialize population uniformly in [-4, 4] (where minimizer lies)
        # Using [-4, 4] instead of [-5, 5] to avoid wasting calls on boundary clipping
        # if the true minimum is strictly inside.
        pop = rng.uniform(-4, 4, size=(pop_size, dim))
        fvals = np.array([func(x) for x in pop])
        budget_used = pop_size

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate

        # Adaptive F and CR or fixed? Fixed is often robust.
        # We will run until budget is exhausted or time runs out.

        while budget_used < budget:
            if time.time() > deadline:
                break

            # Generate next generation
            # Standard DE/rand/1/bin
            new_pop = []
            new_fvals = []

            # We might not be able to fill the whole population if budget is low
            # But budget is 500, pop is 15, so we have ~33 generations.
            # We process one individual at a time to allow early stopping if time is tight?
            # No, vectorizing the generation is faster, but we must check budget per individual.
            
            for i in range(pop_size):
                if budget_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Select 3 distinct individuals different from i
                # Efficient way: sample 3 indices, ensure they are not i
                # For small pop, just sample and retry or use permutation
                # Simple approach: sample 3 random indices
                candidates = rng.choice(pop_size, size=3, replace=False)
                # Ensure none is i
                while i in candidates:
                    candidates = rng.choice(pop_size, size=3, replace=False)
                
                a, b, c = candidates
                
                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds [-5, 5]
                v = np.clip(v, -5, 5)

                # Crossover
                j_rand = rng.integers(dim)
                trial = np.empty_like(pop[i])
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]

                # Evaluation
                f_trial = func(trial)
                budget_used += 1

                # Selection
                if f_trial <= fvals[i]:
                    new_pop.append(trial)
                    new_fvals.append(f_trial)
                    
                    # Update global best
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_pop.append(pop[i])
                    new_fvals.append(fvals[i])

            # If we broke out of the inner loop early due to budget/time, 
            # we might have a partial population. 
            # For DE, a partial population is problematic for the next generation's diversity.
            # However, if we are close to the end, it's fine.
            # If we broke early due to time, we exit the outer loop anyway.
            # If we broke early due to budget, we exit the outer loop anyway.
            # So we only reach here if a full generation was completed.
            
            if len(new_pop) < pop_size:
                break

            pop = np.array(new_pop)
            fvals = np.array(new_fvals)

            # Optional: Adaptive strategy adjustment
            # If progress is slow, increase F or CR?
            # For now, keep it simple and robust.

        return best_x, best_f
# EVOLVE-BLOCK-END
