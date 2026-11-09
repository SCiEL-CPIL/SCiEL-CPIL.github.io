"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution (DE) continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # DE Parameters
        pop_size = 20
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate

        # Domain
        lower = -5.0
        upper = 5.0

        # Initialize population
        positions = np.zeros((pop_size, n))
        for i in range(pop_size):
            if i < 5:
                # Corners
                positions[i] = rng.choice([lower, upper], n)
            elif i < 10:
                # Face centers
                positions[i] = rng.uniform(lower, upper, n)
                axis = i % n
                positions[i, axis] = 0.0
            else:
                # Random
                positions[i] = rng.uniform(lower, upper, n)
        
        # Evaluate initial population
        values = np.zeros(pop_size)
        calls_made = 0
        for i in range(pop_size):
            values[i] = func(positions[i])
            calls_made += 1

        # Find initial global best
        g_best_idx = np.argmin(values)
        g_best_pos = positions[g_best_idx].copy()
        g_best_val = values[g_best_idx]

        stagnation_counter = 0

        # Main loop
        while calls_made < budget:
            improved = False
            
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                
                # Select 3 distinct random indices different from i
                indices = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(indices, 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                mutant = positions[r1] + F * (positions[r2] - positions[r3])
                
                # Clip to domain
                mutant = np.clip(mutant, lower, upper)
                
                # Crossover: Binomial
                j_rand = rng.integers(n)
                trial = positions[i].copy()
                
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Evaluate trial
                trial_val = func(trial)
                calls_made += 1
                
                # Selection: Greedy
                if trial_val < values[i]:
                    positions[i] = trial
                    values[i] = trial_val
                    if trial_val < g_best_val:
                        g_best_val = trial_val
                        g_best_pos = trial.copy()
                        improved = True

            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            # Stagnation handling
            if stagnation_counter >= 5:
                # Restart worst individuals
                worst_indices = np.argsort(values)[-5:]
                for idx in worst_indices:
                    if calls_made >= budget:
                        break
                    positions[idx] = rng.uniform(lower, upper, n)
                    values[idx] = func(positions[idx])
                    calls_made += 1
                    if values[idx] < g_best_val:
                        g_best_val = values[idx]
                        g_best_pos = positions[idx].copy()
                stagnation_counter = 0

        return g_best_pos, g_best_val
# EVOLVE-BLOCK-END
