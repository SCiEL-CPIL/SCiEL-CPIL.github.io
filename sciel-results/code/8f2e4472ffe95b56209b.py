"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Differential Evolution (DE) with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42) # Fixed seed for internal consistency, external seed handles variation

        # Reference point for hypervolume calculation (used implicitly for pruning if needed, 
        # but here we maintain a non-dominated archive)
        
        # Population size
        # Budget is 200. 
        # Let's use a population size of 20.
        # 200 / 20 = 10 generations.
        # Each generation: 20 evaluations (offspring) + maybe some initial.
        # Actually, standard DE: 
        # Init: 20 evals.
        # Loop: for each individual in pop (20), generate 1 offspring (1 eval).
        # Total per gen: 20 evals.
        # 10 generations * 20 = 200 evals.
        # This fits perfectly.
        
        pop_size = 20
        if self.budget < pop_size * 2:
            pop_size = max(5, self.budget // 4)
            if pop_size < 3: pop_size = 3

        # Initialize population
        X_pop = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F_pop = np.array([problem(x) for x in X_pop])
        
        used = pop_size
        
        # Archive of non-dominated points (for final output and potentially pruning)
        # We will maintain the population itself as the archive, but we also need to 
        # ensure we return ALL non-dominated points found so far to maximize HV.
        # The problem statement says "Every returned objective vector is recorded".
        # The evaluator records the return value of __call__? 
        # "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # Wait, "return value of __call__ is ignored" usually means the object returned is ignored,
        # but "Every returned objective vector is recorded" implies that if we *return* vectors, they are recorded?
        # Let's re-read carefully: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This is slightly ambiguous. 
        # Interpretation A: The evaluator wraps `problem` and records every call to `problem`. The return value of `__call__` (the Optimizer) is ignored.
        # Interpretation B: The evaluator records the list of objective vectors returned by `__call__`.
        # "More than 200 evaluations... makes the whole result invalid". This strongly suggests Interpretation A: 
        # The evaluator counts calls to `problem(x)`.
        # However, "Every returned objective vector is recorded" might refer to the vectors returned by `problem(x)`.
        # Let's assume the standard contract: The evaluator intercepts `problem` calls.
        # The return value of `__call__` is ignored.
        # So we just need to make sure we don't exceed 200 calls to `problem`.
        
        # DE Parameters
        CR = 0.8  # Crossover rate
        F = 0.5   # Scaling factor (mutation weight)
        
        # Main Loop
        while used < self.budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            # We generate offspring one by one to respect the budget strictly.
            # Or we can do a full generation if we have enough budget.
            
            # Let's do individual-by-individual update to be safe with budget.
            # Pick a random index to update
            idx = rng.integers(0, pop_size)
            
            # Select 3 distinct random individuals
            # For small pop_size, ensure distinctness
            if pop_size < 4:
                # Fallback to simple random mutation if pop is too small for DE/rand/1
                r1 = idx
                r2 = idx
                r3 = idx
                # Just use random mutation
                v = X_pop[idx].copy()
                mask = rng.random(self.dim) < 0.5
                v[mask] = lo[mask] + (hi[mask] - lo[mask]) * rng.random(mask.sum())
            else:
                # DE/rand/1
                while True:
                    r1, r2, r3 = rng.integers(0, pop_size, 3)
                    if r1 != r2 and r2 != r3 and r3 != r1 and r1 != idx and r2 != idx and r3 != idx:
                        break
                
                # Mutation
                v = X_pop[r1] + F * (X_pop[r2] - X_pop[r3])
                
                # Crossover (Binomial)
                j_rand = rng.integers(0, self.dim)
                cross_mask = rng.random(self.dim) <= CR
                cross_mask[j_rand] = True # Ensure at least one gene is mutated
                
                # Create trial vector
                trial = np.where(cross_mask, v, X_pop[idx])
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
            
            # Evaluate trial
            f_trial = problem(trial)
            used += 1
            
            # Selection: Compare trial with target X_pop[idx]
            # If trial dominates target, replace.
            # If target dominates trial, keep target.
            # If they are non-dominated, we could keep both (increase pop?) or keep the one with better HV contribution?
            # Standard DE for MOEA (like MOEA/DE) often uses a selection based on non-domination.
            # To keep pop size fixed, we can use a simple rule:
            # If trial is non-dominated and target is dominated by trial, replace.
            # If both non-dominated, we might want to keep the one that is "better" in some sense.
            # A common approach is to keep the population non-dominated if possible, or use crowding distance.
            # Given the small budget, let's use a simple replacement strategy:
            # Replace if trial dominates target.
            # If non-dominated, replace with probability 0.5 (to maintain diversity) OR keep the one with lower "distance" to reference?
            # Let's try: If non-dominated, replace if random < 0.5. This maintains diversity.
            
            target_f = F_pop[idx]
            
            # Check dominance
            # Does trial dominate target?
            if np.all(f_trial <= target_f) and np.any(f_trial < target_f):
                X_pop[idx] = trial
                F_pop[idx] = f_trial
            # Does target dominate trial?
            elif np.all(target_f <= f_trial) and np.any(target_f < f_trial):
                pass # Keep target
            else:
                # Non-dominated
                # To prevent premature convergence and maintain diversity, 
                # we can replace with a certain probability, or use a metric.
                # Let's use a simple hypervolume-like proxy or just random.
                # Random replacement in non-dominated case helps exploration.
                if rng.random() < 0.5:
                    X_pop[idx] = trial
                    F_pop[idx] = f_trial

        # After the loop, we have a population of size `pop_size`.
        # However, the evaluator records ALL evaluations made to `problem`.
        # So the "recorded vectors" are all the `f` values returned by `problem` during the run.
        # We don't need to return anything specific, as the return value is ignored.
        # But we must ensure we didn't exceed the budget.
        
        return None
# EVOLVE-BLOCK-END
