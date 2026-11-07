"""Optimizer for BBOB f7 (Step ellipsoidal) using Differential Evolution with adaptive step size."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters for Step ellipsoidal f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        d = self.dim
        max_calls = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def elapsed_ok():
            return time.time() < deadline and calls_made < max_calls

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial population sampling for broad coverage
        # Use a moderate population size for DE
        pop_size = min(20, max(10, max_calls // 25))
        
        # Initialize population with LHS-like spread
        pop = []
        for i in range(pop_size):
            if not elapsed_ok():
                break
            # Use stratified sampling in each dimension for better coverage
            x = np.zeros(d)
            for j in range(d):
                # Stratified sampling: divide each dimension into pop_size bins
                bin_size = 10.0 / pop_size
                lower = -5.0 + (i + rng.random()) * bin_size
                x[j] = lower
            x = np.clip(x, -5.0, 5.0)
            fx = evaluate(x)
            pop.append((fx, x))

        if not pop:
            # Fallback
            x0 = rng.uniform(-5, 5, d)
            evaluate(x0)
            return best_x, best_f

        # Sort population by fitness
        pop.sort(key=lambda t: t[0])
        
        # DE parameters
        F = 0.8  # differential weight
        CR = 0.9  # crossover rate
        stagnation_count = 0
        stagnation_limit = 30
        last_best_f = best_f

        # Main DE loop
        while elapsed_ok():
            # Get current population as list of (f, x)
            # Ensure we have at least 4 individuals for DE
            if len(pop) < 4:
                # Add more random individuals
                while len(pop) < 4 and elapsed_ok():
                    x = rng.uniform(-5, 5, d)
                    fx = evaluate(x)
                    pop.append((fx, x))
                pop.sort(key=lambda t: t[0])
                if len(pop) < 4:
                    break

            # Select best individual
            f_best, x_best = pop[0]
            
            # Check for improvement
            if best_f < last_best_f - 1e-10:
                stagnation_count = 0
                last_best_f = best_f
            else:
                stagnation_count += 1

            # Generate offspring using DE/rand/1/bin
            new_pop = list(pop)  # Keep current population
            
            # Try to generate one offspring per generation to save budget
            if len(pop) >= 4:
                # Select three distinct random individuals
                idxs = rng.choice(len(pop), size=3, replace=False)
                i1, i2, i3 = idxs[0], idxs[1], idxs[2]
                
                # Differential mutation
                v = pop[i1][1] + F * (pop[i2][1] - pop[i3][1])
                
                # Binomial crossover with x_best (or a random individual)
                # Use x_best as the target for exploitation
                target = x_best
                j_rand = rng.integers(0, d)
                trial = np.zeros(d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = target[j]
                
                trial = np.clip(trial, -5.0, 5.0)
                f_trial = evaluate(trial)
                
                # Selection: keep the better of trial and target
                if f_trial < f_best:
                    # Replace the worst in the population with the new best
                    new_pop[-1] = (f_trial, trial)
                    # Or replace target if it was the worst
                    # For simplicity, add to new_pop and sort
                    new_pop.append((f_trial, trial))
            
            # Also try local refinement around best if stagnating
            if stagnation_count >= stagnation_limit // 2:
                if elapsed_ok():
                    # Local search around best_x
                    sigma_local = 0.5
                    for _ in range(3):
                        if not elapsed_ok():
                            break
                        perturbation = rng.standard_normal(d) * sigma_local
                        y = np.clip(best_x + perturbation, -5.0, 5.0)
                        fy = evaluate(y)
                        if fy < best_f:
                            new_pop.append((fy, y))
                    
                    # If still stuck, try a larger jump
                    if stagnation_count >= stagnation_limit:
                        y = rng.uniform(-5, 5, d)
                        fy = evaluate(y)
                        new_pop.append((fy, y))
                        stagnation_count = 0
                        last_best_f = best_f

            # Update population: keep the best pop_size individuals
            new_pop.sort(key=lambda t: t[0])
            pop = new_pop[:pop_size]
            
            # Check for convergence
            if len(pop) > 1:
                f_range = pop[-1][0] - pop[0][0]
                if f_range < 1e-12:
                    break

            # Adaptive F and CR
            if stagnation_count > 10:
                F = min(F + 0.05, 1.5)  # Increase exploration
                CR = min(CR + 0.05, 1.0)
            elif stagnation_count == 0:
                F = max(F - 0.02, 0.5)  # Decrease exploitation
                CR = max(CR - 0.02, 0.7)

        return best_x, best_f
# EVOLVE-BLOCK-END
