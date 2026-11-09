"""Differential Evolution optimizer for BBOB f15."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Population size
        N = 10
        # DE parameters
        F = 0.8  # Mutation scale
        CR = 0.9  # Crossover rate

        # Initialize population
        # Spread initial points to cover the space
        pop = np.empty((N, dim))
        for i in range(N):
            # Mix of random and grid-like starts to ensure diversity
            if i == 0:
                pop[i] = np.zeros(dim)
            else:
                pop[i] = rng.uniform(-5, 5, dim)

        # Evaluate initial population
        fvals = np.empty(N)
        calls_made = 0

        def safe_eval(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            try:
                return func(x)
            except Exception:
                return np.inf

        for i in range(N):
            fvals[i] = safe_eval(pop[i])
            if time.time() > deadline:
                break

        if calls_made >= budget:
            best_idx = np.argmin(fvals)
            return pop[best_idx], fvals[best_idx]

        best_idx = np.argmin(fvals)
        best_val = fvals[best_idx]
        best_x = pop[best_idx].copy()

        # Main DE loop
        # We have budget - calls_made left
        # Each generation costs N evaluations (plus the best one is carried over, 
        # but standard DE evaluates N offspring)
        
        while calls_made + N <= budget and time.time() < deadline:
            # Sort population by fitness (optional but helps some variants, 
            # standard DE doesn't strictly require it, but let's keep it simple)
            
            # Generate offspring
            for i in range(N):
                # Mutation: DE/rand/1
                # Choose 3 distinct indices different from i
                while True:
                    a, b, c = rng.choice(N, 3, replace=False)
                    if a != i and b != i and c != i:
                        break

                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                r = rng.random()
                j_rand = rng.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() <= CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Evaluate
                f_trial = safe_eval(trial)
                
                # Selection
                if f_trial < fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial
                    if f_trial < best_val:
                        best_val = f_trial
                        best_x = trial.copy()
                        best_idx = i
                        # Optional: Adapt F or CR here, but fixed is often robust
                        # Let's slightly decrease F as we converge to refine
                        # But careful not to get stuck. 
                        # Maybe keep F constant for simplicity and robustness.
                        
            # Check time
            if time.time() > deadline:
                break

        # If we have budget left but not enough for a full generation, 
        # do a few local refinements around the best point?
        # Or just return.
        # Let's use remaining budget for a simple (1+1)-ES polish on the best point
        # if there's significant time and budget left.
        
        remaining = budget - calls_made
        if remaining > 0 and time.time() < deadline - 1.0:
            x = best_x.copy()
            fx = best_val
            sigma = 0.5
            # Use a portion of remaining budget, say up to 50 more steps or until time runs out
            max_steps = min(remaining, 50)
            for _ in range(max_steps):
                if time.time() > deadline - 1.0:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_eval(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_val:
                        best_val = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-4:
                        break

        return best_x, best_val
# EVOLVE-BLOCK-END
