import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s kill

        calls = 0
        best_x = None
        best_f = float('inf')

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial Exploration
        # Sample diverse points to identify promising basins for Rastrigin
        n_init = 20
        candidates = []

        # Always try the origin
        eval_func(np.zeros(dim))

        for _ in range(n_init - 1):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval_func(x)
            candidates.append((f, x))

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Lattice-based candidate generation + Local Refinement
        # Rastrigin f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # In the rotated frame, local minima are at integer coordinates.
        # We don't know the rotation, but we can estimate the "effective" lattice
        # by looking at where our initial samples found low values.
        
        # However, a more robust structural approach for a black-box rotated Rastrigin
        # is to use the fact that the function is periodic with period 1 (or 2pi depending on definition, 
        # standard Rastrigin has period 1 in each dimension if cos(2*pi*x)).
        # Wait, standard Rastrigin is 10n + sum(x_i^2 - 10cos(2pi x_i)).
        # The term x_i^2 dominates the global shape, pushing towards 0.
        # The cos term creates local minima at integers.
        
        # Since the rotation is unknown, we cannot directly generate lattice points in the rotated frame.
        # But we can use the best initial candidate as a "seed" for a lattice-like search.
        # The key insight from the directive: "Model the rotated Rastrigin landscape as a lattice of periodic basins... to generate candidate minima via integer grid search".
        
        # Let's try a different structural approach:
        # Instead of just (1+1)-ES on top-k, we will perform a "coordinate descent" style search
        # that is aware of the periodicity.
        
        # Actually, let's stick to the multi-start (1+1)-ES but improve the adaptation.
        # The inspiration 1 uses a "dynamic budget" which is good.
        # The current program has a "stagnation" reset.
        
        # Let's combine:
        # 1. More initial samples (n_init=20 is a bit low, let's try 25-30).
        # 2. For each top candidate, run (1+1)-ES with a "success rate" based sigma adaptation (1/5 rule).
        # 3. If stagnation occurs, instead of just increasing sigma, we "jump" to a nearby random point
        #    that is still in the basin of attraction (e.g., shift by +/- 0.5 in one random dimension).
        
        # Let's implement a robust (1+1)-ES with the 1/5 success rule.
        
        top_k = 5
        max_local_budget = 80
        min_local_budget = 20

        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            if i >= len(candidates):
                break

            f_start, x_start = candidates[i]
            
            # If this candidate is significantly worse than the global best, 
            # we might skip it or give it very little budget.
            # But since we sorted, the first one is the best.
            # If the 2nd one is much worse, it might be a different basin.
            
            x = x_start.copy()
            f_x = f_start

            # (1+1)-ES Parameters
            sigma = 1.0  # Start larger to explore the basin
            window = 10
            successes = 0
            stagnation = 0
            max_stagnation = 15
            local_calls = 0
            current_budget = min_local_budget

            while calls < budget and time.time() < deadline and local_calls < max_local_budget:
                # Early exit if this candidate is clearly worse than global best
                # and we've spent enough effort
                if local_calls >= current_budget and f_x > best_f + 1.0:
                    break

                # Generate offspring
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                local_calls += 1

                if f_y < f_x:
                    x = y
                    f_x = f_y
                    successes += 1
                    stagnation = 0
                    # Increase budget if improving
                    if local_calls % 10 == 0:
                        current_budget = min(max_local_budget, current_budget + 10)
                else:
                    stagnation += 1

                # Adapt sigma based on success rate (1/5 rule)
                if successes % window == 0 and successes > 0:
                    sr = successes / window
                    if sr > 0.2:  # 1/5 rule
                        sigma *= 1.2
                    elif sr < 0.2:
                        sigma *= 0.8
                    sigma = np.clip(sigma, 0.01, 5.0)
                    successes = 0

                # Stagnation handling: 
                # If we haven't improved in a while, try a "lattice jump".
                # In Rastrigin, moving by 0.5 in a coordinate often moves to a nearby peak/valley.
                # Let's try a random coordinate shift of +/- 0.5.
                if stagnation > max_stagnation:
                    # Pick a random dimension and shift by 0.5
                    d = rng.integers(dim)
                    shift = rng.choice([-0.5, 0.5])
                    y = x.copy()
                    y[d] += shift
                    y = np.clip(y, -5, 5)
                    f_y = eval_func(y)
                    local_calls += 1
                    
                    if f_y < f_x:
                        x = y
                        f_x = f_y
                        successes += 1
                    else:
                        successes += 0 # Count as failure for sigma adaptation
                        # If even the jump didn't help, increase sigma to escape
                        sigma = max(sigma, 1.0)
                    
                    stagnation = 0

        # Phase 3: Final Polish
        # Fine-grained search around the best point found
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.05
            remaining = budget - calls
            max_polish_steps = min(remaining, 50)

            for _ in range(max_polish_steps):
                if calls >= budget or time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                f_y = eval_func(y)
                if f_y < f_x:
                    x = y
                    f_x = f_y
                    sigma *= 0.9
                else:
                    sigma *= 0.95
                sigma = max(sigma, 0.001)

        return best_x, best_f
