import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 285.0

        # Wrapper to count calls and enforce deadline
        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls_made >= total_budget:
                raise TimeoutError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(12345)

        def halton_sequence(i, b):
            f = 1.0
            r = 0.0
            while i > 0:
                f /= b
                r += f * (i % b)
                i //= b
            return r

        def halton_point(index, dim):
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            point = np.zeros(dim)
            for d in range(dim):
                point[d] = halton_sequence(index + d + 1, primes[d])
            return point

        # Phase 1: Initial Global Search
        n_global = 50
        candidates = []
        
        for i in range(n_global):
            if calls_made >= total_budget or time.time() > deadline:
                break
            x_unit = halton_point(i + 1, dim)
            x = x_unit * 10.0 - 5.0
            try:
                fx = wrapper(x)
                candidates.append((fx, x.copy()))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except TimeoutError:
                break

        if not candidates:
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda item: item[0])
        
        # Select top 3 distinct candidates
        top_candidates = []
        for fx, x in candidates:
            is_distinct = True
            for _, x_top in top_candidates:
                if np.linalg.norm(x - x_top) < 1.0:
                    is_distinct = False
                    break
            if is_distinct:
                top_candidates.append((fx, x))
            if len(top_candidates) == 3:
                break
        
        # Fill if necessary
        while len(top_candidates) < 3 and calls_made < total_budget and time.time() < deadline:
            x = rng.uniform(-5.0, 5.0, dim)
            try:
                fx = wrapper(x)
                top_candidates.append((fx, x.copy()))
            except TimeoutError:
                break

        # Phase 2: Local Search with (1+1)-ES on top candidates
        # We allocate remaining budget to these candidates
        remaining_budget = total_budget - calls_made
        budget_per_candidate = max(10, remaining_budget // len(top_candidates))

        for fx_init, x_init in top_candidates:
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            # Run (1+1)-ES starting from x_init
            # We don't want to override the global best_f/best_x immediately if this candidate is worse
            # But we track local best
            local_best_x = x_init.copy()
            local_best_f = fx_init
            
            sigma = 1.0
            successes = 0
            total_trials = 0
            no_improve_count = 0
            
            # Run until budget for this candidate is exhausted
            while calls_made < total_budget and time.time() < deadline:
                # Check budget for this specific candidate allocation
                # We don't have a separate counter, so we just run until global budget is hit or we switch
                # To switch, we need to track how many calls used for this candidate.
                # Let's just run for a fixed number of iterations or until no improvement.
                
                x_trial = local_best_x + sigma * rng.standard_normal(dim)
                try:
                    fx_trial = wrapper(x_trial)
                except TimeoutError:
                    break
                
                total_trials += 1
                if fx_trial < local_best_f:
                    local_best_f = fx_trial
                    local_best_x = x_trial.copy()
                    successes += 1
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Update sigma every 20 trials
                if total_trials % 20 == 0:
                    current_success_rate = successes / total_trials
                    if current_success_rate > 0.2:
                        sigma *= 1.5
                    else:
                        sigma /= 1.5
                    sigma = np.clip(sigma, 1e-6, 10.0)
                    successes = 0
                    total_trials = 0
                    
                # Restart condition: if no improvement for 30 steps, break to try next candidate
                if no_improve_count > 30:
                    break

            # Update global best if local search found something better
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        return best_x, best_f
