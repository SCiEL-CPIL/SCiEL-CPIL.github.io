import numpy as np
import time
from scipy.optimize import minimize

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
        # Use Halton for coverage and a few random points for diversity
        n_global = 40
        candidates = []
        
        for i in range(n_global):
            if calls_made >= total_budget or time.time() > deadline:
                break
            if i < 30:
                x_unit = halton_point(i + 1, dim)
                x = x_unit * 10.0 - 5.0
            else:
                x = rng.uniform(-5.0, 5.0, dim)
            
            try:
                fx = wrapper(x)
                candidates.append((fx, x.copy()))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except TimeoutError:
                break

        if best_x is None:
            return np.zeros(dim), best_f

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
            local_best_x = x_init.copy()
            local_best_f = fx_init
            
            sigma = 1.0
            successes = 0
            total_trials = 0
            no_improve_count = 0
            calls_used_for_this_candidate = 0
            
            # Run until budget for this candidate is exhausted or global budget hit
            while calls_made < total_budget and time.time() < deadline and calls_used_for_this_candidate < budget_per_candidate:
                
                x_trial = local_best_x + sigma * rng.standard_normal(dim)
                try:
                    fx_trial = wrapper(x_trial)
                except TimeoutError:
                    break
                
                calls_used_for_this_candidate += 1
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

        # Phase 3: Final Polish with Nelder-Mead
        # Use remaining budget to refine the best point found so far
        remaining = total_budget - calls_made
        if remaining > 10 and best_x is not None and time.time() < deadline:
            # Create a simplex around best_x
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x
            for i in range(dim):
                simplex[i + 1] = best_x
                simplex[i + 1][i] += 0.1  # Small perturbation for NM

            try:
                res = minimize(wrapper, best_x, method='Nelder-Mead',
                               options={
                                   'maxfev': remaining,
                                   'xatol': 1e-8,
                                   'fatol': 1e-8,
                                   'adaptive': True,
                                   'initial_simplex': simplex
                               })
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
                    
            except TimeoutError:
                pass
            except Exception:
                pass

        return best_x, best_f
