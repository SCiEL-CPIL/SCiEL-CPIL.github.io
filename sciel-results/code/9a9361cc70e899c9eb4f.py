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
        best_x = np.zeros(dim)
        
        start_time = time.time()
        # Safety margin: 330s limit, use 325s to be safe
        deadline = start_time + 325.0

        def wrapper(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls_made >= total_budget:
                raise TimeoutError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Initial Global Search using Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    points[i, j] = halton(i + 1, j + 2)
            return points

        global_calls = 20
        halton_points = halton_sequence(global_calls, dim)
        halton_points = halton_points * 10 - 5

        for x in halton_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            try:
                fx = wrapper(x)
            except TimeoutError:
                break
            # wrapper updates best_f and best_x

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                wrapper(best_x)
            except TimeoutError:
                return best_x, best_f

        # Phase 2: Coordinate-wise Adaptive Step Size (CMA-ES lite)
        # This approach maintains a step size for each dimension, adapting based on success rates.
        # It is more robust for non-smooth functions like Weierstrass than standard Nelder-Mead 
        # because it doesn't rely on simplex geometry which can degenerate.
        
        def coordinate_es(start_x, step_limit, rng):
            nonlocal calls_made, best_f, best_x
            
            n = dim
            x = start_x.copy()
            # Evaluate start point if not already done? 
            # The caller usually evaluates or we do it here. 
            # Let's assume we start from a point that has been evaluated or we evaluate it now.
            # To be safe, we evaluate.
            fx = wrapper(x)
            
            # Independent step sizes for each dimension
            sigma = np.ones(n) * 0.5
            # Success history for each dimension
            successes = [np.zeros(10) for _ in range(n)]
            success_idx = [0 for _ in range(n)]
            window_size = 10
            
            steps = 0
            while steps < step_limit and calls_made < total_budget and time.time() < deadline:
                # Choose a random dimension to perturb
                d = rng.integers(n)
                # Choose a random direction (+1 or -1)
                direction = rng.choice([-1, 1])
                
                x_trial = x.copy()
                x_trial[d] += sigma[d] * direction
                x_trial = np.clip(x_trial, -5, 5)
                
                fx_trial = wrapper(x_trial)
                
                # Update best
                if fx_trial < best_f:
                    best_f = fx_trial
                    best_x = x_trial.copy()
                
                # Adapt sigma[d]
                if fx_trial < fx:
                    # Success
                    x = x_trial
                    fx = fx_trial
                    successes[d][success_idx[d]] = 1
                else:
                    # Failure
                    successes[d][success_idx[d]] = 0
                
                success_idx[d] = (success_idx[d] + 1) % window_size
                
                # Check success rate for this dimension
                if np.sum(successes[d]) > window_size * 0.6:
                    sigma[d] *= 1.1
                elif np.sum(successes[d]) < window_size * 0.2:
                    sigma[d] /= 1.1
                
                sigma[d] = np.clip(sigma[d], 1e-6, 5.0)
                steps += 1
            
            return x, fx

        # Phase 3: Finite Difference Gradient Descent for Polishing
        # Weierstrass is continuous but not differentiable everywhere. 
        # However, finite differences can still provide a good local descent direction 
        # in the smooth regions or effectively act as a local search.
        
        def fd_gradient_descent(start_x, step_limit, rng):
            nonlocal calls_made, best_f, best_x
            
            x = start_x.copy()
            fx = wrapper(x)
            
            # Initial step size for FD
            h = 0.01
            lr = 0.1 # Learning rate for the update
            
            steps = 0
            while steps < step_limit and calls_made < total_budget and time.time() < deadline:
                # Estimate gradient
                grad = np.zeros(dim)
                for i in range(dim):
                    x_plus = x.copy()
                    x_plus[i] += h
                    x_plus = np.clip(x_plus, -5, 5)
                    fx_plus = wrapper(x_plus)
                    
                    x_minus = x.copy()
                    x_minus[i] -= h
                    x_minus = np.clip(x_minus, -5, 5)
                    fx_minus = wrapper(x_minus)
                    
                    grad[i] = (fx_plus - fx_minus) / (2 * h)
                
                # Update
                x_new = x - lr * grad
                x_new = np.clip(x_new, -5, 5)
                
                fx_new = wrapper(x_new)
                
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase learning rate slightly if we are making good progress? 
                    # Or keep it constant. Let's keep it constant for stability.
                else:
                    # If no improvement, reduce step size
                    lr *= 0.5
                    h *= 0.5
                    if h < 1e-6:
                        break # Converged or stuck
                
                steps += 1
            
            return x, fx

        # Main Loop: Interleave ES and FD GD
        # We will run a few cycles of ES to explore, then FD GD to exploit.
        
        remaining_budget = total_budget - calls_made
        
        # Divide remaining budget into chunks
        # e.g., 3 cycles of (ES 100 steps, FD GD 20 steps)
        # 500 - 20 (halton) = 480 remaining.
        # 3 * (100 + 20) = 360. Leaves 120 for final polish or more cycles.
        
        num_cycles = 3
        es_steps_per_cycle = 100
        fd_steps_per_cycle = 20
        
        for i in range(num_cycles):
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            # Run Coordinate ES from current best
            es_limit = min(es_steps_per_cycle, total_budget - calls_made)
            if es_limit > 0:
                coordinate_es(best_x, es_limit, rng)
            
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            # Run FD Gradient Descent from current best
            fd_limit = min(fd_steps_per_cycle, total_budget - calls_made)
            if fd_limit > 0:
                fd_gradient_descent(best_x, fd_limit, rng)
        
        # Final Polish: Use remaining budget for a fine-grained ES with small sigma
        if calls_made < total_budget and time.time() > deadline:
            pass # Should not happen due to checks above, but just in case
        
        if calls_made < total_budget and time.time() < deadline:
            # Run a final ES with a smaller initial sigma to refine
            def es_final(start_x, step_limit, rng):
                nonlocal calls_made, best_f, best_x
                n = dim
                x = start_x.copy()
                fx = wrapper(x)
                
                sigma = np.ones(n) * 0.05
                successes = [np.zeros(10) for _ in range(n)]
                success_idx = [0 for _ in range(n)]
                window_size = 10
                
                steps = 0
                while steps < step_limit and calls_made < total_budget and time.time() < deadline:
                    d = rng.integers(n)
                    direction = rng.choice([-1, 1])
                    
                    x_trial = x.copy()
                    x_trial[d] += sigma[d] * direction
                    x_trial = np.clip(x_trial, -5, 5)
                    
                    fx_trial = wrapper(x_trial)
                    
                    if fx_trial < best_f:
                        best_f = fx_trial
                        best_x = x_trial.copy()
                    
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        successes[d][success_idx[d]] = 1
                    else:
                        successes[d][success_idx[d]] = 0
                    
                    success_idx[d] = (success_idx[d] + 1) % window_size
                    
                    if np.sum(successes[d]) > window_size * 0.6:
                        sigma[d] *= 1.05
                    elif np.sum(successes[d]) < window_size * 0.2:
                        sigma[d] /= 1.05
                    
                    sigma[d] = np.clip(sigma[d], 1e-8, 0.5)
                    steps += 1
                return x, fx

            final_limit = total_budget - calls_made
            if final_limit > 0:
                es_final(best_x, final_limit, rng)

        return best_x, best_f
