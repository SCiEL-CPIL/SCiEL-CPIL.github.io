import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)
        
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            x_clip = np.clip(x, -5, 5)
            f_val = func(x_clip)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
            return f_val
        
        def hooke_jeeves(x0, max_evals):
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            f = evaluate(x)
            if f is None:
                return x, f
            
            # Initial step size
            delta = 0.5
            improved = True
            no_improve_count = 0
            min_delta = 1e-6
            
            while calls_made < budget and (budget - calls_made) > 0:
                if time.time() > deadline:
                    break
                
                if not improved:
                    delta *= 0.5
                    if delta < min_delta:
                        break
                    improved = False
                    no_improve_count += 1
                    if no_improve_count > 10:
                        break
                else:
                    delta *= 1.5
                    no_improve_count = 0
                
                # Pattern move
                # Try to continue in the direction of the last successful move
                # For simplicity, we just do coordinate search each time with adaptive step
                
                # Coordinate search
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try positive direction
                    x_trial = x.copy()
                    x_trial[i] += delta
                    f_trial = evaluate(x_trial)
                    if f_trial is None:
                        break
                    
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                        # Try to expand in this direction
                        while calls_made < budget:
                            x_trial = x.copy()
                            x_trial[i] += delta
                            f_trial = evaluate(x_trial)
                            if f_trial is None:
                                break
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                            else:
                                break
                    else:
                        # Try negative direction
                        x_trial = x.copy()
                        x_trial[i] -= delta
                        f_trial = evaluate(x_trial)
                        if f_trial is None:
                            break
                        
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            # Try to expand in this direction
                            while calls_made < budget:
                                x_trial = x.copy()
                                x_trial[i] -= delta
                                f_trial = evaluate(x_trial)
                                if f_trial is None:
                                    break
                                if f_trial < f:
                                    x = x_trial
                                    f = f_trial
                                else:
                                    break
                        
                # If no improvement in this full cycle, shrink step
                if not improved:
                    delta *= 0.5
                    if delta < min_delta:
                        break
        
        return x, f
        
        # Multi-restart strategy
        # We have 500 budget. Let's use a few restarts.
        # Each restart: Hooke-Jeeves from a random point.
        
        # Initial evaluation at origin
        f0 = evaluate(np.zeros(dim))
        
        # List of starting points
        start_points = [np.zeros(dim)]
        
        # Add random points
        for _ in range(10):
            x_rand = rng.uniform(-4, 4, size=dim)
            start_points.append(x_rand)
        
        # Shuffle start points to ensure diversity
        rng.shuffle(start_points)
        
        # Allocate budget
        # First few restarts get more budget, later ones less?
        # Or just run until budget exhausted.
        # Hooke-Jeeves is cheap per step but might take many steps.
        
        # Strategy:
        # 1. Evaluate origin.
        # 2. Run HJ from origin with a small budget (e.g., 50).
        # 3. Run HJ from random points with remaining budget.
        
        # Actually, let's just run HJ sequentially from each start point.
        # If it converges quickly, we move to the next.
        
        # But HJ in the code above runs until budget or time.
        # We need to limit each run.
        
        # Revised HJ with max_evals parameter
        def hooke_jeeves_limited(x0, max_evals):
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            f = evaluate(x)
            if f is None:
                return x, f
            
            delta = 0.5
            improved = True
            no_improve_count = 0
            min_delta = 1e-6
            local_calls = 0
            max_steps = 100 # Safety cap on iterations
            
            while local_calls < max_evals and calls_made < budget:
                if time.time() > deadline:
                    break
                
                if not improved:
                    delta *= 0.5
                    if delta < min_delta:
                        break
                    improved = False
                    no_improve_count += 1
                    if no_improve_count > 10:
                        break
                else:
                    delta *= 1.5
                    no_improve_count = 0
                
                for i in range(dim):
                    if local_calls >= max_evals or calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x_trial = x.copy()
                    x_trial[i] += delta
                    f_trial = evaluate(x_trial)
                    if f_trial is None:
                        break
                    local_calls += 1
                    
                    if f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                        # Expand
                        while local_calls < max_evals and calls_made < budget:
                            x_trial = x.copy()
                            x_trial[i] += delta
                            f_trial = evaluate(x_trial)
                            if f_trial is None:
                                break
                            local_calls += 1
                            if f_trial < f:
                                x = x_trial
                                f = f_trial
                            else:
                                break
                    else:
                        x_trial = x.copy()
                        x_trial[i] -= delta
                        f_trial = evaluate(x_trial)
                        if f_trial is None:
                            break
                        local_calls += 1
                        
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            # Expand
                            while local_calls < max_evals and calls_made < budget:
                                x_trial = x.copy()
                                x_trial[i] -= delta
                                f_trial = evaluate(x_trial)
                                if f_trial is None:
                                    break
                                local_calls += 1
                                if f_trial < f:
                                    x = x_trial
                                    f = f_trial
                                else:
                                    break
                            
                if not improved:
                    delta *= 0.5
                    if delta < min_delta:
                        break
        
        return x, f
        
        # Run HJ from origin
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        if f0 is not None:
            # Run HJ from origin with 100 evals
            x_hj, f_hj = hooke_jeeves_limited(x0, 100)
        
        # Run HJ from random points
        # We have ~400 evals left.
        # Let's do 4 more restarts with ~100 evals each.
        for _ in range(4):
            if calls_made >= budget - 50: # Keep some buffer
                break
            if time.time() > deadline:
                break
            
            x_rand = rng.uniform(-4, 4, size=dim)
            f_rand = evaluate(x_rand)
            if f_rand is None:
                break
            
            # Run HJ from this point
            x_hj, f_hj = hooke_jeeves_limited(x_rand, 100)
        
        return best_x, best_f

