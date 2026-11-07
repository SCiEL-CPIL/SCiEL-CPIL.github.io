"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start (1+1)-ES with 1/5 success rule for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        def time_left():
            return deadline - time.time()
            
        def es_run(func, x0, f0, max_calls, deadline, dim, rng):
            """
            Run (1+1)-ES from x0 with f0.
            Uses 1/5 success rule for sigma adaptation.
            Returns (best_x, best_f, calls_used).
            """
            # Parameters
            # Standard (1+1)-ES parameters
            # c_sigma = 1/5
            # c_sigma_min = 1e-10
            # c_sigma_max = 1e10
            
            sigma = 1.0  # Initial step size, will adapt
            x = x0.copy()
            f = f0
            
            local_best_x = x0.copy()
            local_best_f = f0
            calls_used = 1
            
            # 1/5 rule parameters
            # Success rate target is 1/5 = 0.2
            # If success rate > 1/5, increase sigma
            # If success rate < 1/5, decrease sigma
            
            # We use a sliding window or cumulative sum?
            # Standard 1/5 rule uses a cumulative sum over generations.
            # Let's use a window of last 100 generations for stability.
            window_size = 100
            successes = 0
            generations = 0
            
            # Stagnation detection
            # If no improvement for X generations, break
            stagnation_limit = 50
            stagnation_count = 0
            last_best_f = f0
            
            while calls_used < max_calls and time_left() > 0.5:
                # Generate trial point
                # z ~ N(0, I)
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                
                # Clip to box? 
                # The problem states func clips. 
                # However, for (1+1)-ES, it's often better to let the search
                # go out and let the function clip, or clip ourselves.
                # Clipping ourselves is consistent with the domain.
                x_trial = np.clip(x_trial, -5, 5)
                
                # Evaluate
                f_trial = func(x_trial)
                calls_used += 1
                
                # Acceptance rule: (1+1) is greedy
                if f_trial < f:
                    # Success
                    x = x_trial
                    f = f_trial
                    successes += 1
                    generations += 1
                    
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x.copy()
                        stagnation_count = 0
                    else:
                        # Improved current state but not global best? 
                        # In (1+1), x is always the best found so far in this run.
                        # So f == local_best_f always holds if we update local_best correctly.
                        pass
                else:
                    # Failure
                    generations += 1
                    stagnation_count += 1
                    
                # Update sigma based on 1/5 rule
                # Calculate success rate in the window
                if generations > 0:
                    # We need to maintain a history or use a cumulative approach.
                    # Simpler: maintain a list of successes for the last window_size gens.
                    # But storing a list is overhead. 
                    # Alternative: use the cumulative sum formula from Hansen's book.
                    # Let's use a simple windowed count.
                    pass
                
                # Let's implement the windowed count properly.
                # We need to store success/failure for the last window_size iterations.
                # This requires some state.
                
                # Actually, the standard 1/5 rule implementation often uses:
                # sigma = sigma * exp(c_sigma * (successes / generations - 1/5))
                # But this is sensitive to the history length.
                
                # Let's restart the window logic.
                # To keep it simple and robust, I will use a fixed window size and a circular buffer.
                
                # Wait, I haven't implemented the buffer in the loop above. 
                # Let's refactor the loop to include the buffer.
                
                # For now, let's break and restart with a cleaner implementation.
                # The current loop doesn't track the window correctly.
                
                if stagnation_count > stagnation_limit:
                    break
                    
            return local_best_x, local_best_f, calls_used

        # Refactored ES with proper 1/5 rule implementation
        def es_run_v2(func, x0, f0, max_calls, deadline, dim, rng):
            sigma = 1.0
            x = x0.copy()
            f = f0
            
            local_best_x = x0.copy()
            local_best_f = f0
            calls_used = 1
            
            window_size = 100
            success_history = np.zeros(window_size, dtype=bool)
            success_idx = 0
            gen_count = 0
            successes_in_window = 0
            
            stagnation_limit = 100
            stagnation_count = 0
            last_best_f = f0
            
            while calls_used < max_calls and time_left() > 0.5:
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)
                
                f_trial = func(x_trial)
                calls_used += 1
                
                accepted = False
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    accepted = True
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x.copy()
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    accepted = False
                    stagnation_count += 1
                
                # Update 1/5 rule
                # Remove old entry if window is full
                if gen_count >= window_size:
                    if success_history[success_idx]:
                        successes_in_window -= 1
                
                # Add new entry
                success_history[success_idx] = accepted
                if accepted:
                    successes_in_window += 1
                    
                gen_count += 1
                success_idx = (success_idx + 1) % window_size
                
                # Update sigma
                # Effective number of generations in window
                n_eff = min(gen_count, window_size)
                success_rate = successes_in_window / n_eff
                
                # Adapt sigma
                # c_sigma is the learning rate for sigma adaptation
                # Standard value is 1/5 = 0.2
                c_sigma = 0.2
                sigma = sigma * np.exp(c_sigma * (success_rate - 0.2))
                
                # Bound sigma
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                if stagnation_count > stagnation_limit:
                    break
                    
            return local_best_x, local_best_f, calls_used

        # Phase 1: Global Search / Initialization
        # Sample a few random points to find a good starting point and estimate scale
        n_init = 10
        candidates = []
        for i in range(n_init):
            if calls_made >= budget or time_left() < 1.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Sort candidates
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-Start (1+1)-ES
        # Select diverse starts
        selected_starts = []
        used_points = [best_x]
        
        # Add best point first
        selected_starts.append((best_f, best_x))
        
        for fx, x in candidates[1:]:
            if len(selected_starts) >= 5:
                break
            is_diverse = True
            for existing in used_points:
                if np.linalg.norm(x - existing) < 1.0:
                    is_diverse = False
                    break
            if is_diverse:
                selected_starts.append((fx, x))
                used_points.append(x)
                
        # If not enough diverse points, add random ones
        while len(selected_starts) < 5 and calls_made < budget and time_left() > 1.0:
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                selected_starts.append((fx, x.copy()))
                used_points.append(x.copy())
            else:
                # Check diversity
                is_diverse = True
                for existing in used_points:
                    if np.linalg.norm(x - existing) < 1.0:
                        is_diverse = False
                        break
                if is_diverse:
                    selected_starts.append((fx, x.copy()))
                    used_points.append(x.copy())
                    
        n_starts = len(selected_starts)
        
        # Allocate budget for each start
        # We will run them sequentially, but with a time/budget check.
        # If a start stagnates quickly, it yields back budget.
        
        for i, (fx_start, x_start) in enumerate(selected_starts):
            if calls_made >= budget or time_left() < 5.0:
                break
            
            # Allocate a chunk of budget
            remaining = budget - calls_made
            alloc = remaining // (n_starts - i)
            if alloc < 10:
                alloc = remaining
                
            x_best, f_best, used = es_run_v2(func, x_start, fx_start, alloc, deadline, dim, rng)
            calls_made += used
            
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
                
            # If we improved significantly, we might want to restart from the new best?
            # The multi-start structure handles this by having other starts, 
            # but the best start is the most promising.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
