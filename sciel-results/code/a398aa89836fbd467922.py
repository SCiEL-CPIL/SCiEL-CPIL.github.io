"""Optimizer for BBOB f6 (Attractive sector) using Derandomized Coordinate Descent."""
import numpy as np
import time

class Optimizer:
    """
    Derandomized Coordinate Descent optimizer for f6.
    1. Initial sampling.
    2. Coordinate descent with adaptive step sizes.
    3. Local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        max_calls = budget

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x_clipped))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(42)

        # Phase 1: Initial Sampling
        # We start with a few random points to get a rough idea of the landscape
        n_init = 20
        if n_init > budget:
            n_init = budget // 2

        # Start with a few random points
        for _ in range(n_init):
            if time_left() < 5.0 or calls >= max_calls:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            eval_func(x)

        if calls >= max_calls or time_left() < 10.0:
            return best_x, best_f

        # Phase 2: Coordinate Descent with Adaptive Step Sizes
        # Initialize step sizes
        # Start with a moderate step size
        step_sizes = np.full(dim, 1.0)
        
        # History for adaptive step sizing
        # We track the last few successes/failures for each dimension
        # If a move improves, increase step size slightly or keep it.
        # If a move fails, decrease step size.
        # Simple heuristic: 
        #   success -> step *= 1.5
        #   failure -> step *= 0.5
        #   clamp step to [0.01, 5.0]

        x_current = best_x.copy()
        f_current = best_f
        
        # Main Loop
        # We alternate between exploring new random points (if stuck) and refining
        # But primarily, we do coordinate descent.
        
        # To handle the "rotated" nature, pure coordinate descent might be slow.
        # However, adaptive step sizes help. 
        # Also, f6 is "Attractive Sector", which might have a specific structure.
        # Let's stick to robust CD.
        
        # We will run CD until budget is exhausted or time runs out.
        # We also periodically try a "restart" from a random point if we seem stuck?
        # No, let's keep it simple: just CD. If CD gets stuck, the step sizes will shrink.
        # If we shrink too much, we stop improving.
        
        # Let's add a small "jitter" or "restart" mechanism if no improvement for a while.
        last_improvement_call = calls
        no_improve_count = 0
        max_no_improve = 50 # If we make 50 calls without improvement, try a random restart?
        
        # Actually, a better strategy for BBOB f6 (which is rotated) is to use 
        # a population-based method or a smarter local search.
        # But CD is required by the prompt's strategy hint.
        # Let's enhance CD:
        # 1. Try moving in + and - direction for each dimension.
        # 2. If improvement, accept and adjust step.
        # 3. If no improvement in a full cycle, reduce all steps.
        
        while calls < max_calls and time_left() > 5.0:
            improved_this_cycle = False
            
            for d in range(dim):
                if calls >= max_calls or time_left() < 5.0:
                    break
                
                # Try positive step
                x_try = x_current.copy()
                x_try[d] += step_sizes[d]
                x_try = np.clip(x_try, -5.0, 5.0)
                
                # Only evaluate if it's different from current
                if not np.allclose(x_try, x_current):
                    f_try = eval_func(x_try)
                    
                    if f_try < f_current:
                        # Improvement
                        x_current = x_try
                        f_current = f_try
                        step_sizes[d] *= 1.5 # Increase step
                        step_sizes[d] = min(step_sizes[d], 5.0)
                        improved_this_cycle = True
                        last_improvement_call = calls
                    else:
                        # No improvement, try negative step
                        x_try_neg = x_current.copy()
                        x_try_neg[d] -= step_sizes[d]
                        x_try_neg = np.clip(x_try_neg, -5.0, 5.0)
                        
                        if not np.allclose(x_try_neg, x_current):
                            f_try_neg = eval_func(x_try_neg)
                            
                            if f_try_neg < f_current:
                                # Improvement
                                x_current = x_try_neg
                                f_current = f_try_neg
                                step_sizes[d] *= 1.5
                                step_sizes[d] = min(step_sizes[d], 5.0)
                                improved_this_cycle = True
                                last_improvement_call = calls
                            else:
                                # Failure
                                step_sizes[d] *= 0.5
                                step_sizes[d] = max(step_sizes[d], 0.01)
                        
                if not improved_this_cycle:
                    # If we haven't improved in this dimension, we might have already tried it?
                    # No, the logic above handles + and -.
                    pass

            # If we completed a full cycle without any improvement
            if not improved_this_cycle:
                no_improve_count += 1
                # If we are stuck, maybe reduce all step sizes further?
                # Or try a random perturbation?
                if no_improve_count > 10:
                    # Try a random restart from a point near the best
                    # Or just reduce steps drastically
                    step_sizes *= 0.5
                    step_sizes = np.clip(step_sizes, 0.01, 5.0)
                    no_improve_count = 0 # Reset counter to avoid infinite shrinkage immediately
                    
                    # If steps are too small, we are likely at a local minimum.
                    # Since f6 is rotated, CD might be stuck.
                    # Let's try a random jump to escape?
                    # The prompt says "Derandomized Coordinate Descent". 
                    # Let's stick to CD but if we are really stuck, we can try 
                    # a random point to see if we find a better basin.
                    if np.min(step_sizes) < 0.05:
                        # Try a random point
                        x_rand = rng.uniform(-5.0, 5.0, dim)
                        f_rand = eval_func(x_rand)
                        if f_rand < f_current:
                            x_current = x_rand
                            f_current = f_rand
                            step_sizes = np.full(dim, 1.0) # Reset steps
                        else:
                            # Stay in current basin, keep shrinking or stop?
                            # If we keep shrinking, we waste budget.
                            # Let's just continue, the loop will exit when calls are done.
                            pass
            else:
                no_improve_count = 0

        return best_x, best_f
