import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def do_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= total_budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(12345)
        
        # Initialize at the center of the box
        # For Rastrigin, the global minimum is at the shifted origin.
        # Starting at 0 is a good heuristic guess for the "center" of the search space.
        current_x = np.zeros(n)
        f_curr = do_call(current_x)
        best_x = current_x.copy()
        best_f = f_curr

        # Adaptive step size
        # Start with a large step to explore, shrink as we approach the end
        # We use a heavy-tailed distribution (Lévy flight-like) to allow occasional large jumps
        # but mostly small steps for fine-grained search.
        
        # Base step size
        base_step = 2.0
        
        # We will run until budget is exhausted or time runs out
        # The step size will decay linearly or exponentially.
        # Let's use a decay that starts at 2.0 and ends at 0.05
        # Total calls remaining: total_budget - 1
        calls_remaining = total_budget - 1
        
        for i in range(calls_remaining):
            if not time_ok() or calls_made >= total_budget:
                break
            
            # Calculate current step size
            # Linear decay from base_step to min_step
            progress = i / max(1, calls_remaining - 1)
            min_step = 0.05
            step_size = base_step - (base_step - min_step) * progress
            
            # Generate a random direction
            # Use a normal distribution for the direction, but scale it
            # To get heavy-tailed steps, we can multiply by a random factor
            # Or simply use a normal distribution with a large sigma initially.
            # Let's stick to a standard normal direction, scaled by step_size.
            # To add "Lévy" flavor, we can occasionally make a big jump.
            
            # 10% chance of a "long jump" (restart-like behavior but local)
            if rng.random() < 0.05:
                # Jump to a random point in the box, but biased towards current best?
                # No, pure random restart is risky. Let's do a large local jump.
                dir_vec = rng.standard_normal(n)
                # Normalize
                norm = np.linalg.norm(dir_vec)
                if norm > 0:
                    dir_vec /= norm
                # Large step
                jump_size = step_size * 5.0
                candidate = current_x + dir_vec * jump_size
            else:
                # Normal step
                dir_vec = rng.standard_normal(n)
                candidate = current_x + dir_vec * step_size
            
            candidate = np.clip(candidate, -5.0, 5.0)
            f_cand = do_call(candidate)
            
            if f_cand < f_curr:
                # Accept
                current_x = candidate
                f_curr = f_cand
                # If we found a new best, we might want to keep the step size slightly larger
                # or reset the decay a bit? 
                # For simplicity, we continue the decay. The heavy-tailed nature handles exploration.
            else:
                # Reject
                # We stay at current_x
                pass

        return best_x, best_f
