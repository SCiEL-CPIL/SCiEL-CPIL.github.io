import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        d = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            
            calls += 1
            try:
                x_arr = np.asarray(x, dtype=float).flatten()
                f_val = float(func(x_arr))
            except Exception:
                return best_f
                
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        # Phase 1: Estimate gradient using axis-aligned probes
        # For linear f(x) = w·x + c, estimate w_i by:
        # f(e_i * h) - f(-e_i * h) gives 2*h*w_i
        # Use h=5 (full range) for maximum signal
        f_origin = wrapper(np.zeros(d))
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Estimate gradient components
        w_est = np.zeros(d)
        for i in range(d):
            if calls >= budget or time.time() > deadline:
                break
            
            x_pos = np.zeros(d)
            x_pos[i] = 5.0
            x_neg = np.zeros(d)
            x_neg[i] = -5.0
            
            f_pos = wrapper(x_pos)
            f_neg = wrapper(x_neg)
            
            # w_i ≈ (f_pos - f_neg) / (5 - (-5)) = (f_pos - f_neg) / 10
            w_est[i] = (f_pos - f_neg) / 10.0
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: Compute the optimal vertex
        # For linear f, min is at x* = -sign(w) * 5
        # If w_i > 0, optimum is at x_i = -5
        # If w_i < 0, optimum is at x_i = 5
        # If w_i ≈ 0, either value works
        x_opt = np.zeros(d)
        for i in range(d):
            if w_est[i] > 0:
                x_opt[i] = -5.0
            elif w_est[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0  # doesn't matter for linear
        
        f_opt = wrapper(x_opt)
        
        # Phase 3: Verify with a few random points and other vertices
        rng = np.random.default_rng(42)
        all_corners = list(np.ndindex(*([2]*d)))
        rng2 = np.random.default_rng(123)
        rng2.shuffle(all_corners)
        
        # Try the best corner first, then a few random vertices
        for corner in all_corners[:10]:
            if calls >= budget or time.time() > deadline:
                break
            x_c = np.array([(1 if c == 0 else -1) for c in corner], dtype=float) * 5.0
            if not np.allclose(x_c, x_opt):
                wrapper(x_c)
        
        # Phase 4: If budget remains, do additional random sampling
        while calls < budget and time.time() < deadline:
            x_rand = rng.uniform(-5, 5, size=d)
            wrapper(x_rand)
        
        return best_x, best_f
