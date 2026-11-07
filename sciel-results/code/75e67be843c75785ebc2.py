import numpy as np
import time

class Optimizer:
    """
    Deterministic Grid Search + Fixed-Step Hill Climb (Hooke-Jeeves variant).
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0  # 30s safety margin
        calls = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            # The contract says func clips, but we clip here too for safety
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                f_val = func(x_clipped)
            except Exception:
                return np.inf
            
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: Deterministic Grid Search
        # Evaluate 32 corners and the origin
        grid_points = []
        for mask in range(2 ** self.dim):
            pt = np.zeros(self.dim)
            for i in range(self.dim):
                if mask & (1 << i):
                    pt[i] = 5.0
                else:
                    pt[i] = -5.0
            grid_points.append(pt)
        
        # Add the center (zero vector)
        grid_points.append(np.zeros(self.dim))
        
        for pt in grid_points:
            if calls >= self.budget:
                break
            eval_point(pt)

        if best_x is None:
            # Should not happen, but handle gracefully
            return np.zeros(self.dim), np.inf

        x = best_x.copy()
        f = best_f
        
        # Phase 2: Fixed-Step Local Hill Climb
        # Using a coordinate-wise scan with adaptive step size (Pattern Search)
        step = 2.0
        min_step = 1e-6
        
        while calls < self.budget and time.time() < deadline:
            improved = False
            
            # Scan each dimension
            for i in range(self.dim):
                # Try positive step
                x_trial = x.copy()
                x_trial[i] += step
                f_trial = eval_point(x_trial)
                
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True
                    # After a move, we typically restart the scan or continue.
                    # Continuing is faster per cycle.
                    
                # Try negative step
                x_trial = x.copy()
                x_trial[i] -= step
                f_trial = eval_point(x_trial)
                
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True

            if not improved:
                step *= 0.5
                if step < min_step:
                    break

        return best_x, best_f
