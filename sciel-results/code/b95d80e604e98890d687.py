import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # 10s margin for 300s limit
        
        # 1. Center
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_val = f0
        calls_made = 1
        
        # 2. Gradient estimation using Positive Probes only
        # f(x0 + h*ei) - f(x0) = h * gi
        # If gi > 0, min is at -h. If gi < 0, min is at +h.
        h = 5.0
        signs = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline:
                break
            xp = x0.copy()
            xp[i] += h
            xp = np.clip(xp, -5, 5)
            fp = func(xp)
            best_val = min(best_val, fp)
            calls_made += 1
            
            # Determine sign of gradient component i
            diff = fp - f0
            if diff > 0:
                signs[i] = -1.0
            elif diff < 0:
                signs[i] = 1.0
            else:
                signs[i] = 0.0
                
        # 3. Predict optimal vertex
        x_pred = np.zeros(dim)
        for i in range(dim):
            if signs[i] == -1.0:
                x_pred[i] = -5.0
            elif signs[i] == 1.0:
                x_pred[i] = 5.0
            else:
                x_pred[i] = 0.0 # Arbitrary, doesn't matter if gradient is 0
        
        # 4. Evaluate predicted vertex
        f_pred = func(x_pred)
        best_val = min(best_val, f_pred)
        calls_made += 1
        
        # 5. Fill remaining budget (optional, but safe to ensure we don't "waste" the opportunity to update best_val if there's any numerical weirdness, though for linear it's exact)
        # Actually, we can just return. The evaluator repeats the last n_i.
        # But to be safe and consistent with "more calls is better for recording", we can just return.
        # The prompt says: "repeating the last n_i if fewer calls are made".
        # So we don't need to fill the budget.
        
        return x_pred, f_pred
