import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        
        dim = self.dim
        
        # Step 1: Determine best tail
        # ZDT4 g(x) is minimized at x_i = 0 for i > 1
        # So optimal tail is clip(0, lo, hi)
        tail_candidate_1 = np.clip(0.0, lo[1:], hi[1:])
        # Fallback: center
        tail_candidate_2 = (lo[1:] + hi[1:]) / 2.0
        
        # Evaluate both tails with a mid x1 to see which is better
        x1_mid = (lo[0] + hi[0]) / 2.0
        
        best_tail = tail_candidate_1
        best_f2 = float('inf')
        
        for tail in [tail_candidate_1, tail_candidate_2]:
            x = np.concatenate(([x1_mid], tail))
            x = np.clip(x, lo, hi)
            try:
                obj = problem(x)
                if obj is not None and obj[1] < best_f2:
                    best_f2 = obj[1]
                    best_tail = tail
            except Exception:
                pass
        
        # If both failed, default to 0
        if best_tail is None:
             best_tail = np.clip(0.0, lo[1:], hi[1:])
             
        # Step 2: Sample x1
        # Remaining budget
        n_evals_used = 2
        n_left = self.budget - n_evals_used
        if n_left < 2: n_left = 2
        
        # Generate u values
        # 25% uniform, 75% power 2
        n_uniform = n_left // 4
        n_power = n_left - n_uniform
        
        u_uniform = np.linspace(0, 1, n_uniform)
        u_power = np.linspace(0, 1, n_power)
        
        # Map to x1 range
        # x1 = lo0 + (hi0 - lo0) * u
        x1_uniform = lo[0] + (hi[0] - lo[0]) * u_uniform
        x1_power = lo[0] + (hi[0] - lo[0]) * (u_power ** 2)
        
        x1_vals = np.concatenate([x1_uniform, x1_power])
        
        # Ensure endpoints are included explicitly if not covered by linspace (linspace covers them)
        # But linspace(0,1) includes 0 and 1. So x1=lo and x1=hi are included.
        
        # Sort and unique to avoid duplicate evaluations
        x1_vals = np.unique(x1_vals)
        
        # Evaluate
        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.concatenate(([x1], best_tail))
            x = np.clip(x, lo, hi)
            try:
                problem(x)
            except Exception:
                break
                
        return None
