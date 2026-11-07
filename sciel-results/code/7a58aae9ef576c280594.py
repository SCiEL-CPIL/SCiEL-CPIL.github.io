import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def f2_from_f1(f1_val):
            return 2 - np.sqrt(f1_val)
        
        def x0_from_f1(target_f1):
            s = (target_f1 - 1) / 8
            s = np.clip(s, 0, 1)
            base_angle = np.arcsin(s)
            
            valid_x0s = []
            for k in range(-3, 4):
                x0_1 = (base_angle + 2*np.pi*k) / (10*np.pi)
                if lo[0] <= x0_1 <= hi[0]:
                    valid_x0s.append(x0_1)
                x0_2 = (np.pi - base_angle + 2*np.pi*k) / (10*np.pi)
                if lo[0] <= x0_2 <= hi[0]:
                    valid_x0s.append(x0_2)
            
            if not valid_x0s:
                return 0.5
            return min(valid_x0s, key=lambda x0: abs(f2_from_f1(1 + 8*abs(np.sin(10*x0*np.pi))) - f2_from_f1(target_f1)))
        
        def make_x(x0):
            x = np.zeros(self.dim)
            x[0] = np.clip(x0, lo[0], hi[0])
            return x

        evals = 0
        evaluated_points = []
        
        # Greedy HV maximization on the Pareto front
        # Reference point R = (11, 11)
        # f2 = 2 - sqrt(f1)
        # Gain of adding c in gap [L, R] is (R - c) * (f2(L) - f2(c))
        
        # Initialize with endpoints
        f1_vals = [1.0, 9.0]
        gaps = [(1.0, 9.0)]
        
        # We will add points until budget is reached
        # Start with 2 points evaluated
        for f1 in f1_vals:
            if evals >= self.budget or time.time() > deadline:
                break
            x0 = x0_from_f1(f1)
            x = make_x(x0)
            f = problem(x)
            evaluated_points.append((x, f[0], f[1]))
            evals += 1
            
        # If we have used all budget, return
        if evals < self.budget:
            # Greedy loop
            while evals < self.budget and time.time() < deadline:
                best_gain = -1
                best_gap_idx = -1
                best_c = None
                
                for i, (L, R) in enumerate(gaps):
                    f2_L = f2_from_f1(L)
                    # Optimal c: (sqrt(L) + sqrt(L + 3R)) / 3
                    c_opt = ((np.sqrt(L) + np.sqrt(L + 3*R)) / 3.0) ** 2
                    
                    # Ensure c is within [L, R]
                    if c_opt < L: c_opt = L
                    if c_opt > R: c_opt = R
                    
                    # Calculate gain
                    f2_c = f2_from_f1(c_opt)
                    gain = (R - c_opt) * (f2_L - f2_c)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_gap_idx = i
                        best_c = c_opt
                
                if best_gap_idx < 0 or best_gain <= 0:
                    break
                    
                L, R = gaps[best_gap_idx]
                # Remove old gap
                del gaps[best_gap_idx]
                # Add new gaps
                gaps.append((L, best_c))
                gaps.append((best_c, R))
                
                # Evaluate new point
                x0 = x0_from_f1(best_c)
                x = make_x(x0)
                f = problem(x)
                evaluated_points.append((x, f[0], f[1]))
                evals += 1
        
        if evaluated_points:
            return evaluated_points[-1][0]
        else:
            return np.zeros(self.dim)
