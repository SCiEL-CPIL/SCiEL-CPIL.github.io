import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 280.0  # Safety margin (300s limit, leave 20s)

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        
        n = self.dim
        best_x = None
        best_fx = float('inf')
        
        def eval_func(x):
            nonlocal self
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return float(func(np.clip(x, -5.0, 5.0)))
        
        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
        # Strategy: Multi-start Nelder-Mead + Coordinate Descent
        # Bent Cigar (f12) has one wide axis and narrow others.
        # Nelder-Mead adapts to this, but coordinate descent helps refine.
        
        # Generate diverse starting points
        rng = np.random.RandomState(42)
        start_points = [np.zeros(n)]
        
        # Axis-aligned points to help detect the wide axis early
        for i in range(n):
            p_pos = np.zeros(n)
            p_pos[i] = 4.0
            start_points.append(p_pos)
            
            p_neg = np.zeros(n)
            p_neg[i] = -4.0
            start_points.append(p_neg)
            
        # Random points
        for _ in range(10):
            start_points.append(rng.uniform(-4.5, 4.5, size=n))
            
        # Shuffle to ensure variety
        indices = list(range(len(start_points)))
        rng.shuffle(indices)
        start_points = [start_points[i] for i in indices]
        
        # Phase 1: Global Search with Nelder-Mead
        # We allocate most of the budget to this phase.
        # Nelder-Mead is effective for smooth, non-convex landscapes like f12.
        
        for i, x0 in enumerate(start_points):
            if self.calls >= self.budget - 10:
                break
            if time.time() - self.start_time > self.time_limit:
                break
            
            try:
                # Limit iterations to prevent single run from consuming all budget
                # But allow enough for convergence if found early.
                max_iter = 500
                res = minimize(eval_func, x0, method='Nelder-Mead',
                              options={
                                  'maxiter': max_iter,
                                  'xatol': 1e-8,
                                  'fatol': 1e-10,
                                  'adaptive': True
                              })
                
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
                    
            except StopIteration:
                break
            except Exception:
                continue
        
        # Phase 2: Local Refinement with Coordinate Descent
        # If we have budget left, use coordinate descent to squeeze out more precision.
        # This is particularly good for the narrow axes of the cigar.
        
        if best_x is not None and self.calls < self.budget - 5 and time.time() - self.start_time < self.time_limit:
            x = best_x.copy()
            fx = best_fx
            step = 1.0
            max_cd_iter = 100
            
            for _ in range(max_cd_iter):
                if self.calls >= self.budget - 5:
                    break
                if time.time() - self.start_time > self.time_limit:
                    break
                
                improved = False
                for i in range(n):
                    if self.calls >= self.budget - 5:
                        break
                    if time.time() - self.start_time > self.time_limit:
                        break
                    
                    # Try positive step
                    x_trial = x.copy()
                    x_trial[i] += step
                    if x_trial[i] > 5.0: x_trial[i] = 5.0
                    f_trial = eval_func(x_trial)
                    
                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        improved = True
                    else:
                        # Try negative step
                        x_trial = x.copy()
                        x_trial[i] -= step
                        if x_trial[i] < -5.0: x_trial[i] = -5.0
                        f_trial = eval_func(x_trial)
                        
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            improved = True
                            
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
                
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_fx = eval_func(best_x)
            
        return best_x, best_fx
