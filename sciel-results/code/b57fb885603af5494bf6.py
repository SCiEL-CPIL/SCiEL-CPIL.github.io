import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Deterministic multi-start L-BFGS-B on a coarse grid for Discus (f11).
    
    Strategy:
    1. Generate a diverse set of 10 initial points (center, corners, mid-edges, random).
    2. Evaluate all to find the best starting point.
    3. Run L-BFGS-B from the best starting point with numerical gradients.
    4. If budget remains, run L-BFGS-B from other promising starting points.
    5. Final polish from the global best found.
    
    This leverages the smooth, twice-differentiable nature of f11 to rapidly
    traverse the narrow valley with minimal function evaluations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill
        
        d = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        best_x = np.zeros(d)
        best_f = float('inf')
        calls = 0
        
        # Wrapper to count calls and enforce budget
        def counting_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                raise Exception("BudgetExhausted")
            
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # Numerical gradient using central differences
        # Cost: 2*d calls
        def counting_grad(x):
            nonlocal calls
            if calls + 2 * d > budget or not time_ok():
                raise Exception("BudgetExhausted")
                
            h = 1e-5
            grad = np.zeros(d)
            for i in range(d):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                # Clip to box to handle boundaries
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                f_plus = counting_func(x_plus)
                f_minus = counting_func(x_minus)
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            return grad

        # Generate 10 diverse candidate starting points
        rng = np.random.RandomState(42)
        candidates = []
        
        # 1. Center
        candidates.append(np.zeros(d))
        
        # 2-3. Corners
        candidates.append(np.full(d, -4.5))
        candidates.append(np.full(d, 4.5))
        
        # 4-9. Mid-edge points (positive and negative along each axis)
        for i in range(d):
            pt_pos = np.zeros(d)
            pt_pos[i] = 4.0
            candidates.append(pt_pos)
            
            pt_neg = np.zeros(d)
            pt_neg[i] = -4.0
            candidates.append(pt_neg)
        
        # 10. Random point
        candidates.append(rng.uniform(-4.0, 4.0, d))
        
        # Keep only 10 best candidates if we have more
        if len(candidates) > 10:
            candidates = candidates[:10]
        
        # Phase 1: Evaluate all candidates to find the best start
        cand_vals = []
        for pt in candidates:
            if calls >= budget or not time_ok():
                break
            try:
                f_val = counting_func(pt)
                cand_vals.append((f_val, pt))
            except Exception:
                break
        
        if not cand_vals:
            return best_x, best_f
        
        # Sort by value
        cand_vals.sort(key=lambda x: x[0])
        
        # Phase 2: Run L-BFGS-B from the best candidate first
        bounds = [(-5.0, 5.0)] * d
        
        # Try the best candidate with a generous budget
        f_val, start_pt = cand_vals[0]
        if calls < budget and time_ok():
            try:
                res = minimize(
                    counting_func, 
                    start_pt, 
                    method='L-BFGS-B', 
                    jac=counting_grad, 
                    bounds=bounds,
                    options={
                        'maxiter': 50, 
                        'ftol': 1e-12, 
                        'gtol': 1e-8, 
                        'maxls': 20
                    }
                )
            except Exception:
                pass
        
        # Phase 3: Run L-BFGS-B from other promising candidates if budget remains
        for f_val, start_pt in cand_vals[1:]:
            if calls >= budget or not time_ok():
                break
            
            # Only try if this candidate was reasonably close to the best
            if f_val > best_f + 10.0:
                continue
                
            try:
                res = minimize(
                    counting_func, 
                    start_pt, 
                    method='L-BFGS-B', 
                    jac=counting_grad, 
                    bounds=bounds,
                    options={
                        'maxiter': 30, 
                        'ftol': 1e-12, 
                        'gtol': 1e-8, 
                        'maxls': 20
                    }
                )
            except Exception:
                continue
        
        # Phase 4: Final polish from the global best
        if calls < budget and time_ok():
            try:
                res = minimize(
                    counting_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    jac=counting_grad, 
                    bounds=bounds,
                    options={
                        'maxiter': 30, 
                        'ftol': 1e-14, 
                        'gtol': 1e-10, 
                        'maxls': 20
                    }
                )
            except Exception:
                pass
        
        return best_x, best_f
