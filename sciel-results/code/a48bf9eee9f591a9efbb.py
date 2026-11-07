import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Deterministic multi-start Powell + L-BFGS-B hybrid for Discus (f11).
    
    Strategy:
    1. Use a stratified Sobol-like sequence to generate diverse initial points.
    2. Run Powell (derivative-free, coordinate-wise) from top candidates to
       navigate the narrow valley without gradient estimation overhead.
    3. Polish with L-BFGS-B from the best points found by Powell.
    4. Final polish from global best.
    
    Powell's coordinate-wise line search is well-suited for the narrow valley
    structure of f11, and avoids the 2*d cost of numerical gradients per step.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        
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

        # Numerical gradient using central differences for L-BFGS-B polish
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
                
                x_plus = np.clip(x_plus, -5.0, 5.0)
                x_minus = np.clip(x_minus, -5.0, 5.0)
                
                f_plus = counting_func(x_plus)
                f_minus = counting_func(x_minus)
                
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            return grad

        # Generate diverse candidate starting points using stratified sampling
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
        
        # 10-15. Stratified random points (one per dimension pair)
        for i in range(6):
            pt = rng.uniform(-4.0, 4.0, d)
            candidates.append(pt)
        
        # Phase 1: Evaluate all candidates to find the best starts
        cand_vals = []
        for pt in candidates:
            if calls >= budget or not time_ok():
                break
            try:
                f_val = counting_func(pt)
                cand_vals.append((f_val, pt.copy()))
            except Exception:
                break
        
        if not cand_vals:
            return best_x, best_f
        
        # Sort by value
        cand_vals.sort(key=lambda x: x[0])
        
        bounds = [(-5.0, 5.0)] * d
        
        # Phase 2: Run Powell from the top 3 candidates
        # Powell is derivative-free and uses coordinate-wise line searches,
        # which can navigate the narrow valley of f11 more efficiently than
        # gradient-based methods when the gradient estimation is expensive.
        for idx in range(min(3, len(cand_vals))):
            if calls >= budget or not time_ok():
                break
            
            f_val, start_pt = cand_vals[idx]
            
            # Skip if this candidate is much worse than current best
            if idx > 0 and f_val > best_f + 50.0:
                continue
                
            try:
                res = minimize(
                    counting_func, 
                    start_pt, 
                    method='Powell', 
                    bounds=bounds,
                    options={
                        'maxiter': 30, 
                        'ftol': 1e-10, 
                        'xtol': 1e-10,
                        'niter': 20
                    }
                )
            except Exception:
                pass
        
        # Phase 3: Polish with L-BFGS-B from the global best found so far
        if calls < budget and time_ok():
            try:
                res = minimize(
                    counting_func, 
                    best_x.copy(), 
                    method='L-BFGS-B', 
                    jac=counting_grad, 
                    bounds=bounds,
                    options={
                        'maxiter': 40, 
                        'ftol': 1e-12, 
                        'gtol': 1e-8, 
                        'maxls': 20
                    }
                )
            except Exception:
                pass
        
        # Phase 4: If budget remains, run Powell from next-best candidates
        for idx in range(3, min(6, len(cand_vals))):
            if calls >= budget or not time_ok():
                break
            
            f_val, start_pt = cand_vals[idx]
            
            if f_val > best_f + 20.0:
                continue
                
            try:
                res = minimize(
                    counting_func, 
                    start_pt, 
                    method='Powell', 
                    bounds=bounds,
                    options={
                        'maxiter': 20, 
                        'ftol': 1e-8, 
                        'xtol': 1e-8,
                        'niter': 15
                    }
                )
            except Exception:
                continue
        
        # Phase 5: Final L-BFGS-B polish from global best
        if calls < budget and time_ok():
            try:
                res = minimize(
                    counting_func, 
                    best_x.copy(), 
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
