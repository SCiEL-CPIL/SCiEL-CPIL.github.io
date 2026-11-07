import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f14 (Different powers) using Multi-start Powell's Method.
    Powell's method is derivative-free and robust for smooth, non-convex functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        eval_count = 0

        def wrapped(x):
            nonlocal eval_count
            if eval_count >= budget:
                return 1e10
            eval_count += 1
            # Ensure input is a valid numpy array and clipped
            x = np.asarray(x)
            if x.shape != (n,):
                x = x.reshape(n)
            return func(np.clip(x, -5, 5))

        # Phase 1: Generate diverse starting points
        starts = []
        # Origin
        starts.append(np.zeros(n))
        
        # Random points in the box
        for _ in range(15):
            starts.append(rng.uniform(-4.5, 4.5, n))
            
        # Points along axes (to capture single-variable dependencies)
        for i in range(n):
            for val in [-3.0, 3.0]:
                pt = np.zeros(n)
                pt[i] = val
                starts.append(pt)
        
        # Remove duplicates (approximate)
        unique_starts = []
        for s in starts:
            is_dup = False
            for u in unique_starts:
                if np.linalg.norm(s - u) < 1e-5:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        starts = unique_starts

        # Evaluate all starting points to find the best ones
        start_evals = []
        for s in starts:
            if time.time() > deadline - 5.0 or eval_count >= budget:
                break
            fx = wrapped(s)
            start_evals.append((fx, s))
            if fx < best_fx:
                best_fx = fx
                best_x = s.copy()
        
        # Sort by fitness
        start_evals.sort(key=lambda x: x[0])
        
        # Phase 2: Refine the top 5 starting points using Powell's Method
        # Powell is derivative-free and robust
        top_k = min(5, len(start_evals))
        remaining_budget = budget - eval_count
        
        if top_k > 0 and remaining_budget > 10:
            # Allocate budget roughly evenly to top candidates, but prioritize the best
            # Best gets more iterations
            allocations = []
            total_weight = 0
            for i in range(top_k):
                w = (top_k - i) + 1
                allocations.append(w)
                total_weight += w
            
            for i in range(top_k):
                if time.time() > deadline - 5.0 or eval_count >= budget:
                    break
                
                x0 = start_evals[i][1]
                # Estimate max iterations for this run
                # Powell uses function evaluations per iteration.
                # In n-dimensions, one iteration of Powell is roughly O(n) evaluations.
                # Let's allow a few dozen iterations.
                max_iter = max(1, int(allocations[i] / total_weight * (remaining_budget * 0.5)))
                max_iter = min(max_iter, 100) # Cap to avoid one run consuming all
                
                try:
                    res = minimize(
                        wrapped, x0, method='Powell',
                        bounds=[(-5, 5)] * n,
                        options={'maxiter': max_iter, 'maxfev': budget - eval_count}
                    )
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Phase 3: Final Polish from the global best
        if time.time() < deadline - 2.0 and eval_count < budget and best_x is not None:
            remaining = budget - eval_count
            if remaining >= 10:
                try:
                    # Use Nelder-Mead for a final local polish, or Powell again
                    # Nelder-Mead is often good for final touches
                    res = minimize(
                        wrapped, best_x, method='Nelder-Mead',
                        bounds=[(-5, 5)] * n,
                        options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(n)
            best_fx = wrapped(best_x)

        return best_x, best_fx
