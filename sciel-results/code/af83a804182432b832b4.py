import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def safe_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Exploration using Halton Sequence
        # Halton sequence provides low-discrepancy sampling, good for covering the space
        # We use about 40% of the budget for this phase
        n_explore = int(budget * 0.4)
        
        # Generate Halton points
        # We need a fast Halton generator or just use random if Halton is too slow to generate
        # For dim=5, generating 200 Halton points is fast.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index = index // base
            return r

        # Primes for Halton bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        candidates = []
        
        # Ensure we sample the center as well
        candidates.append(np.zeros(dim))
        
        for i in range(1, n_explore + 1):
            x = np.array([halton(i, primes[d]) for d in range(dim)])
            # Map [0, 1] to [-5, 5]
            x = x * 10.0 - 5.0
            candidates.append(x)
            
            # Check budget/time
            if calls_made >= budget or time.time() > deadline - 10.0:
                break

        # Evaluate all candidates
        for x in candidates:
            if calls_made >= budget or time.time() > deadline - 10.0:
                break
            safe_call(x)

        # Phase 2: Local Refinement using Nelder-Mead on the best candidates
        # We need to identify the best candidates from the evaluations above.
        # Since safe_call updates best_x, we don't have a list of all evaluated points and their values 
        # stored in a way that's easy to sort unless we keep them.
        
        # Let's refactor safe_call to store history if needed, or just re-evaluate top points?
        # No, re-evaluating wastes budget.
        # Let's store the evaluated points and values in Phase 1.
        
        # Revised Plan:
        # 1. Generate points
        # 2. Evaluate and store (x, f)
        # 3. Sort by f
        # 4. Take top K
        # 5. Run NM on each, tracking global best.

        # Let's restart the logic with storage.
        
        # Reset state for clean implementation
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        eval_history = [] # List of (f, x)

        def safe_call_stored(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            eval_history.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Exploration
        n_explore = int(budget * 0.4)
        
        # Add origin
        safe_call_stored(np.zeros(dim))
        
        for i in range(1, n_explore):
            if calls_made >= budget or time.time() > deadline - 10.0:
                break
            x = np.array([halton(i, primes[d]) for d in range(dim)])
            x = x * 10.0 - 5.0
            safe_call_stored(x)

        # Phase 2: Polish
        # Get top 5 distinct points from eval_history
        if eval_history:
            # Sort by function value
            eval_history.sort(key=lambda item: item[0])
            
            # Select top K unique-ish points (simple check: just take top K)
            top_k = 5
            start_points = [x for f, x in eval_history[:top_k]]
            
            # Distribute remaining budget among start points
            remaining_budget = budget - calls_made
            if remaining_budget > 0:
                budget_per_point = remaining_budget // top_k
                
                for x0 in start_points:
                    if calls_made >= budget or time.time() > deadline - 5.0:
                        break
                    
                    # Run Nelder-Mead
                    # maxiter is not strictly call count, but we can limit it
                    # A reasonable maxiter for NM in 5D is around 100-200
                    # But we must respect the call budget.
                    # We can wrap func to count calls and stop if exceeded.
                    
                    local_calls = 0
                    local_budget = budget_per_point
                    
                    def nm_func(x):
                        nonlocal local_calls
                        if local_calls >= local_budget:
                            return np.inf
                        local_calls += 1
                        return safe_call_stored(x)
                    
                    try:
                        res = minimize(
                            nm_func,
                            x0=x0,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 500, # High limit, but budget enforced by nm_func
                                'xatol': 1e-8,
                                'fatol': 1e-8,
                                'adaptive': True
                            }
                        )
                    except Exception:
                        pass

        return best_x, best_f
