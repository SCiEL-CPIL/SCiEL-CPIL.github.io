"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """(1+1)-ES with Nelder-Mead polish for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin before 330s kill
        
        # Use a fixed seed for reproducibility in evaluation, but the task implies 
        # we can use any deterministic or stochastic method. 
        # A fixed seed ensures consistent behavior across instances if the instances 
        # were deterministic, but since instances have hidden shifts, we rely on 
        # the robustness of the algorithm.
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            return fx

        # 1. Global Phase: (1+1)-ES with Halton-like initialization
        # We use a Halton sequence to generate initial points to cover the space well.
        # Then we run a (1+1)-ES starting from the best of these initial points.
        
        # Generate a small set of Halton points for initialization
        # Halton sequence in base 2, 3, 5, 7, 11 for 5 dimensions
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        init_points = []
        num_init = min(10, budget // 10)  # Use up to 10 initial points
        for i in range(1, num_init + 1):
            p = np.array([halton(i, base) for base in [2, 3, 5, 7, 11][:dim]])
            # Map [0,1] to [-5,5]
            p = p * 10.0 - 5.0
            init_points.append(p)

        # Evaluate initial points
        for p in init_points:
            if time.time() > deadline - 5.0 or calls >= budget:
                break
            fp = safe_func(p)
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # (1+1)-ES Parameters
        sigma = 1.0  # Initial step size
        success_rate = 0.2  # 1/5 success rule
        sigma_inc = 1.2
        sigma_dec = 1.0 / 1.2
        min_sigma = 1e-6
        
        # Run (1+1)-ES until time or budget is nearly exhausted
        # Reserve 20% of budget for Nelder-Mead
        nm_budget = budget // 5
        es_budget = budget - nm_budget
        
        while calls < es_budget and time.time() < deadline - 5.0:
            # Generate candidate
            z = rng.standard_normal(dim)
            x_new = best_x + sigma * z
            x_new = np.clip(x_new, -5.0, 5.0)
            
            f_new = safe_func(x_new)
            
            if f_new < best_f:
                best_x = x_new
                best_f = f_new
                sigma *= sigma_inc
            else:
                sigma *= sigma_dec
            
            if sigma < min_sigma:
                break

        # 2. Local Phase: Nelder-Mead Polish
        # Use the best point found by (1+1)-ES as the start for Nelder-Mead
        if time.time() < deadline - 5.0 and calls < budget:
            # Nelder-Mead options
            # maxiter is set to ensure we don't exceed budget
            # We check time inside the callback or just rely on the fact that NM is fast
            # But we must be careful not to exceed the function call budget.
            # SciPy's minimize doesn't have a strict function call limit in options for NM, 
            # so we wrap the function to count calls and raise an exception or return early if needed.
            
            nm_calls = 0
            def nm_func(x):
                nonlocal nm_calls, calls, best_f, best_x
                if calls >= budget:
                    return best_f # Return current best to stop improvement
                nm_calls += 1
                x_clipped = np.clip(x, -5.0, 5.0)
                fx = func(x_clipped)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                return fx

            try:
                # Nelder-Mead is robust for local refinement
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12})
                
                # Update best if NM found something better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
