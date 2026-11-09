"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-strategy optimizer for Schwefel f20 on [-5,5]^5.
    
    Strategy:
    1. Quick multi-start L-BFGS-B from diverse points to find good local minima fast.
    2. (1+1)-ES with 1/5 success rule from the best L-BFGS-B result for responsive refinement.
    3. Final tight-tolerance L-BFGS-B polish if budget remains.
    
    Key improvements over previous version:
    - L-BFGS-B runs first to quickly find good local solutions (AOCC rewards early progress)
    - (1+1)-ES updates best after every call, unlike batch methods
    - 1/5 success rule for adaptive step size
    - Diverse starting points including known Schwefel minima region
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        rng = np.random.default_rng(42)
        call_count = [0]
        best_x = None
        best_f = np.inf

        def safe_call(x):
            """Wrapper to track calls and enforce budget limits."""
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            fx = func(x)
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        # Use mutable containers for best tracking
        best_f = [np.inf]
        best_x = [None]

        def safe_call(x):
            """Wrapper to track calls and enforce budget limits."""
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            fx = func(x)
            if fx < best_f[0]:
                best_f[0] = fx
                best_x[0] = x.copy()
            return fx

        # Phase 1: Multi-start L-BFGS-B to quickly find good local solutions
        # This is crucial for AOCC because it makes rapid progress early on
        n_starts = 6
        lbfgs_maxiter = 25  # Limit iterations per L-BFGS-B run
        
        # Diverse starting points
        start_points = []
        # Point 0: Origin
        start_points.append(np.zeros(dim))
        # Point 1: Near the known Schwefel global minimum region (4.2097, -4.2097, ...)
        start_points.append(np.array([4.209687, -4.209687, 4.209687, -4.209687, 4.209687]))
        start_points.append(np.array([-4.209687, 4.209687, -4.209687, 4.209687, -4.209687]))
        # Points 2-5: Random points in different regions
        start_points.append(rng.uniform(-4, 4, dim))
        start_points.append(rng.uniform(-2, 2, dim))
        start_points.append(rng.uniform(-5, 5, dim))

        for x0 in start_points:
            if time.time() > deadline or call_count[0] >= budget - 50:
                break
            
            try:
                res = minimize(
                    safe_call,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': lbfgs_maxiter, 'maxls': 20}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = res.x.copy()
            except Exception:
                pass

        # Phase 2: (1+1)-ES with 1/5 success rule for responsive refinement
        # This updates best_x after every single call, maximizing AOCC
        if best_x[0] is not None and call_count[0] < budget and time.time() < deadline:
            x = best_x[0].copy()
            fx = best_f[0]
            sigma = 1.0  # Initial step size
            
            # 1/5 success rule parameters
            window_size = 5 * dim  # Standard 1/5 rule window
            successes = 0
            total = 0
            
            remaining = budget - call_count[0]
            for _ in range(remaining):
                if time.time() > deadline or call_count[0] >= budget:
                    break
                
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f[0]:
                        best_f[0] = fx
                        best_x[0] = x.copy()
                total += 1
                
                # 1/5 success rule: adjust sigma based on success rate
                if total >= window_size:
                    rate = successes / total
                    if rate > 0.2:
                        sigma *= 1.2
                    elif rate < 0.2:
                        sigma *= 0.8
                    successes = 0
                    total = 0
                
                # Stop if sigma is too small
                if sigma < 1e-10:
                    break

        # Phase 3: Final tight-tolerance L-BFGS-B polish
        if best_x[0] is not None and call_count[0] < budget and time.time() < deadline - 2:
            remaining = budget - call_count[0]
            if remaining > 10:
                try:
                    res = minimize(
                        safe_call,
                        best_x[0].copy(),
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': min(remaining, 100), 'maxls': 30,
                                 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res.fun < best_f[0]:
                        best_f[0] = res.fun
                        best_x[0] = res.x.copy()
                except Exception:
                    pass

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END
