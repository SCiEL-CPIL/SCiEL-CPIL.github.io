"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead with adaptive vertex scaling for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(123)
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit

        call_count = 0
        max_calls = self.budget

        best_x = None
        best_f = float('inf')

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= max_calls:
                return float('inf')
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def check_time():
            return time.time() < deadline and call_count < max_calls

        def check_budget():
            return call_count < max_calls

        # Phase 1: Latin Hypercube Sampling for global coverage
        # Sample 50 points using LHS
        n_lhs = 50
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = (perm + rng.random(n_lhs)) / n_lhs
            lhs_points[:, d] = lhs_points[:, d] * 10 - 5  # Map to [-5, 5]

        # Evaluate all LHS points
        for i in range(n_lhs):
            if not check_time() or not check_budget():
                break
            counted_func(lhs_points[i])

        # Sort by function value and take top candidates
        # We'll track which points were evaluated
        evaluated_indices = []
        eval_values = []
        for i in range(n_lhs):
            if call_count > 0:
                # We need to track which were actually evaluated
                pass
        
        # Let's just re-evaluate the best few from our tracking
        # Actually, let's keep track properly
        # Reset and do it more carefully
        
        # Re-do Phase 1 with proper tracking
        call_count = 0
        best_x = None
        best_f = float('inf')
        
        def counted_func2(x):
            nonlocal call_count, best_x, best_f
            if call_count >= max_calls:
                return float('inf')
            call_count += 1
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Evaluate LHS points and track them
        lhs_results = []
        for i in range(n_lhs):
            if not check_time() or not check_budget():
                break
            val = counted_func2(lhs_points[i])
            lhs_results.append((val, lhs_points[i].copy()))
        
        # Sort by value
        lhs_results.sort(key=lambda x: x[0])
        
        # Phase 2: Nelder-Mead from top candidates
        # Use top 10 candidates for local refinement
        n_starts = min(10, len(lhs_results))
        
        for idx in range(n_starts):
            if not check_time() or not check_budget():
                break
            
            start_x = lhs_results[idx][1]
            
            # Nelder-Mead with adaptive parameters
            try:
                res = minimize(
                    counted_func2,
                    start_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
            
            # Check if we've made significant progress
            # If not, try next start
            
        # Phase 3: If budget remains, do a final refinement from the best point
        if best_x is not None and check_time() and check_budget():
            try:
                # Final polish with tighter tolerances
                res = minimize(
                    counted_func2,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Fallback
        if best_x is None:
            best_x = np.zeros(self.dim)
            if call_count < max_calls:
                best_f = func(best_x)
                call_count += 1
            else:
                best_f = float('inf')
        
        return best_x, best_f
# EVOLVE-BLOCK-END
