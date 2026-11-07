"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with coordinate-wise golden-section polish for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Track total calls
        total_calls = [0]

        def tracked_func(x):
            nonlocal total_calls
            if time.time() >= deadline:
                raise Exception("timeout")
            if total_calls[0] >= budget:
                raise Exception("budget exceeded")
            total_calls[0] += 1
            return func(x)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = tracked_func(x)

        best_x = x.copy()
        best_fx = fx
        sigma = 3.0

        # Phase 1: (1+1)-ES for the bulk of the budget
        # Reserve last 150 calls for coordinate-wise polish
        polish_budget = min(150, budget // 3)
        es_budget = budget - polish_budget

        for _ in range(es_budget - 1):
            if time.time() >= deadline:
                break
            if total_calls[0] >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = tracked_func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.92

            # Clamp sigma
            if sigma > 15.0:
                sigma = 15.0
            elif sigma < 1e-12:
                sigma = 1e-12

        # Phase 2: Coordinate-wise golden-section search polish
        # This exploits the Bent Cigar structure by optimizing each dimension
        # sequentially, which handles the high conditioning well
        remaining = budget - total_calls[0]
        if remaining > 10 and time.time() < deadline:
            golden_ratio = (1 + np.sqrt(5)) / 2
            inv_phi = 1 - 1 / golden_ratio  # ≈ 0.381966
            
            # Start from best_x found by ES
            x_polish = best_x.copy()
            fx_polish = best_fx
            
            # Cycle through dimensions multiple times
            # Each cycle optimizes each dimension once via golden-section
            max_cycles = min(5, remaining // (dim * 8))
            
            for cycle in range(max_cycles):
                for d in range(dim):
                    if time.time() >= deadline:
                        break
                    if total_calls[0] >= budget:
                        break
                    
                    # Golden-section search on dimension d
                    # Initial interval: current value ± some range
                    # The range should shrink as we converge
                    if cycle == 0:
                        half_width = 2.0
                    else:
                        half_width = 1.0 / (cycle + 1)
                    
                    a = max(-5.0, x_polish[d] - half_width)
                    b = min(5.0, x_polish[d] + half_width)
                    
                    if b - a < 1e-10:
                        continue
                    
                    # Golden-section points
                    c = b - inv_phi * (b - a)
                    d_pt = a + inv_phi * (b - a)
                    
                    # Evaluate at c and d_pt
                    x_c = x_polish.copy()
                    x_c[d] = c
                    fc = tracked_func(x_c)
                    
                    x_d = x_polish.copy()
                    x_d[d] = d_pt
                    fd = tracked_func(x_d)
                    
                    # Iterate golden-section
                    for _ in range(min(10, remaining - (total_calls[0] - (budget - remaining)))):
                        if time.time() >= deadline:
                            break
                        if total_calls[0] >= budget:
                            break
                        
                        if fc < fd:
                            b = d_pt
                            d_pt = c
                            fd = fc
                            c = b - inv_phi * (b - a)
                            x_c[d] = c
                            fc = tracked_func(x_c)
                        else:
                            a = c
                            c = d_pt
                            fc = fd
                            d_pt = a + inv_phi * (b - a)
                            x_d[d] = d_pt
                            fd = tracked_func(x_d)
                        
                        if b - a < 1e-8:
                            break
                    
                    # Update x_polish with the best point found
                    # The minimum is near the midpoint of the final interval
                    best_d = (a + b) / 2
                    x_polish[d] = best_d
                    
                    # Evaluate at the new best point
                    fx_new = tracked_func(x_polish)
                    if fx_new < fx_polish:
                        fx_polish = fx_new
                        if fx_new < best_fx:
                            best_fx = fx_new
                            best_x = x_polish.copy()

        # Phase 3: Final Nelder-Mead polish if budget remains
        remaining2 = budget - total_calls[0]
        if remaining2 > 5 and time.time() < deadline:
            nm_calls = [0]
            
            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining2:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                return func(z)
            
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining2, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
