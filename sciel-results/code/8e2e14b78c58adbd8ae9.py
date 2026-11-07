"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-Free Coordinate Descent with multi-resolution grid scan."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # 300s limit, 25s safety margin
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.full(n, np.nan)
        best_f = np.inf
        calls_made = 0

        def _time_ok():
            return (time.time() - start_time) < deadline

        def _eval(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def _coordinate_descent(x0, sigma_init, max_iter):
            """Perform coordinate descent with adaptive step sizes."""
            nonlocal best_x, best_f, calls_made
            x = np.array(x0, dtype=float)
            fx = _eval(x)
            step_sizes = np.full(n, sigma_init)
            
            for it in range(max_iter):
                if not _time_ok() or calls_made >= total_budget:
                    break
                
                improved = False
                for d in range(n):
                    if not _time_ok() or calls_made >= total_budget:
                        break
                    
                    # Try positive step
                    x_pos = x.copy()
                    x_pos[d] += step_sizes[d]
                    x_pos = np.clip(x_pos, -5, 5)
                    f_pos = _eval(x_pos)
                    
                    if f_pos < fx:
                        x = x_pos
                        fx = f_pos
                        step_sizes[d] *= 1.5
                        improved = True
                    else:
                        # Try negative step
                        x_neg = x.copy()
                        x_neg[d] -= step_sizes[d]
                        x_neg = np.clip(x_neg, -5, 5)
                        f_neg = _eval(x_neg)
                        
                        if f_neg < fx:
                            x = x_neg
                            fx = f_neg
                            step_sizes[d] *= 1.5
                            improved = True
                        else:
                            step_sizes[d] *= 0.7
                    
                    if not _time_ok() or calls_made >= total_budget:
                        break
                
                if not improved:
                    # Shrink all steps if no improvement
                    step_sizes *= 0.8
                    if np.all(step_sizes < 1e-4):
                        break
        
        def _grid_scan(x0, resolutions):
            """Multi-resolution grid scan around x0."""
            nonlocal best_x, best_f, calls_made
            x = np.array(x0, dtype=float)
            fx = _eval(x)
            
            for res in resolutions:
                if not _time_ok() or calls_made >= total_budget:
                    break
                
                # For each dimension, try a few points around current best
                for d in range(n):
                    if not _time_ok() or calls_made >= total_budget:
                        break
                    
                    # Try offsets from current position
                    offsets = np.linspace(-1, 1, res)
                    for off in offsets:
                        if not _time_ok() or calls_made >= total_budget:
                            break
                        
                        x_try = x.copy()
                        x_try[d] += off * 2.0  # Scan range of [-2, 2] around current
                        x_try = np.clip(x_try, -5, 5)
                        f_try = _eval(x_try)
                        
                        if f_try < fx:
                            x = x_try
                            fx = f_try
        
        # Phase 1: Coarse grid scan from multiple starting points
        # Start from origin
        x0 = np.zeros(n)
        _grid_scan(x0, [5])  # Coarse scan
        
        # Start from random points to explore bi-modal landscape
        if _time_ok() and calls_made < total_budget:
            x0 = rng.uniform(-4, 4, n)
            _grid_scan(x0, [3])
        
        if _time_ok() and calls_made < total_budget:
            x0 = rng.uniform(-3, 3, n)
            _grid_scan(x0, [3])
        
        # Phase 2: Coordinate descent refinement from best point
        if _time_ok() and calls_made < total_budget and not np.isnan(best_x).any():
            _coordinate_descent(best_x, 1.0, 20)
        
        # Phase 3: Fine grid scan
        if _time_ok() and calls_made < total_budget and not np.isnan(best_x).any():
            _grid_scan(best_x, [7])
        
        # Phase 4: Final coordinate descent with smaller steps
        if _time_ok() and calls_made < total_budget and not np.isnan(best_x).any():
            _coordinate_descent(best_x, 0.3, 15)
        
        # Phase 5: Very fine refinement
        if _time_ok() and calls_made < total_budget and not np.isnan(best_x).any():
            _grid_scan(best_x, [11])
            _coordinate_descent(best_x, 0.1, 10)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
