"""Parallel coordinate descent with golden-section search for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import log, ceil


class Optimizer:
    """Parallel coordinate descent with golden-section search for 21-peak f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = float(func(np.asarray(x, dtype=float)))
                if not np.isfinite(val):
                    val = best_f
                if val < best_f:
                    best_f = val
                    best_x = np.asarray(x, dtype=float).copy()
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(42)

        # Golden-section search along one coordinate
        def golden_section_1d(x_base, coord_idx, lower, upper, max_iter=30):
            """Minimize f along coordinate coord_idx using golden-section search."""
            nonlocal calls, best_f, best_x
            
            if calls >= budget or time.time() > deadline:
                return x_base
            
            gr = (5.0 ** 0.5 - 1.0) / 2.0  # golden ratio ~0.618
            a = lower
            b = upper
            
            # Initial points
            x1 = b - gr * (b - a)
            x2 = a + gr * (b - a)
            
            def f_1d(val):
                x = x_base.copy()
                x[coord_idx] = val
                return eval_and_track(x)
            
            f1 = f_1d(x1)
            f2 = f_1d(x2)
            
            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = b - gr * (b - a)
                    f1 = f_1d(x1)
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = a + gr * (b - a)
                    f2 = f_1d(x2)
            
            # Return the better of the two points
            mid = (a + b) / 2.0
            if f1 < f2:
                return x1
            else:
                return x2

        # Phase 1: Coarse random sampling to find promising basins
        n_random = max(20, budget // 5)
        candidates = []

        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Coordinate descent refinement on top candidates
        top_k = min(6, len(candidates))
        
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break

            x = candidates[i][1].copy()
            
            # Multiple passes of coordinate descent with random shuffles
            n_passes = 5
            for pass_idx in range(n_passes):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Shuffle coordinate order
                coord_order = rng.permutation(dim)
                
                for coord_idx in coord_order:
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Determine search range around current value
                    cur_val = x[coord_idx]
                    # Adaptive range: start wide, narrow as we converge
                    if pass_idx < 2:
                        half_range = 4.0
                    elif pass_idx < 4:
                        half_range = 2.0
                    else:
                        half_range = 1.0
                    
                    lower = max(-5.0, cur_val - half_range)
                    upper = min(5.0, cur_val + half_range)
                    
                    if upper - lower < 0.1:
                        continue
                    
                    # Golden-section search along this coordinate
                    x_new = golden_section_1d(x, coord_idx, lower, upper, max_iter=25)
                    
                    # Check if we improved
                    if x_new is not None:
                        x = x_new

            # After coordinate descent, do a local random perturbation search
            if calls < budget and time.time() < deadline:
                step = 0.5
                for _ in range(20):
                    if calls >= budget or time.time() > deadline:
                        break
                    x_pert = x + rng.normal(0, step, size=dim)
                    x_pert = np.clip(x_pert, -5, 5)
                    f_pert = eval_and_track(x_pert)
                    if f_pert < best_f:
                        x = x_pert.copy()
                        step = max(0.1, step * 0.8)
                    else:
                        step = min(2.0, step * 1.2)

        # Phase 3: Final refinement around the global best
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            for _ in range(3):
                if calls >= budget or time.time() > deadline:
                    break
                coord_order = rng.permutation(dim)
                for coord_idx in coord_order:
                    if calls >= budget or time.time() > deadline:
                        break
                    cur_val = x[coord_idx]
                    half_range = 0.5
                    lower = max(-5.0, cur_val - half_range)
                    upper = min(5.0, cur_val + half_range)
                    if upper - lower < 0.05:
                        continue
                    x_new = golden_section_1d(x, coord_idx, lower, upper, max_iter=20)
                    if x_new is not None:
                        x = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
