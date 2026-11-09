"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5 using CMA-ES with lattice-aware initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Rastrigin-aware optimizer: lattice sampling + CMA-ES with multi-start refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        call_count = [0]
        best_x = np.zeros(dim)
        best_f = np.inf

        def do_call(x):
            nonlocal best_f, best_x
            if call_count[0] >= budget:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            call_count[0] += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        def time_ok():
            return time.time() < deadline

        # Phase 1: Lattice-aware global search
        # Rastrigin has period 2 in each dimension. Sample at key lattice positions.
        # Use a structured set of points that covers the likely basin locations.
        
        # Generate candidate points:
        # 1. Origin
        # 2. Axis-aligned points at ±2, ±4
        # 3. Pairs of axes at ±1, ±3
        # 4. Random fill to reach ~150 points
        
        structured_points = [np.zeros(dim)]
        
        # Axis-aligned: 5 dims × 4 values = 20 points
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[i] = val
                structured_points.append(p)
        
        # Pairwise: C(5,2)=10 pairs × 4×4=16 values = 160 points (too many, use subset)
        # Use ±1, ±3 for pairs: 10 × 16 = 160, still too many
        # Use just ±1: 10 × 4 = 40 points
        for i in range(dim):
            for j in range(i + 1, dim):
                for vi in [-1.0, 1.0]:
                    for vj in [-1.0, 1.0]:
                        p = np.zeros(dim)
                        p[i] = vi
                        p[j] = vj
                        structured_points.append(p)
        
        # Total so far: 1 + 20 + 40 = 61 points
        # Add random points to reach ~120
        global_budget = min(120, budget // 4)
        num_random = max(0, global_budget - len(structured_points))
        
        all_points = structured_points[:]
        if num_random > 0:
            random_points = rng.uniform(-5, 5, (num_random, dim))
            all_points.extend(random_points)
        
        # Evaluate all points
        for p in all_points:
            if not time_ok() or call_count[0] >= global_budget:
                break
            do_call(p)
        
        # Phase 2: Multi-start CMA-ES from best candidates
        remaining_budget = budget - call_count[0]
        
        # Select diverse starting points for CMA-ES
        starts = []
        starts.append(best_x.copy())
        
        # Add a few more diverse starts near the best
        for _ in range(3):
            if not time_ok() or call_count[0] >= budget - 30:
                break
            x_start = best_x + rng.normal(0, 1.0, dim)
            x_start = np.clip(x_start, -5, 5)
            starts.append(x_start)
        
        # Also try origin and a few axis points
        extra_starts = [np.zeros(dim)]
        for i in range(dim):
            extra_starts.append(np.concatenate([np.zeros(i), [2.0], np.zeros(dim - i - 1)]))
            extra_starts.append(np.concatenate([np.zeros(i), [-2.0], np.zeros(dim - i - 1)]))
        
        for es in extra_starts[:4]:
            if len(starts) < 7:
                starts.append(es)
        
        # Sort starts by distance to best
        starts.sort(key=lambda x: np.linalg.norm(x - best_x))
        
        # Run CMA-ES from each start
        for i, x0 in enumerate(starts):
            if not time_ok() or call_count[0] >= budget:
                break
            
            remaining = budget - call_count[0]
            if i == 0:
                alloc = min(remaining, 120)
            elif i == 1:
                alloc = min(remaining, 80)
            elif i == 2:
                alloc = min(remaining, 60)
            else:
                alloc = min(remaining, 40)
            
            if alloc < 15:
                break
            
            cma_calls = [0]
            
            def cma_func(x, cma_calls=cma_calls, alloc=alloc):
                if cma_calls[0] >= alloc or call_count[0] >= budget or not time_ok():
                    return best_f
                cma_calls[0] += 1
                return do_call(x)
            
            try:
                res = minimize(cma_func, x0, method='Powell',
                               options={'maxiter': alloc, 'ftol': 1e-10, 
                                        'xtol': 1e-10, 'maxfev': alloc})
            except Exception:
                pass
        
        # Phase 3: Final high-precision Nelder-Mead polish
        remaining_budget = budget - call_count[0]
        if remaining_budget > 10 and time_ok():
            nm_calls = [0]
            
            def nm_func(x, nm_calls=nm_calls):
                if nm_calls[0] >= remaining_budget or call_count[0] >= budget or not time_ok():
                    return best_f
                nm_calls[0] += 1
                return do_call(x)
            
            try:
                res = minimize(nm_func, best_x, method='Nelder-Mead',
                               options={'maxiter': remaining_budget, 'xatol': 1e-12, 
                                        'fatol': 1e-12, 'adaptive': True, 
                                        'maxfev': remaining_budget})
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
