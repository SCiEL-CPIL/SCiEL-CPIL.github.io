"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5 using lattice-aware multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Rastrigin-aware optimizer: lattice sampling + multi-start Nelder-Mead with refinement."""

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
        # Rastrigin has period 2 in each dimension. Sample at integer and half-integer
        # positions to hit the lattice of local minima.
        # 3^5 = 243 points at {-4,-2,0,2,4} but that's 5^5=3125. Too many.
        # Instead, sample at a coarser lattice: {-4, -2, 0, 2, 4} is too many.
        # Use a subset: center + a few key positions.
        
        # Strategy: sample at a grid of size ~150-200 points
        # Use the fact that Rastrigin minima are at integer points (after shift/rotation)
        # But we don't know the shift/rotation, so we sample broadly.
        
        # Phase 1a: Structured sampling
        # Sample at a regular grid with spacing 2.0 (matching Rastrigin period)
        # Grid points: -4, -2, 0, 2, 4 in each dim -> 5^5 = 3125, too many
        # Use spacing 2.5: -4, -1.5, 1, 3.5 -> 4^5 = 1024, still too many
        # Use a smaller structured set:
        
        # 1. Origin and axis-aligned points
        structured_points = [np.zeros(dim)]
        for i in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                p = np.zeros(dim)
                p[i] = val
                structured_points.append(p)
        
        # 2. Add points at half-period offsets
        for i in range(dim):
            for j in range(i + 1, dim):
                for vi in [-3.0, -1.0, 1.0, 3.0]:
                    for vj in [-3.0, -1.0, 1.0, 3.0]:
                        p = np.zeros(dim)
                        p[i] = vi
                        p[j] = vj
                        structured_points.append(p)
        
        # 3. Fill remaining global budget with random sampling
        global_budget = min(150, budget // 3)
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
        
        # Phase 2: Multi-start Nelder-Mead from best candidates
        # We need to track which points were good. Let's re-evaluate top candidates
        # by running NM from several diverse starting points.
        
        remaining_budget = budget - call_count[0]
        
        # Select diverse starting points for NM
        # Use the best few points found, plus some random perturbations
        starts = []
        
        # Start from current best
        starts.append(best_x.copy())
        
        # Add a few more diverse starts
        for _ in range(4):
            if not time_ok() or call_count[0] >= budget - 20:
                break
            # Random start near the best
            x_start = best_x + rng.normal(0, 1.5, dim)
            x_start = np.clip(x_start, -5, 5)
            starts.append(x_start)
        
        # Also try the origin and a few axis points
        extra_starts = [np.zeros(dim)]
        for i in range(dim):
            extra_starts.append(np.concatenate([np.zeros(i), [3.0], np.zeros(dim - i - 1)]))
            extra_starts.append(np.concatenate([np.zeros(i), [-3.0], np.zeros(dim - i - 1)]))
        
        for es in extra_starts[:3]:
            if len(starts) < 8:
                starts.append(es)
        
        # Run Nelder-Mead from each start
        starts.sort(key=lambda x: np.linalg.norm(x - best_x))
        
        for i, x0 in enumerate(starts):
            if not time_ok() or call_count[0] >= budget:
                break
            
            # Allocate budget: more for earlier (better) starts
            remaining = budget - call_count[0]
            if i == 0:
                alloc = min(remaining, 100)
            elif i == 1:
                alloc = min(remaining, 70)
            elif i == 2:
                alloc = min(remaining, 50)
            else:
                alloc = min(remaining, 30)
            
            if alloc < 10:
                break
            
            # Wrap function for scipy minimize
            nm_calls = [0]
            
            def nm_func(x, nm_calls=nm_calls, alloc=alloc):
                if nm_calls[0] >= alloc or call_count[0] >= budget or not time_ok():
                    return best_f
                nm_calls[0] += 1
                return do_call(x)
            
            try:
                res = minimize(nm_func, x0, method='Nelder-Mead',
                               options={'maxiter': alloc, 'xatol': 1e-8, 'fatol': 1e-8, 
                                        'adaptive': True, 'maxfev': alloc})
            except Exception:
                pass
        
        # Phase 3: Final refinement with high-precision NM
        remaining_budget = budget - call_count[0]
        if remaining_budget > 5 and time_ok():
            nm_calls = [0]
            
            def nm_func2(x, nm_calls=nm_calls, alloc=remaining_budget):
                if nm_calls[0] >= alloc or call_count[0] >= budget or not time_ok():
                    return best_f
                nm_calls[0] += 1
                return do_call(x)
            
            try:
                res = minimize(nm_func2, best_x, method='Nelder-Mead',
                               options={'maxiter': remaining_budget, 'xatol': 1e-10, 
                                        'fatol': 1e-10, 'adaptive': True, 
                                        'maxfev': remaining_budget})
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
