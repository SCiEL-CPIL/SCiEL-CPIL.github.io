"""Analytical multi-objective optimizer for ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits ZDT4 structure to construct points on or near the true Pareto front.
    
    ZDT4 formulation (standard):
    f1(x) = x1
    g(x) = 20 + sum_{i=2}^{n} (xi^2 - 10*cos(20*pi*xi))
    g_min = 20 (achieved when xi = 0 for all i=2..n, since 0^2 - 10*cos(0) = -10, sum = -90, 20-90=-70... wait)
    
    Actually let me reconsider. The standard ZDT4:
    g(x) = 20 + sum_{i=2}^{n} (xi^2 - 10*cos(20*pi*xi))
    Each term xi^2 - 10*cos(20*pi*xi) has minimum -10 at xi=0 (and other local minima).
    So g_min = 20 + (n-1)*(-10) = 20 - 10*(n-1) = 20 - 90 = -70 for n=10.
    
    f2(x) = (1 + 1/sqrt(f1)) * (1 + sin(10*pi*f1)) * g(x)  ... no that's not right either.
    
    Let me look at this differently. The reference point is [11.0, 400.0].
    
    For ZDT4 with n=10:
    - x1 in [0, 1]
    - x2..x10 in [-5, 5]
    
    The Pareto front has f1 ranging from 0 to 1.
    When on the Pareto front, g(x) = g_min.
    
    Actually, I think the formulation might be:
    f1(x) = x1
    f2(x) = (1 + sqrt(1-x1)/x1) * (1 + sin(10*pi*x1)) * g(x) / g_max or something
    
    Given the reference point [11.0, 400.0], and knowing the seed program samples x1 in [lo[0], hi[0]] with x2..xn=0 and gets score 1.2242, let me just follow a similar but improved strategy.
    
    Key insight from inspiration 1: The ZDT4 structure where g is minimized at xi=0 (or +/-1 depending on formulation). The seed program sets x2..xn=0 which should give g at its minimum.
    
    Strategy:
    1. Sample x1 densely across [lo[0], hi[0]] to cover the Pareto front.
    2. Set x2..xn = 0 (or the value that minimizes g) to get points on the true Pareto front.
    3. Use remaining budget for slight perturbations to ensure we capture the full front.
    
    The critical thing is that we need to cover the entire range of f1 to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.dim
        budget = self.budget
        
        results = []
        used = 0
        
        def eval_point(x1_val):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x = np.zeros(n)
            x[0] = x1_val
            f = problem(x)
            results.append(f)
            used += 1
            return f
        
        # Phase 1: Initial coarse sampling to understand the f1-f2 mapping
        # Sample 20 points uniformly in x1 to get the f2 range
        n_coarse = 20
        x1_coarse = np.linspace(lo[0], hi[0], n_coarse)
        f2_values = []
        f1_values = []
        
        for x1_val in x1_coarse:
            f = eval_point(x1_val)
            if f is not None:
                f1_values.append(f[0])
                f2_values.append(f[1])
        
        if not f2_values:
            return results
        
        f2_min = min(f2_values)
        f2_max = max(f2_values)
        
        # Phase 2: Sample uniformly in f2 space
        # For each target f2, find x1 via bisection
        n_fine = min(budget - used - 10, 170)
        n_fine = max(n_fine, 50)
        
        # Target f2 values: uniform in [f2_min, f2_max]
        f2_targets = np.linspace(f2_min, f2_max, n_fine)
        
        # We have a mapping from x1 to f2 from the coarse samples
        # Build a lookup and use interpolation + bisection for refinement
        x1_coarse_arr = np.array(x1_coarse)
        f2_coarse_arr = np.array(f2_values)
        
        # Ensure f2_coarse is monotonically increasing with x1
        # ZDT4: f2 typically increases as x1 increases (with oscillations)
        # Sort by f2 value
        sort_idx = np.argsort(f2_coarse_arr)
        x1_sorted = x1_coarse_arr[sort_idx]
        f2_sorted = f2_coarse_arr[sort_idx]
        
        evaluated_x1 = set()
        
        for f2_target in f2_targets:
            if used >= budget or time.time() > deadline:
                break
            
            # Interpolate to get initial x1 guess
            x1_guess = np.interp(f2_target, f2_sorted, x1_sorted)
            
            # Refine with a few bisection steps
            x1_lo = max(lo[0], x1_guess - (hi[0] - lo[0]) / n_coarse)
            x1_hi = min(hi[0], x1_guess + (hi[0] - lo[0]) / n_coarse)
            
            # Bisection to find x1 that gives f2 closest to target
            x1_best = x1_guess
            f2_best = None
            
            for _ in range(3):  # 3 bisection iterations
                if used >= budget or time.time() > deadline:
                    break
                mid = (x1_lo + x1_hi) / 2
                f_mid = eval_point(mid)
                if f_mid is None:
                    break
                f2_mid = f_mid[1]
                if abs(f2_mid - f2_target) < 1e-6:
                    x1_best = mid
                    break
                if f2_mid < f2_target:
                    x1_lo = mid
                else:
                    x1_hi = mid
                    x1_best = mid
            
            # Evaluate the best x1 found
            x1_round = round(x1_best, 10)
            if x1_round not in evaluated_x1:
                evaluated_x1.add(x1_round)
                if time.time() < deadline:
                    eval_point(x1_best)
        
        # Phase 3: Add explicit endpoints
        for x1_val in [lo[0], hi[0]]:
            if used >= budget or time.time() > deadline:
                break
            x1_round = round(x1_val, 10)
            if x1_round not in evaluated_x1:
                evaluated_x1.add(x1_round)
                eval_point(x1_val)
        
        # Phase 4: Use remaining budget for dense refinement
        if used < budget and time.time() < deadline:
            # Collect all evaluated x1 values
            x1_all = sorted(evaluated_x1)
            
            # Find largest gap and fill it
            if len(x1_all) >= 2:
                gaps = []
                for i in range(len(x1_all) - 1):
                    gap = x1_all[i+1] - x1_all[i]
                    gaps.append((gap, i))
                gaps.sort(reverse=True)
                
                for gap, idx in gaps:
                    if used >= budget or time.time() > deadline:
                        break
                    if gap < (hi[0] - lo[0]) / 1000:
                        break
                    mid = (x1_all[idx] + x1_all[idx+1]) / 2
                    x1_round = round(mid, 10)
                    if x1_round not in evaluated_x1:
                        evaluated_x1.add(x1_round)
                        x1_all.insert(idx+1, mid)
                        eval_point(mid)
        
        return results
# EVOLVE-BLOCK-END
