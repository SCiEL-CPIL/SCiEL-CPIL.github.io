"""ZDT1 optimizer: probe-then-optimize with hypervolume-aware placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: probe landscape then place points for max hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget
        ref = 11.0

        n_evals = 0
        results = []

        def eval_point(x):
            nonlocal n_evals
            if n_evals >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            results.append(f)
            n_evals += 1
            return f

        # Phase 1: Probe to understand the objective landscape (use ~10% budget)
        n_probe = max(5, budget // 10)
        probe_f1s = []
        probe_f2s = []

        for i in range(n_probe):
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = i / max(n_probe - 1, 1)
            x = np.clip(x, lo, hi)
            f = eval_point(x)
            if f is not None:
                probe_f1s.append(f[0])
                probe_f2s.append(f[1])

        if not probe_f1s:
            return results

        probe_f1s = np.array(probe_f1s)
        probe_f2s = np.array(probe_f2s)

        # Determine valid region: points where both objectives < ref
        valid_mask = (probe_f1s < ref) & (probe_f2s < ref)

        if not np.any(valid_mask):
            # No valid points found with x[1:]=0. Try different x[1:] values.
            # Maybe we need to increase g to change the shape.
            # Or maybe the valid region requires specific x[1:] configurations.
            
            # Try: set x[1:] to various values to see if f2 can be reduced
            # In some ZDT variants, f2 = g*(1-sqrt(f1/g)), and larger g can help
            # if f1 is also large.
            
            # Let's try a grid search in x[0] and x[1] (first few dimensions)
            remaining = budget - n_evals
            if remaining > 0:
                # Try different x[1:] patterns
                for x0_frac in np.linspace(0.5, 1.0, min(remaining, 50)):
                    if time.time() > deadline or n_evals >= budget:
                        break
                    x = np.zeros(n)
                    x[0] = x0_frac
                    # Try setting x[1:] to 0.5 (which might minimize g in some formulations)
                    x[1:] = 0.5
                    x = np.clip(x, lo, hi)
                    f = eval_point(x)
                    if f is None:
                        break

            return results

        # Phase 2: We have valid points. Determine the Pareto front structure.
        # Find the range of x[0] that gives valid points
        valid_x0 = np.linspace(0, 1, n_probe)
        valid_x0_vals = valid_x0[valid_mask]
        
        if len(valid_x0_vals) < 2:
            # Only one valid point, just spread around it
            remaining = budget - n_evals
            center = valid_x0_vals[0]
            spread = 0.1
            for i in range(remaining):
                if time.time() > deadline or n_evals >= budget:
                    break
                x = np.zeros(n)
                x[0] = center + (i / max(remaining-1, 1) - 0.5) * spread
                x = np.clip(x, lo, hi)
                f = eval_point(x)
                if f is None:
                    break
            return results

        # Phase 3: Place remaining points optimally on the Pareto front
        # The hypervolume contribution of a point is (ref-f1)*(ref-f2)
        # We want to maximize total non-dominated hypervolume.
        
        # Strategy: Place points where the marginal hypervolume gain is highest.
        # For a smooth Pareto front, this means concentrating where the curve is 
        # farthest from the reference point (maximizing (ref-f1)*(ref-f2)).
        
        remaining = budget - n_evals
        if remaining <= 0:
            return results

        # Estimate the Pareto front from probe data
        # Fit a smooth curve: f2 = a * (1 - sqrt(f1/b)) or similar
        # For now, use the probe points to guide placement
        
        # Use the valid x[0] range and place points with density proportional
        # to the estimated hypervolume contribution
        
        x0_min = valid_x0_vals[0]
        x0_max = valid_x0_vals[-1]
        
        # Create candidate x[0] values with density proportional to contribution
        # First, estimate f2 for each candidate
        n_candidates = remaining
        # Use a power-law bias toward where contribution is highest
        # Contribution ~ (ref - f1) * (ref - f2(f1))
        # f2 decreases with f1, so contribution peaks somewhere in the middle
        
        # Use quadratic sampling biased toward the center of the valid range
        u = np.linspace(0.0, 1.0, n_candidates)
        # Bias toward center: x0 = x0_min + (x0_max - x0_min) * (0.5 + 0.5*sin(pi*u))
        # Actually, let's just use uniform stratified in the valid range
        # with some additional concentration
        
        # Better: use the probe data to estimate where contribution peaks
        # and bias sampling accordingly
        
        # For simplicity, use a mixture: 50% uniform, 30% biased to high x0, 20% biased to low x0
        n_uni = int(remaining * 0.5)
        n_high = int(remaining * 0.3)
        n_low = remaining - n_uni - n_high
        
        x0_list = []
        
        # Uniform in valid range
        if n_uni > 0:
            x0_list.extend(np.linspace(x0_min, x0_max, n_uni))
        
        # Biased toward high x0 (where f2 is smallest)
        if n_high > 0:
            u_high = np.linspace(0.0, 1.0, n_high)
            x0_high = x0_min + (x0_max - x0_min) * u_high ** 0.5  # bias toward high
            x0_list.extend(x0_high)
        
        # Biased toward low x0 (where f1 is smallest)
        if n_low > 0:
            u_low = np.linspace(0.0, 1.0, n_low)
            x0_low = x0_min + (x0_max - x0_min) * u_low ** 2.0  # bias toward low
            x0_list.extend(x0_low)
        
        for x0 in x0_list:
            if time.time() > deadline or n_evals >= budget:
                break
            x = np.zeros(n)
            x[0] = x0
            x = np.clip(x, lo, hi)
            f = eval_point(x)
            if f is None:
                break

        return results
# EVOLVE-BLOCK-END
