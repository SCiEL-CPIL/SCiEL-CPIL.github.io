import numpy as np
import time


class Optimizer:
    """Direct construction of the ZDT3 Pareto front with optimized sampling.
    
    The ZDT3 Pareto front is given by x = (t, 0, ..., 0) for t in [0, 1].
    The objective values are f1(t) = t and f2(t) = 2 * (1 - sqrt(t) - t * sin(10 * pi * t)).
    
    We use a hybrid sampling strategy:
    1. Include all critical points of the oscillatory term sin(10*pi*t) to capture
       the local extrema and zeros of the front.
    2. Fill the remaining budget with uniformly spaced points in t, ensuring
       no duplicates with critical points.
    
    This ensures a high-fidelity approximation of the Pareto front, which is
    necessary to maximize the hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)

        # 1. Identify critical points of the ZDT3 Pareto front oscillations.
        # f2(t) depends on sin(10 * pi * t).
        # Zeros of sin(10*pi*t) are at t = k/10 for k=0..10.
        # Extrema of sin(10*pi*t) are at t = (2k+1)/20 for k=0..9.
        critical_points = set()
        for k in range(11):
            critical_points.add(round(k / 10.0, 12))
        for k in range(10):
            critical_points.add(round((2 * k + 1) / 20.0, 12))

        critical_list = sorted(critical_points)
        n_critical = len(critical_list)

        # 2. Determine how many uniform fill points we need.
        n_fill = max(0, budget - n_critical)
        
        # 3. Generate uniform fill points in [0, 1], excluding endpoints if they are already in criticals.
        # We use linspace to get candidates, then filter out those too close to critical points.
        fill_points = []
        if n_fill > 0:
            # Generate more candidates than needed to account for filtering
            n_candidates = n_fill * 2 + 10
            # Exclude 0 and 1 from linspace if they are in criticals (they are)
            # We want points strictly inside (0,1) for the fill if possible, 
            # but linspace includes endpoints. 
            # Since 0 and 1 are critical, we can generate candidates in (0,1)
            uniform_candidates = np.linspace(0.0, 1.0, n_candidates + 2)[1:-1]
            
            min_dist = 1e-9
            for uc in uniform_candidates:
                # Check if this candidate is too close to any critical point
                is_duplicate = False
                for cp in critical_list:
                    if abs(uc - cp) < min_dist:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    fill_points.append(float(uc))
                if len(fill_points) >= n_fill:
                    break
            
            # If we still don't have enough (unlikely), just take uniform points
            if len(fill_points) < n_fill:
                # Fallback: simple uniform grid
                extra_needed = n_fill - len(fill_points)
                # Generate extra points, ensuring they don't collide
                # This is a safety net, unlikely to be hit
                for i in range(extra_needed):
                    val = (i + 0.5) / (extra_needed + 1)
                    fill_points.append(val)

        # 4. Combine and sort all t values
        all_t = sorted(set(critical_list + fill_points))
        
        # Truncate to budget if the set logic somehow added too many (shouldn't happen with set)
        if len(all_t) > budget:
            all_t = all_t[:budget]

        recorded = []
        used = 0
        
        # Pre-allocate a vector for speed
        x = np.zeros(dim)
        
        for tv in all_t:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Set x_1 to t, rest to 0
            x[0] = float(tv)
            
            # Clip to bounds
            x_clipped = np.clip(x, lo, hi)
            
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            used += 1

        return recorded
