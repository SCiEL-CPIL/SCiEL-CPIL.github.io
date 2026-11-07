"""Direct approach for BBOB f22 (Gallagher 21 peaks) optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Coordinate descent with golden-section 1D search for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                val = float(func(x_clipped))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                return best_f

        def time_ok():
            return time.time() < deadline and calls < budget

        def golden_section_1d(x_fixed, axis, lo, hi, max_iter=15):
            """Minimize along one axis using golden-section search."""
            if not time_ok():
                return x_fixed.copy()
            
            inv_phi = (np.sqrt(5) - 1) / 2  # ~0.618
            inv_phi2 = inv_phi * inv_phi
            x1 = x_fixed.copy()
            x2 = x_fixed.copy()
            x1[axis] = lo + inv_phi2 * (hi - lo)
            x2[axis] = lo + inv_phi * (hi - lo)
            
            f1 = eval_and_track(x1)
            f2 = eval_and_track(x2)
            
            for _ in range(max_iter):
                if not time_ok():
                    break
                if f1 < f2:
                    hi = x2[axis]
                    x2 = x1.copy()
                    f2 = f1
                    x1[axis] = lo + inv_phi2 * (hi - lo)
                    f1 = eval_and_track(x1)
                else:
                    lo = x1[axis]
                    x1 = x2.copy()
                    f1 = f2
                    x2[axis] = lo + inv_phi * (hi - lo)
                    f2 = eval_and_track(x2)
            
            # Return the better of the last two points
            if f1 <= f2:
                return x1
            else:
                return x2

        def coordinate_descent(x0, max_sweeps=3):
            """Run coordinate descent with golden-section on each axis."""
            x = x0.copy()
            for sweep in range(max_sweeps):
                if not time_ok():
                    break
                for axis in range(dim):
                    if not time_ok():
                        break
                    x = golden_section_1d(x, axis, -5.0, 5.0, max_iter=12)
            return x

        # Phase 1: Broad exploration with stratified sampling
        n_initial = min(60, budget)
        
        # Stratified sampling: 2 bins per dimension
        initial_points = []
        n_strata = 2 ** dim
        points_per_stratum = max(1, n_initial // n_strata)
        
        for s in range(n_strata):
            bins = []
            temp = s
            for d in range(dim):
                bins.append(temp % 2)
                temp //= 2
            bins = np.array(bins)
            
            for _ in range(points_per_stratum):
                pt = np.zeros(dim)
                for d in range(dim):
                    if bins[d] == 0:
                        pt[d] = rng.uniform(-5.0, 0.0)
                    else:
                        pt[d] = rng.uniform(0.0, 5.0)
                initial_points.append(pt)
        
        # Add center and structured points
        initial_points.append(np.zeros(dim))
        for d in range(dim):
            for v in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[d] = v
                initial_points.append(pt)
        
        for x in initial_points:
            if not time_ok():
                break
            eval_and_track(x)
        
        if not time_ok():
            return best_x, best_f

        # Phase 2: Coordinate descent from best point
        if time_ok():
            x_cd = coordinate_descent(best_x.copy(), max_sweeps=4)
            # Track the result
            f_cd = eval_and_track(x_cd)
        
        if not time_ok():
            return best_x, best_f

        # Phase 3: Multi-start coordinate descent
        # Generate several starting points and run coordinate descent from each
        n_starts = 12
        sigmas = [2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 2.0, 1.5, 1.0, 0.8, 0.5, 0.3]
        
        for i in range(n_starts):
            if not time_ok():
                break
            
            sigma = sigmas[i % len(sigmas)]
            if i % 2 == 0:
                x0 = best_x + rng.normal(0, sigma, size=dim)
            else:
                x0 = best_x + rng.uniform(-sigma, sigma, size=dim)
            
            x0 = np.clip(x0, -5.0, 5.0)
            
            # Run coordinate descent with fewer sweeps for multi-start
            x_cd = coordinate_descent(x0, max_sweeps=2)
            eval_and_track(x_cd)
        
        if not time_ok():
            return best_x, best_f

        # Phase 4: Fine-tuning with smaller step coordinate descent
        # Use a local search around best_x with tighter bounds
        sigma_local = 0.3
        n_fine = 8
        for i in range(n_fine):
            if not time_ok():
                break
            
            x0 = best_x + rng.normal(0, sigma_local, size=dim)
            x0 = np.clip(x0, -5.0, 5.0)
            
            # Tighter bounds for local refinement
            lo = np.maximum(best_x - sigma_local * 2, -5.0)
            hi = np.minimum(best_x + sigma_local * 2, 5.0)
            
            x = x0.copy()
            for axis in range(dim):
                if not time_ok():
                    break
                x = golden_section_1d(x, axis, lo[axis], hi[axis], max_iter=10)
            
            eval_and_track(x)
        
        if not time_ok():
            return best_x, best_f

        # Phase 5: Final polish - one more full coordinate descent sweep
        if time_ok():
            x_cd = coordinate_descent(best_x.copy(), max_sweeps=2)
            eval_and_track(x_cd)

        return best_x, best_f
# EVOLVE-BLOCK-END
