"""Direct approach for BBOB f22 (Gallagher 21 peaks) optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start coordinate descent + Nelder-Mead optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

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

        # Phase 1: Latin Hypercube Sampling for initial exploration
        n_phase1 = 80
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=dim, seed=42)
        samples = sampler.random(n=n_phase1)
        lhs_points = samples * 10.0 - 5.0

        # Add structured points
        structured = [np.zeros(dim)]
        for d in range(dim):
            for v in [-4.0, -2.0, 0.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[d] = v
                structured.append(pt)

        all_initial = structured + list(lhs_points)
        for x in all_initial:
            if not time_ok():
                break
            eval_and_track(x)

        # Phase 2: Coordinate descent refinement on the best point
        # Use golden-section search along each coordinate
        def golden_section_1d(f, x, coord, lo, hi, tol=1e-6):
            """Golden-section search along one coordinate."""
            if not time_ok():
                return x[coord]
            
            gr = (np.sqrt(5) - 1) / 2
            c = 1 - gr
            a = lo
            b = hi
            
            # Initial points
            x1 = a + c * (b - a)
            x2 = a + gr * (b - a)
            
            # Evaluate
            x_test1 = x.copy()
            x_test1[coord] = x1
            x_test2 = x.copy()
            x_test2[coord] = x2
            
            f1 = eval_and_track(x_test1)
            f2 = eval_and_track(x_test2)
            
            # Iterate
            max_iter = 30
            for _ in range(max_iter):
                if not time_ok():
                    break
                if b - a < tol:
                    break
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = a + c * (b - a)
                    x_test1 = x.copy()
                    x_test1[coord] = x1
                    f1 = eval_and_track(x_test1)
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = a + gr * (b - a)
                    x_test2 = x.copy()
                    x_test2[coord] = x2
                    f2 = eval_and_track(x_test2)
            
            # Return midpoint of final interval
            return (a + b) / 2

        def coordinate_descent(x0, max_rounds=10):
            """Cycle through coordinates with golden-section search."""
            x = x0.copy()
            f_curr = eval_and_track(x)
            
            for _ in range(max_rounds):
                if not time_ok():
                    break
                improved = False
                for coord in range(dim):
                    if not time_ok():
                        break
                    # Determine search range for this coordinate
                    # Use a range around the current value
                    center = x[coord]
                    half_width = 2.0
                    lo = max(-5.0, center - half_width)
                    hi = min(5.0, center + half_width)
                    
                    new_val = golden_section_1d(eval_and_track, x, coord, lo, hi)
                    if abs(new_val - x[coord]) > 1e-8:
                        x[coord] = new_val
                        improved = True
                
                if not improved:
                    break
            
            return x

        # Apply coordinate descent to the best point
        if time_ok():
            x_cd = coordinate_descent(best_x.copy())
            f_cd = eval_and_track(x_cd)
            if f_cd < best_f:
                best_f = f_cd
                best_x = x_cd.copy()

        # Phase 3: Nelder-Mead refinement
        if time_ok():
            try:
                res = minimize(
                    eval_and_track,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
            except Exception:
                pass

        # Phase 4: If budget remains, try additional random restarts with local refinement
        if calls < budget - 30 and time_ok():
            n_restarts = 5
            for _ in range(n_restarts):
                if not time_ok():
                    break
                # Generate a random point near the current best or a new random point
                if rng.random() < 0.5:
                    x_new = best_x + rng.normal(0, 1.0, size=dim)
                else:
                    x_new = rng.uniform(-5, 5, size=dim)
                x_new = np.clip(x_new, -5, 5)
                
                f_new = eval_and_track(x_new)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                
                # Quick coordinate descent from this point
                if time_ok() and f_new < best_f + 10:
                    x_cd2 = coordinate_descent(x_new.copy(), max_rounds=3)
                    f_cd2 = eval_and_track(x_cd2)
                    if f_cd2 < best_f:
                        best_f = f_cd2
                        best_x = x_cd2.copy()

        # Phase 5: Final Nelder-Mead polish
        if time_ok():
            try:
                res = minimize(
                    eval_and_track,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
