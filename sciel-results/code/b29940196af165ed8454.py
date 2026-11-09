"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start coordinate line search for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def golden_section_1d(x, axis, fval, max_iter=20):
            """Perform 1D golden-section search along a given axis."""
            if calls_made[0] >= budget:
                return x, fval
            
            # Determine initial bracket
            lo = -5.0
            hi = 5.0
            
            # Try to narrow the bracket using the current point
            # Use a small step to find a better direction
            step = 0.5
            x_test = x.copy()
            x_test[axis] = x[axis] + step
            f_test = wrapped_func(x_test)
            
            if f_test < fval:
                # Move in positive direction
                lo = x[axis]
                hi = x_test[axis]
                x = x_test
                fval = f_test
            else:
                x_test = x.copy()
                x_test[axis] = x[axis] - step
                f_test = wrapped_func(x_test)
                if f_test < fval:
                    # Move in negative direction
                    hi = x[axis]
                    lo = x_test[axis]
                    x = x_test
                    fval = f_test
                else:
                    # No improvement, try smaller steps
                    for s in [0.2, 0.1, 0.05]:
                        x_test = x.copy()
                        x_test[axis] = x[axis] + s
                        f_test = wrapped_func(x_test)
                        if f_test < fval:
                            lo = x[axis]
                            hi = x_test[axis]
                            x = x_test
                            fval = f_test
                            break
                        x_test = x.copy()
                        x_test[axis] = x[axis] - s
                        f_test = wrapped_func(x_test)
                        if f_test < fval:
                            hi = x[axis]
                            lo = x_test[axis]
                            x = x_test
                            fval = f_test
                            break
                    else:
                        return x, fval
            
            # Golden-section search
            if calls_made[0] >= budget:
                return x, fval
                
            gr = (np.sqrt(5) - 1) / 2  # Golden ratio conjugate ~0.618
            
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            
            x1_pt = x.copy()
            x1_pt[axis] = x1
            f1 = wrapped_func(x1_pt)
            
            if calls_made[0] >= budget:
                return x, fval
                
            x2_pt = x.copy()
            x2_pt[axis] = x2
            f2 = wrapped_func(x2_pt)
            
            for _ in range(max_iter):
                if calls_made[0] >= budget:
                    break
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    x2_pt = x1_pt
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    x1_pt = x.copy()
                    x1_pt[axis] = x1
                    f1 = wrapped_func(x1_pt)
                else:
                    lo = x1
                    x1 = x2
                    x1_pt = x2_pt
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    x2_pt = x.copy()
                    x2_pt[axis] = x2
                    f2 = wrapped_func(x2_pt)
                
                if abs(hi - lo) < 1e-6:
                    break
            
            # Update x with the best found
            if f1 < f2:
                x = x1_pt
                fval = f1
            else:
                x = x2_pt
                fval = f2
            
            return x, fval

        def coordinate_descent(x, fval, max_rounds=5):
            """Perform coordinate-wise line search."""
            for _ in range(max_rounds):
                if calls_made[0] >= budget:
                    break
                improved = False
                for axis in range(n):
                    if calls_made[0] >= budget:
                        break
                    old_f = fval
                    x, fval = golden_section_1d(x, axis, fval)
                    if fval < old_f - 1e-10:
                        improved = True
                if not improved:
                    break
            return x, fval

        # Generate initial points: grid + center
        # Use a coarse grid to cover the space
        init_points = []
        
        # Center
        init_points.append(np.zeros(n))
        
        # Grid points at -3, 0, 3 in each dimension (3^5 = 243 points, too many)
        # Use a sparser set: ±2 and 0
        for d1 in [-2.0, 0.0, 2.0]:
            for d2 in [-2.0, 0.0, 2.0]:
                for d3 in [-2.0, 0.0, 2.0]:
                    for d4 in [-2.0, 0.0, 2.0]:
                        for d5 in [-2.0, 0.0, 2.0]:
                            init_points.append(np.array([d1, d2, d3, d4, d5]))
        
        # Add some random points for diversity
        rng = np.random.default_rng(42)
        for _ in range(20):
            init_points.append(rng.uniform(-4, 4, n))
        
        # Sort initial points by distance to origin (prioritize center)
        init_points.sort(key=lambda p: np.linalg.norm(p))
        
        # Process each initial point
        for init_x in init_points:
            if calls_made[0] >= budget or time.time() > deadline:
                break
            
            # Evaluate initial point
            f_init = wrapped_func(init_x)
            
            if f_init < best_f:
                best_f = f_init
                best_x = init_x.copy()
            
            # Perform coordinate descent
            x_curr = init_x.copy()
            f_curr = f_init
            x_curr, f_curr = coordinate_descent(x_curr, f_curr)
            
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr.copy()
            
            # If we've used a lot of budget, stop
            if calls_made[0] > budget * 0.8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
