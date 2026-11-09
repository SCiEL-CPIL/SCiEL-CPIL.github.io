import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def powell_method(x0, max_iter=50):
            """Powell's method for local optimization."""
            nonlocal calls, best_x, best_f
            x = x0.copy()
            fx = try_call(x)
            if fx is None:
                return x, fx
            
            # Set up initial directions (coordinate axes)
            directions = np.eye(dim)
            x_prev = x.copy()
            
            for iteration in range(max_iter):
                if calls >= budget or time.time() >= deadline:
                    break
                
                delta_min = 0.0
                x_min = x.copy()
                f_min = fx
                
                for i in range(dim):
                    if calls >= budget or time.time() >= deadline:
                        break
                    
                    # Minimize along direction i
                    xi = x.copy()
                    fi = fx
                    best_t = 0.0
                    best_fi = fi
                    best_xi = xi.copy()
                    
                    # Golden section search along direction i
                    # First, find a bracket
                    t = 1.0
                    ft = try_call(xi + t * directions[i])
                    if ft is None:
                        break
                    
                    if ft < fi:
                        # Try to expand
                        t2 = 2.0 * t
                        ft2 = try_call(xi + t2 * directions[i])
                        if ft2 is None:
                            break
                        if ft2 < ft:
                            t3 = 3.0 * t
                            ft3 = try_call(xi + t3 * directions[i])
                            if ft3 is None:
                                break
                            if ft3 < ft2:
                                # Use t3 as the best found
                                best_t = t3
                                best_fi = ft3
                                best_xi = xi + t3 * directions[i]
                            else:
                                # Golden section between t and t3
                                lo, hi = t, t3
                                f_lo, f_hi = ft, ft3
                                for _ in range(10):
                                    if calls >= budget or time.time() >= deadline:
                                        break
                                    gr = (np.sqrt(5.0) - 1.0) / 2.0
                                    t1 = lo + (1.0 - gr) * (hi - lo)
                                    t2 = lo + gr * (hi - lo)
                                    ft1 = try_call(xi + t1 * directions[i])
                                    if ft1 is None:
                                        break
                                    ft2 = try_call(xi + t2 * directions[i])
                                    if ft2 is None:
                                        break
                                    if ft1 < ft2:
                                        hi = t2
                                        f_hi = ft2
                                    else:
                                        lo = t1
                                        f_lo = ft1
                                    if abs(hi - lo) < 1e-6:
                                        break
                                mid = (lo + hi) / 2.0
                                fm = try_call(xi + mid * directions[i])
                                if fm is not None:
                                    if fm < best_fi:
                                        best_t = mid
                                        best_fi = fm
                                        best_xi = xi + mid * directions[i]
                        else:
                            # Golden section between 0 and t2
                            lo, hi = 0.0, t2
                            f_lo, f_hi = fi, ft2
                            for _ in range(10):
                                if calls >= budget or time.time() >= deadline:
                                    break
                                gr = (np.sqrt(5.0) - 1.0) / 2.0
                                t1 = lo + (1.0 - gr) * (hi - lo)
                                t2 = lo + gr * (hi - lo)
                                ft1 = try_call(xi + t1 * directions[i])
                                if ft1 is None:
                                    break
                                ft2 = try_call(xi + t2 * directions[i])
                                if ft2 is None:
                                    break
                                if ft1 < ft2:
                                    hi = t2
                                    f_hi = ft2
                                else:
                                    lo = t1
                                    f_lo = ft1
                                if abs(hi - lo) < 1e-6:
                                    break
                            mid = (lo + hi) / 2.0
                            fm = try_call(xi + mid * directions[i])
                            if fm is not None:
                                if fm < best_fi:
                                    best_t = mid
                                    best_fi = fm
                                    best_xi = xi + mid * directions[i]
                    else:
                        # Try negative direction
                        t = -1.0
                        ft = try_call(xi + t * directions[i])
                        if ft is not None and ft < fi:
                            t2 = 2.0 * t
                            ft2 = try_call(xi + t2 * directions[i])
                            if ft2 is None:
                                break
                            if ft2 < ft:
                                t3 = 3.0 * t
                                ft3 = try_call(xi + t3 * directions[i])
                                if ft3 is None:
                                    break
                                if ft3 < ft2:
                                    best_t = t3
                                    best_fi = ft3
                                    best_xi = xi + t3 * directions[i]
                                else:
                                    lo, hi = t3, 0.0
                                    f_lo, f_hi = ft3, fi
                                    for _ in range(10):
                                        if calls >= budget or time.time() >= deadline:
                                            break
                                        gr = (np.sqrt(5.0) - 1.0) / 2.0
                                        t1 = lo + (1.0 - gr) * (hi - lo)
                                        t2 = lo + gr * (hi - lo)
                                        ft1 = try_call(xi + t1 * directions[i])
                                        if ft1 is None:
                                            break
                                        ft2 = try_call(xi + t2 * directions[i])
                                        if ft2 is None:
                                            break
                                        if ft1 < ft2:
                                            hi = t2
                                            f_hi = ft2
                                        else:
                                            lo = t1
                                            f_lo = ft1
                                        if abs(hi - lo) < 1e-6:
                                            break
                                    mid = (lo + hi) / 2.0
                                    fm = try_call(xi + mid * directions[i])
                                    if fm is not None:
                                        if fm < best_fi:
                                            best_t = mid
                                            best_fi = fm
                                            best_xi = xi + mid * directions[i]
                            else:
                                lo, hi = 0.0, t2
                                f_lo, f_hi = fi, ft2
                                for _ in range(10):
                                    if calls >= budget or time.time() >= deadline:
                                        break
                                    gr = (np.sqrt(5.0) - 1.0) / 2.0
                                    t1 = lo + (1.0 - gr) * (hi - lo)
                                    t2 = lo + gr * (hi - lo)
                                    ft1 = try_call(xi + t1 * directions[i])
                                    if ft1 is None:
                                        break
                                    ft2 = try_call(xi + t2 * directions[i])
                                    if ft2 is None:
                                        break
                                    if ft1 < ft2:
                                        hi = t2
                                        f_hi = ft2
                                    else:
                                        lo = t1
                                        f_lo = ft1
                                    if abs(hi - lo) < 1e-6:
                                        break
                                mid = (lo + hi) / 2.0
                                fm = try_call(xi + mid * directions[i])
                                if fm is not None:
                                    if fm < best_fi:
                                        best_t = mid
                                        best_fi = fm
                                        best_xi = xi + mid * directions[i]
                    
                    if best_fi < f_min:
                        delta_min += (f_min - best_fi)
                        f_min = best_fi
                        x_min = best_xi.copy()
                    
                    x = best_xi.copy()
                    fx = best_fi
                    if best_t != 0.0:
                        directions[i] = best_xi - xi + (best_t / t) * (xi - x_prev) if t != 0 else best_xi - x_prev
                
                # Check convergence
                if abs(delta_min) < 1e-10:
                    break
                
                # Update best direction
                new_dir = x_min - x_prev
                norm_new = np.linalg.norm(new_dir)
                if norm_new > 1e-10:
                    # Find the direction that gave the largest decrease
                    max_decrease = -1e-10
                    max_dir_idx = 0
                    for i in range(dim):
                        d = x_min - (x_min - directions[i] * 0.0)  # This is just x_min
                        # Actually, we need to track which direction gave the most improvement
                        pass
                    # Replace the direction that gave the least improvement with the new direction
                    # For simplicity, replace the last direction
                    directions[-1] = new_dir / norm_new
                
                x_prev = x.copy()
                fx = f_min
            
            if f_min < best_f:
                best_f = f_min
                best_x = x_min.copy()
            
            return x_min, f_min

        # Phase 1: Coarse grid search to identify promising basins
        # Use a 3^5 grid over [-4, 4]^5
        n_grid = 3
        grid_points = []
        for i in range(n_grid ** dim):
            point = []
            idx = i
            for d in range(dim):
                idx, digit = divmod(idx, n_grid)
                # Map digit 0,1,2 to -4, 0, 4
                point.append(-4.0 + 4.0 * digit)
            grid_points.append(np.array(point))
        
        # Evaluate grid points
        for gp in grid_points:
            if calls >= budget or time.time() >= deadline:
                break
            try_call(gp)
        
        # Phase 1b: Refine around the best grid points
        if best_x is not None and calls < budget and time.time() < deadline:
            # Sort grid points by their function values (we only stored best)
            # Instead, do local search around best and a few nearby points
            n_local_starts = min(5, budget - calls - 100)
            if n_local_starts > 0:
                # Generate local starting points near the best
                local_starts = [best_x.copy()]
                for i in range(1, n_local_starts):
                    # Random perturbation around best
                    pert = best_x + rng.standard_normal(dim) * 1.0
                    pert = np.clip(pert, -4.5, 4.5)
                    local_starts.append(pert)
                
                # Also add some random points in the box for diversity
                for i in range(min(3, n_local_starts)):
                    rp = rng.uniform(-4, 4, size=dim)
                    local_starts.append(rp)
                
                # Run Powell's method from each local start
                for ls in local_starts:
                    if calls >= budget or time.time() >= deadline:
                        break
                    powell_method(ls, max_iter=30)

        # Phase 2: Final fine-tuning with small random perturbations
        if best_x is not None and calls < budget and time.time() < deadline:
            step_size = 0.1
            no_improve = 0
            max_no_improve = 30
            
            while calls < budget and time.time() < deadline and no_improve < max_no_improve:
                y = best_x + step_size * rng.standard_normal(dim)
                fy = try_call(y)
                if fy is None:
                    break
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    no_improve = 0
                    step_size *= 0.9
                else:
                    no_improve += 1
                    step_size *= 0.85
                    if step_size < 1e-7:
                        break

        return best_x, best_f
