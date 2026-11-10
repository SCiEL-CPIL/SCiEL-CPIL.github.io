# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in unit square.
    Returns (centers, sides) where centers is (10, 2) and sides is (10,).
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin

    def _is_valid(centers, sides, tol=1e-7):
        n = len(sides)
        if n != 10:
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                if dx < t and dy < t:
                    return False
        return True

    def _voronoi_limit(cx, cy, i):
        """
        Compute the maximum side length for square i such that it doesn't overlap
        with any other square and stays within the unit square.
        For square i with center (cx[i], cy[i]), the constraint from square j is:
            If we increase s_i, we need |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2
        The limiting factor is the minimum over all j of:
            max(2*(|cx_i - cx_j| - s_j/2), 2*(|cy_i - cy_j| - s_j/2))  (if positive)
        And boundary limits: 2 * min(cx_i, 1-cx_i, cy_i, 1-cy_i)
        """
        # Boundary limits
        limit = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i]) * 2.0
        
        for j in range(10):
            if i == j:
                continue
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            # For square i to not overlap with square j:
            # dx >= (s_i + s_j)/2 OR dy >= (s_i + s_j)/2
            # => s_i <= 2*dx - s_j OR s_i <= 2*dy - s_j
            # So the limit from this pair is max(2*dx - s_j, 2*dy - s_j)
            lim_j = max(2.0 * dx - sides[j], 2.0 * dy - sides[j])
            if lim_j < limit:
                limit = lim_j
        
        return max(0.0, limit)

    def _objective_voronoi(x, beta=20.0):
        """
        Objective: -sum(sides) where sides are determined by Voronoi limits.
        We optimize centers only; sides are derived.
        x layout: [cx0..cx9, cy0..cy9]
        """
        cx = x[0:10]
        cy = x[10:20]
        
        # Compute Voronoi-limited side lengths
        s = np.zeros(10)
        for i in range(10):
            limit = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i]) * 2.0
            for j in range(10):
                if i == j:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # s_i <= 2*dx - s_j OR s_i <= 2*dy - s_j
                # But s_j is also variable... This is coupled.
                # For a smooth approximation, we use the current s_j values
                # This is a fixed-point / Gauss-Seidel style approach
                lim_j = max(2.0 * dx - s[j], 2.0 * dy - s[j])
                if lim_j < limit:
                    limit = lim_j
            s[i] = max(0.0, limit)
        
        return -np.sum(s)

    def _objective_voronoi_smooth(x, beta=30.0):
        """
        Smooth version: use soft-max instead of hard max for the Voronoi limits.
        The constraint is: s_i <= max(2*dx - s_j, 2*dy - s_j) for each j.
        """
        cx = x[0:10]
        cy = x[10:20]
        
        # First pass: compute initial s
        s = np.zeros(10)
        for i in range(10):
            limit = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i]) * 2.0
            s[i] = max(0.0, limit)
        
        # Iterate a few times for fixed point
        for _ in range(5):
            for i in range(10):
                limit = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i]) * 2.0
                for j in range(10):
                    if i == j:
                        continue
                    dx = abs(cx[i] - cx[j])
                    dy = abs(cy[i] - cy[j])
                    # Smooth max of (2*dx - s[j], 2*dy - s[j])
                    a = 2.0 * dx - s[j]
                    b = 2.0 * dy - s[j]
                    # soft max
                    m = max(a, b)
                    sm = m + np.logaddexp(beta * (a - m), beta * (b - m)) / beta
                    if sm < limit:
                        limit = sm
                s[i] = max(0.0, limit)
        
        return -np.sum(s)

    def _repair_and_clamp(centers, sides):
        """
        Ensure validity by shrinking sides if necessary.
        Returns valid (centers, sides).
        """
        centers = centers.copy()
        sides = sides.copy()
        
        # Clamp centers to be within [s/2, 1-s/2]
        for i in range(10):
            h = sides[i] / 2.0
            centers[i, 0] = np.clip(centers[i, 0], h, 1.0 - h)
            centers[i, 1] = np.clip(centers[i, 1], h, 1.0 - h)
            
        # Iteratively shrink squares to remove overlaps
        for _ in range(200):
            overlap_found = False
            for i in range(10):
                for j in range(i+1, 10):
                    t = (sides[i] + sides[j]) / 2.0
                    dx = abs(centers[i,0] - centers[j,0])
                    dy = abs(centers[i,1] - centers[j,1])
                    if dx < t and dy < t:
                        overlap_found = True
                        # Shrink the smaller square by 15%
                        if sides[i] < sides[j]:
                            sides[i] *= 0.85
                        else:
                            sides[j] *= 0.85
                        # Re-clamp centers for the shrunk square
                        if sides[i] < sides[j]:
                            h = sides[i] / 2.0
                            centers[i, 0] = np.clip(centers[i, 0], h, 1.0 - h)
                            centers[i, 1] = np.clip(centers[i, 1], h, 1.0 - h)
                        else:
                            h = sides[j] / 2.0
                            centers[j, 0] = np.clip(centers[j, 0], h, 1.0 - h)
                            centers[j, 1] = np.clip(centers[j, 1], h, 1.0 - h)
            if not overlap_found:
                break
        
        return centers, sides

    def _optimize_centers_voronoi(cx0, cy0, max_time=30.0):
        """
        Optimize center positions to maximize sum of Voronoi-limited side lengths.
        """
        x0 = np.concatenate([cx0, cy0])
        
        bounds = [(0.0, 1.0)] * 20
        
        t0 = time.time()
        call_count = [0]
        
        def callback(xk):
            call_count[0] += 1
            if time.time() - t0 > max_time:
                raise RuntimeError("Time limit")
        
        try:
            res = minimize(_objective_voronoi_smooth, x0, method='SLSQP', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-10, 'maxfun': 500},
                           callback=callback)
            
            x_opt = res.x
            new_cx = x_opt[0:10]
            new_cy = x_opt[10:20]
            
            # Compute final side lengths
            sides = np.zeros(10)
            for i in range(10):
                limit = min(new_cx[i], 1.0 - new_cx[i], new_cy[i], 1.0 - new_cy[i]) * 2.0
                for j in range(10):
                    if i == j:
                        continue
                    dx = abs(new_cx[i] - new_cx[j])
                    dy = abs(new_cy[i] - new_cy[j])
                    lim_j = max(2.0 * dx - sides[j], 2.0 * dy - sides[j])
                    if lim_j < limit:
                        limit = lim_j
                sides[i] = max(0.01, limit)
            
            centers = np.column_stack([new_cx, new_cy])
            centers, sides = _repair_and_clamp(centers, sides)
            
            if _is_valid(centers, sides):
                return centers, sides
        except Exception:
            pass
        
        return None

    def _optimize_full(x0, max_time=40.0, method='SLSQP'):
        """
        Full optimization of centers and sides with penalty method.
        """
        bounds = []
        for _ in range(10):
            bounds.append((0.0, 1.0))
        for _ in range(10):
            bounds.append((0.0, 1.0))
        for _ in range(10):
            bounds.append((0.01, 1.0))
        
        t0 = time.time()
        
        def callback(xk):
            if time.time() - t0 > max_time:
                raise RuntimeError("Time limit")
        
        def _obj(x, penalty_weight=500.0):
            cx = x[0:10]
            cy = x[10:20]
            s = x[20:30]
            
            obj = -np.sum(s)
            
            # Boundary penalties
            for i in range(10):
                for v in [cx[i] - s[i]/2.0, (1.0 - s[i]/2.0) - cx[i],
                          cy[i] - s[i]/2.0, (1.0 - s[i]/2.0) - cy[i]]:
                    if v < 0:
                        obj += penalty_weight * v * v
            
            # Overlap penalties
            beta = 30.0
            for i in range(10):
                for j in range(i+1, 10):
                    dx = abs(cx[i] - cx[j])
                    dy = abs(cy[i] - cy[j])
                    t = (s[i] + s[j]) / 2.0
                    ov_x = t - dx
                    ov_y = t - dy
                    smax_u = 0.5 * (ov_x + np.sqrt(ov_x**2 + 1e-6))
                    smax_v = 0.5 * (ov_y + np.sqrt(ov_y**2 + 1e-6))
                    obj += penalty_weight * smax_u * smax_v
            
            return obj
        
        try:
            if method == 'SLSQP':
                res = minimize(_obj, x0, method='SLSQP', bounds=bounds,
                               options={'maxiter': 150, 'ftol': 1e-10, 'maxfun': 1000},
                               callback=callback)
            else:
                res = minimize(_obj, x0, method='COBYLA',
                               options={'maxiter': 150, 'rhobeg': 0.05},
                               callback=callback)
            
            x_opt = res.x
            new_centers = np.column_stack([x_opt[0:10], x_opt[10:20]])
            new_sides = x_opt[20:30]
            new_centers, new_sides = _repair_and_clamp(new_centers, new_sides)
            
            if _is_valid(new_centers, new_sides):
                return new_centers, new_sides
        except Exception:
            pass
        
        return None

    # --- Main Search Loop ---
    
    best_centers = None
    best_sides = None
    best_score = -1.0
    
    def _update_best(c, s):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(c, s):
            score = np.sum(s)
            if score > best_score:
                best_score = score
                best_centers = c.copy()
                best_sides = s.copy()

    def init_grid_33():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        coords = [0.15, 0.5, 0.85]
        idx = 0
        for r in range(3):
            for col in range(3):
                c[idx, 0] = coords[col]
                c[idx, 1] = coords[r]
                s[idx] = 0.3
                idx += 1
        c[9, 0] = 0.5
        c[9, 1] = 0.5
        s[9] = 0.05
        return c, s

    def init_corner():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        c[0] = [0.2, 0.2]; s[0] = 0.4
        c[1] = [0.8, 0.2]; s[1] = 0.4
        c[2] = [0.2, 0.8]; s[2] = 0.4
        c[3] = [0.8, 0.8]; s[3] = 0.4
        c[4] = [0.5, 0.1]; s[4] = 0.2
        c[5] = [0.1, 0.5]; s[5] = 0.2
        c[6] = [0.9, 0.5]; s[6] = 0.2
        c[7] = [0.5, 0.9]; s[7] = 0.2
        c[8] = [0.35, 0.35]; s[8] = 0.1
        c[9] = [0.65, 0.65]; s[9] = 0.1
        return c, s

    def init_random(seed):
        rng = np.random.default_rng(seed)
        c = rng.uniform(0.1, 0.9, size=(10, 2))
        s = rng.uniform(0.1, 0.4, size=10)
        return c, s

    def init_stripe():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            row = i // 5
            col = i % 5
            c[i, 0] = 0.1 + col * 0.2
            c[i, 1] = 0.25 + row * 0.5
            s[i] = 0.18
        return c, s

    def init_shelf():
        """
        Analytical shelf layout: 4 squares of side 0.4 in corners, 
        4 squares of side 0.2 along edges, 2 small squares in center.
        Starting sum = 4*0.4 + 4*0.2 + 2*0.1 = 1.6 + 0.8 + 0.2 = 2.6
        """
        c = np.zeros((10, 2))
        s = np.zeros(10)
        # Four corner squares
        c[0] = [0.2, 0.2]; s[0] = 0.4
        c[1] = [0.8, 0.2]; s[1] = 0.4
        c[2] = [0.2, 0.8]; s[2] = 0.4
        c[3] = [0.8, 0.8]; s[3] = 0.4
        # Edge squares
        c[4] = [0.5, 0.1]; s[4] = 0.2
        c[5] = [0.1, 0.5]; s[5] = 0.2
        c[6] = [0.9, 0.5]; s[6] = 0.2
        c[7] = [0.5, 0.9]; s[7] = 0.2
        # Center squares
        c[8] = [0.35, 0.35]; s[8] = 0.1
        c[9] = [0.65, 0.65]; s[9] = 0.1
        return c, s

    def init_optimal_shelf():
        """
        Optimized shelf: try to fit larger squares.
        Layout: 3 squares of side ~0.33 in a row at bottom, 
        3 squares of side ~0.33 in a row at top,
        4 smaller squares filling gaps.
        Starting sum ≈ 6*0.33 + 4*0.25 = 1.98 + 1.0 = 2.98
        """
        c = np.zeros((10, 2))
        s = np.zeros(10)
        # Bottom row: 3 squares of side 1/3
        s[0] = 1.0/3.0
        c[0] = [1.0/6.0, 1.0/6.0]
        s[1] = 1.0/3.0
        c[1] = [1.0/2.0, 1.0/6.0]
        s[2] = 1.0/3.0
        c[2] = [5.0/6.0, 1.0/6.0]
        # Top row: 3 squares of side 1/3
        s[3] = 1.0/3.0
        c[3] = [1.0/6.0, 5.0/6.0]
        s[4] = 1.0/3.0
        c[4] = [1.0/2.0, 5.0/6.0]
        s[5] = 1.0/3.0
        c[5] = [5.0/6.0, 5.0/6.0]
        # Middle: 4 squares of side 0.25
        s[6] = 0.25
        c[6] = [0.125, 0.5]
        s[7] = 0.25
        c[7] = [0.375, 0.5]
        s[8] = 0.25
        c[8] = [0.625, 0.5]
        s[9] = 0.25
        c[9] = [0.875, 0.5]
        return c, s

    inits = [
        init_optimal_shelf,
        init_shelf,
        init_grid_33,
        init_corner,
        init_stripe,
        lambda: init_random(1),
        lambda: init_random(42),
        lambda: init_random(123),
        lambda: init_random(999),
        lambda: init_random(777),
        lambda: init_random(2024),
        lambda: init_random(555),
    ]
    
    for i, init_fn in enumerate(inits):
        if time.time() > deadline:
            break
        
        c, s = init_fn()
        c, s = _repair_and_clamp(c, s)
        _update_best(c, s)
        
        # Voronoi-based center optimization
        if time.time() < deadline - 60:
            result_v = _optimize_centers_voronoi(c[:, 0], c[:, 1], max_time=30.0)
            if result_v is not None:
                _update_best(result_v[0], result_v[1])
        
        # Full optimization
        if time.time() < deadline - 40:
            x0 = np.concatenate([c[:, 0], c[:, 1], s])
            result = _optimize_full(x0, max_time=30.0, method='SLSQP')
            if result is not None:
                _update_best(result[0], result[1])
        
        # Additional refinement with tighter tolerance
        if time.time() < deadline - 35 and best_centers is not None:
            x0 = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_sides])
            result_tight = _optimize_full(x0, max_time=15.0, method='SLSQP')
            if result_tight is not None:
                _update_best(result_tight[0], result_tight[1])
        
        # COBYLA refinement
        if time.time() < deadline - 30 and best_centers is not None:
            x0 = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_sides])
            result2 = _optimize_full(x0, max_time=20.0, method='COBYLA')
            if result2 is not None:
                _update_best(result2[0], result2[1])
        
        # Perturbation-based local search
        if best_centers is not None and time.time() < deadline - 50:
            rng = np.random.default_rng(i * 100 + 1)
            c3 = best_centers + rng.normal(0, 0.03, size=(10, 2))
            s3 = best_sides * (1 + rng.normal(0, 0.08, size=10))
            c3, s3 = _repair_and_clamp(c3, s3)
            _update_best(c3, s3)
            
            x0 = np.concatenate([c3[:, 0], c3[:, 1], s3])
            result3 = _optimize_full(x0, max_time=20.0, method='SLSQP')
            if result3 is not None:
                _update_best(result3[0], result3[1])

    # Final polish
    if best_centers is not None and time.time() < deadline - 20:
        x0 = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_sides])
        result_final = _optimize_full(x0, max_time=15.0, method='SLSQP')
        if result_final is not None:
            _update_best(result_final[0], result_final[1])
        
        # Voronoi polish
        if time.time() < deadline - 10:
            result_vp = _optimize_centers_voronoi(best_centers[:, 0], best_centers[:, 1], max_time=8.0)
            if result_vp is not None:
                _update_best(result_vp[0], result_vp[1])

    # Fallback
    if best_centers is None:
        best_centers = np.zeros((10, 2))
        best_sides = np.zeros(10)
        for i in range(10):
            r = i // 5
            col = i % 5
            best_centers[i, 0] = 0.1 + col * 0.2
            best_centers[i, 1] = 0.1 + r * 0.2
            best_sides[i] = 0.1
        best_centers, best_sides = _repair_and_clamp(best_centers, best_sides)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
