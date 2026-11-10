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
    deadline = start_time + 560.0  # Safety margin

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
        for _ in range(300):
            overlap_found = False
            for i in range(10):
                for j in range(i+1, 10):
                    t = (sides[i] + sides[j]) / 2.0
                    dx = abs(centers[i,0] - centers[j,0])
                    dy = abs(centers[i,1] - centers[j,1])
                    if dx < t and dy < t:
                        overlap_found = True
                        # Shrink the smaller square by 12%
                        if sides[i] < sides[j]:
                            sides[i] *= 0.88
                        else:
                            sides[j] *= 0.88
                        # Re-clamp centers for the shrunk square
                        idx = i if sides[i] < sides[j] else j
                        h = sides[idx] / 2.0
                        centers[idx, 0] = np.clip(centers[idx, 0], h, 1.0 - h)
                        centers[idx, 1] = np.clip(centers[idx, 1], h, 1.0 - h)
            if not overlap_found:
                break
        
        return centers, sides

    def _objective_smooth(x, penalty_weight=800.0, eps=1e-7):
        """
        Objective: -sum(sides) + penalty(overlaps, boundaries)
        Uses smooth soft-max for overlap penalties.
        x layout: [cx0..cx9, cy0..cy9, s0..s9]
        """
        cx = x[0:10]
        cy = x[10:20]
        s = x[20:30]
        
        obj = -np.sum(s)
        
        # Boundary penalties (smooth quadratic)
        for i in range(10):
            v = cx[i] - s[i]/2.0
            if v < 0:
                obj += penalty_weight * v * v
            v = (1.0 - s[i]/2.0) - cx[i]
            if v < 0:
                obj += penalty_weight * v * v
            v = cy[i] - s[i]/2.0
            if v < 0:
                obj += penalty_weight * v * v
            v = (1.0 - s[i]/2.0) - cy[i]
            if v < 0:
                obj += penalty_weight * v * v

        # Overlap penalties using smooth soft-max
        for i in range(10):
            for j in range(i+1, 10):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                
                ov_x = t - dx
                ov_y = t - dy
                
                # Smooth max(0, z)
                smax_u = 0.5 * (ov_x + np.sqrt(ov_x**2 + eps))
                smax_v = 0.5 * (ov_y + np.sqrt(ov_y**2 + eps))
                
                obj += penalty_weight * smax_u * smax_v
        
        return obj

    def _objective_grad(x, penalty_weight=800.0, eps=1e-7):
        """
        Gradient of _objective_smooth for L-BFGS-B.
        """
        cx = x[0:10]
        cy = x[10:20]
        s = x[20:30]
        
        grad = np.zeros_like(x)
        
        # Gradient of -sum(s)
        grad[20:30] = -1.0
        
        # Boundary gradient contributions
        for i in range(10):
            # Left boundary: penalty = pw * (cx[i] - s[i]/2)^2 if cx[i] - s[i]/2 < 0
            v = cx[i] - s[i]/2.0
            if v < 0:
                grad[i] += 2 * penalty_weight * v
                grad[20+i] += -penalty_weight * v
            # Right boundary
            v = (1.0 - s[i]/2.0) - cx[i]
            if v < 0:
                grad[i] += -2 * penalty_weight * v
                grad[20+i] += -penalty_weight * v
            # Bottom boundary
            v = cy[i] - s[i]/2.0
            if v < 0:
                grad[10+i] += 2 * penalty_weight * v
                grad[20+i] += -penalty_weight * v
            # Top boundary
            v = (1.0 - s[i]/2.0) - cy[i]
            if v < 0:
                grad[10+i] += -2 * penalty_weight * v
                grad[20+i] += -penalty_weight * v

        # Overlap gradient contributions
        for i in range(10):
            for j in range(i+1, 10):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                adx = abs(dx)
                ady = abs(dy)
                t = (s[i] + s[j]) / 2.0
                
                ov_x = t - adx
                ov_y = t - ady
                
                smax_u = 0.5 * (ov_x + np.sqrt(ov_x**2 + eps))
                smax_v = 0.5 * (ov_y + np.sqrt(ov_y**2 + eps))
                
                # d(smax_u)/d(ov_x) = 0.5 * (1 + ov_x / sqrt(ov_x^2 + eps))
                dsmax_u = 0.5 * (1.0 + ov_x / np.sqrt(ov_x**2 + eps))
                dsmax_v = 0.5 * (1.0 + ov_y / np.sqrt(ov_y**2 + eps))
                
                # d(ov_x)/d(cx[i]) = -sign(dx) when dx != 0, else 0
                # d(ov_x)/d(cx[j]) = +sign(dx)
                sign_x = np.sign(dx) if dx != 0 else 0.0
                sign_y = np.sign(dy) if dy != 0 else 0.0
                
                # Penalty = pw * smax_u * smax_v
                # d(penalty)/d(cx[i]) = pw * dsmax_u * (-sign_x) * smax_v + ...
                # Actually: d(ov_x)/d(cx[i]) = -sign_x, d(ov_x)/d(cx[j]) = sign_x
                # d(ov_y)/d(cy[i]) = -sign_y, d(ov_y)/d(cy[j]) = sign_y
                # d(ov_x)/d(s[i]) = 0.5, d(ov_x)/d(s[j]) = 0.5
                # d(ov_y)/d(s[i]) = 0.5, d(ov_y)/d(s[j]) = 0.5
                
                # d(penalty)/d(cx[i]) = pw * dsmax_u * (-sign_x) * smax_v
                grad[i] += penalty_weight * (-sign_x) * dsmax_u * smax_v
                grad[j] += penalty_weight * (sign_x) * dsmax_u * smax_v
                
                # d(penalty)/d(cy[i]) = pw * dsmax_v * (-sign_y) * smax_u
                grad[10+i] += penalty_weight * (-sign_y) * dsmax_v * smax_u
                grad[10+j] += penalty_weight * (sign_y) * dsmax_v * smax_u
                
                # d(penalty)/d(s[i]) = pw * (dsmax_u * 0.5 * smax_v + dsmax_v * 0.5 * smax_u)
                grad[20+i] += penalty_weight * (0.5 * dsmax_u * smax_v + 0.5 * dsmax_v * smax_u)
                grad[20+j] += penalty_weight * (0.5 * dsmax_u * smax_v + 0.5 * dsmax_v * smax_u)
        
        return grad

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
        
        try:
            if method == 'SLSQP':
                res = minimize(_objective_smooth, x0, method='SLSQP', bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-13, 'maxfun': 2000},
                               callback=callback)
            elif method == 'L-BFGS-B':
                res = minimize(_objective_smooth, x0, method='L-BFGS-B', bounds=bounds,
                               jac=_objective_grad,
                               options={'maxiter': 500, 'gtol': 1e-12, 'maxfun': 3000},
                               callback=callback)
            else:
                res = minimize(_objective_smooth, x0, method='COBYLA',
                               options={'maxiter': 400, 'rhobeg': 0.02},
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

    def _optimize_centers_voronoi(cx0, cy0, max_time=30.0):
        """
        Optimize center positions to maximize sum of Voronoi-limited side lengths.
        """
        def _objective_voronoi_smooth(x, beta=40.0):
            """
            Smooth version: use soft-min instead of hard min for the Voronoi limits.
            """
            cx = x[0:10]
            cy = x[10:20]
            
            s = np.zeros(10)
            for i in range(10):
                limit = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i]) * 2.0
                s[i] = max(0.0, limit)
            
            for _ in range(6):
                for i in range(10):
                    limit = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i]) * 2.0
                    for j in range(10):
                        if i == j:
                            continue
                        dx = abs(cx[i] - cx[j])
                        dy = abs(cy[i] - cy[j])
                        a = 2.0 * dx - s[j]
                        b = 2.0 * dy - s[j]
                        m = min(a, b)
                        sm = m - np.logaddexp(-beta * (a - m), -beta * (b - m)) / beta
                        if sm < limit:
                            limit = sm
                    s[i] = max(0.0, limit)
            
            return -np.sum(s)
        
        x0 = np.concatenate([cx0, cy0])
        bounds = [(0.0, 1.0)] * 20
        
        t0 = time.time()
        
        def callback(xk):
            if time.time() - t0 > max_time:
                raise RuntimeError("Time limit")
        
        try:
            res = minimize(_objective_voronoi_smooth, x0, method='SLSQP', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-13, 'maxfun': 1500},
                           callback=callback)
            
            x_opt = res.x
            new_cx = x_opt[0:10]
            new_cy = x_opt[10:20]
            
            sides = np.zeros(10)
            for i in range(10):
                limit = min(new_cx[i], 1.0 - new_cx[i], new_cy[i], 1.0 - new_cy[i]) * 2.0
                for j in range(10):
                    if i == j:
                        continue
                    dx = abs(new_cx[i] - new_cx[j])
                    dy = abs(new_cy[i] - new_cy[j])
                    lim_j = min(2.0 * dx - sides[j], 2.0 * dy - sides[j])
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

    def init_diagonal():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            t = i / 9.0
            c[i, 0] = 0.1 + 0.8 * t
            c[i, 1] = 0.1 + 0.8 * t
            s[i] = 0.2
        return c, s

    inits = [
        init_grid_33,
        init_corner,
        init_stripe,
        init_diagonal,
        lambda: init_random(1),
        lambda: init_random(42),
        lambda: init_random(123),
        lambda: init_random(999),
        lambda: init_random(777),
        lambda: init_random(2024),
        lambda: init_random(555),
        lambda: init_random(31415),
        lambda: init_random(8888),
        lambda: init_random(4242),
    ]
    
    for i, init_fn in enumerate(inits):
        if time.time() > deadline:
            break
        
        c, s = init_fn()
        c, s = _repair_and_clamp(c, s)
        _update_best(c, s)
        
        # Voronoi-based center optimization
        if time.time() < deadline - 60:
            result_v = _optimize_centers_voronoi(c[:, 0], c[:, 1], max_time=20.0)
            if result_v is not None:
                _update_best(result_v[0], result_v[1])
        
        # Full optimization with SLSQP
        if time.time() < deadline - 50:
            x0 = np.concatenate([c[:, 0], c[:, 1], s])
            result = _optimize_full(x0, max_time=20.0, method='SLSQP')
            if result is not None:
                _update_best(result[0], result[1])
        
        # L-BFGS-B optimization with gradient
        if time.time() < deadline - 40 and best_centers is not None:
            x0 = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_sides])
            result_lbfgs = _optimize_full(x0, max_time=20.0, method='L-BFGS-B')
            if result_lbfgs is not None:
                _update_best(result_lbfgs[0], result_lbfgs[1])
        
        # Perturbation-based local search
        if best_centers is not None and time.time() < deadline - 40:
            rng = np.random.default_rng(i * 100 + 1)
            c3 = best_centers + rng.normal(0, 0.03, size=(10, 2))
            s3 = best_sides * (1 + rng.normal(0, 0.08, size=10))
            c3, s3 = _repair_and_clamp(c3, s3)
            _update_best(c3, s3)
            
            x0 = np.concatenate([c3[:, 0], c3[:, 1], s3])
            result3 = _optimize_full(x0, max_time=15.0, method='SLSQP')
            if result3 is not None:
                _update_best(result3[0], result3[1])

    # Final polish with L-BFGS-B
    if best_centers is not None and time.time() < deadline - 30:
        x0 = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_sides])
        result_final = _optimize_full(x0, max_time=25.0, method='L-BFGS-B')
        if result_final is not None:
            _update_best(result_final[0], result_final[1])
        
        # Additional SLSQP polish
        if time.time() < deadline - 15:
            x0 = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_sides])
            result_final2 = _optimize_full(x0, max_time=15.0, method='SLSQP')
            if result_final2 is not None:
                _update_best(result_final2[0], result_final2[1])

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
