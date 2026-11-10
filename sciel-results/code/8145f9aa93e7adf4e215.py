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
    deadline = start_time + 580.0  # Safety margin

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

    def _objective(x, penalty_weight=1000.0):
        """
        Objective: -sum(sides) + penalty(overlaps, boundaries)
        x layout: [cx0..cx9, cy0..cy9, s0..s9]
        """
        cx = x[0:10]
        cy = x[10:20]
        s = x[20:30]
        
        obj = -np.sum(s)
        
        # Boundary penalties (smooth quadratic)
        for i in range(10):
            # Left
            v = cx[i] - s[i]/2.0
            if v < 0:
                obj += penalty_weight * v * v
            # Right
            v = (1.0 - s[i]/2.0) - cx[i]
            if v < 0:
                obj += penalty_weight * v * v
            # Bottom
            v = cy[i] - s[i]/2.0
            if v < 0:
                obj += penalty_weight * v * v
            # Top
            v = (1.0 - s[i]/2.0) - cy[i]
            if v < 0:
                obj += penalty_weight * v * v

        # Overlap penalties
        for i in range(10):
            for j in range(i+1, 10):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t_x = (s[i] + s[j]) / 2.0
                t_y = (s[i] + s[j]) / 2.0
                
                ov_x = t_x - dx
                ov_y = t_y - dy
                
                pos_ov_x = max(0.0, ov_x)
                pos_ov_y = max(0.0, ov_y)
                
                if pos_ov_x > 0 and pos_ov_y > 0:
                    obj += penalty_weight * pos_ov_x * pos_ov_y
        
        return obj

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

    def _optimize_from(centers, sides, max_time=30.0, method='SLSQP'):
        """
        Run optimization from a given start point.
        """
        x0 = np.concatenate([centers[:, 0], centers[:, 1], sides])
        
        # Bounds
        bounds = []
        for _ in range(10):
            bounds.append((0.0, 1.0))
        for _ in range(10):
            bounds.append((0.0, 1.0))
        for _ in range(10):
            bounds.append((0.01, 1.0))
            
        t0 = time.time()
        call_count = [0]
        
        def callback(xk):
            call_count[0] += 1
            if time.time() - t0 > max_time:
                raise RuntimeError("Time limit")
                
        try:
            if method == 'SLSQP':
                res = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                               options={'maxiter': 150, 'ftol': 1e-9, 'maxfun': 1000},
                               callback=callback)
            else:
                # COBYLA doesn't support bounds directly, so we rely on penalty
                res = minimize(_objective, x0, method='COBYLA',
                               options={'maxiter': 150, 'rhobeg': 0.1},
                               callback=callback)
            
            if res.success or (time.time() - t0 < max_time):
                x_opt = res.x
                new_centers = np.column_stack([x_opt[0:10], x_opt[10:20]])
                new_sides = x_opt[20:30]
                
                # Repair
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
    
    # --- Structural Change: Hybrid Approach ---
    # 1. Multi-start with diverse heuristics
    # 2. Local refinement with SLSQP and COBYLA
    # 3. Perturbation-based local search (simulated annealing-like)

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
        # 10th square: place it in a gap or shrink
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
        # 2 rows of 5
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            row = i // 5
            col = i % 5
            c[i, 0] = 0.1 + col * 0.2
            c[i, 1] = 0.25 + row * 0.5
            s[i] = 0.18
        return c, s

    inits = [
        init_grid_33,
        init_corner,
        init_stripe,
        lambda: init_random(1),
        lambda: init_random(42),
        lambda: init_random(123),
        lambda: init_random(999),
        lambda: init_random(777),
    ]
    
    for i, init_fn in enumerate(inits):
        if time.time() > deadline:
            break
            
        c, s = init_fn()
        c, s = _repair_and_clamp(c, s)
        _update_best(c, s)
        
        # Optimize with SLSQP
        result = _optimize_from(c, s, max_time=45.0, method='SLSQP')
        if result is not None:
            _update_best(result[0], result[1])
        
        if time.time() > deadline:
            break
        
        # Optimize with COBYLA from the SLSQP result if available, else from repaired init
        if result is not None:
            c2, s2 = result
        else:
            c2, s2 = c, s
            
        result2 = _optimize_from(c2, s2, max_time=45.0, method='COBYLA')
        if result2 is not None:
            _update_best(result2[0], result2[1])

    # If no valid result found, use a trivial valid one
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
        best_score = np.sum(best_sides)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
