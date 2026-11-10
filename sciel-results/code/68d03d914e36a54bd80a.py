# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, positive sides, non-overlapping."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside [-tol, 1+tol]^2
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
            return False
    # Non-overlap: for i < j, must not have |cx_i-cx_j| < t AND |cy_i-cy_j| < t
    # where t = (s_i + s_j)/2 - tol
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            if dx < t and dy < t:
                return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    for i in range(len(sides)):
        half = sides[i] / 2.0
        centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
        centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
    return centers


def _resolve_overlap(centers, sides, tol=1e-7):
    """
    Attempt to resolve overlaps by pushing squares apart.
    Returns modified centers, sides.
    """
    n = len(sides)
    for iteration in range(200):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                if abs(dx) < t and abs(dy) < t:
                    # Overlapping: push apart along the axis of smaller gap
                    if abs(dx) < abs(dy):
                        # Push horizontally
                        push = (t - abs(dx)) / 2.0 + tol
                        if dx >= 0:
                            centers[j, 0] += push
                            centers[i, 0] -= push
                        else:
                            centers[j, 0] -= push
                            centers[i, 0] += push
                    else:
                        # Push vertically
                        push = (t - abs(dy)) / 2.0 + tol
                        if dy >= 0:
                            centers[j, 1] += push
                            centers[i, 1] -= push
                        else:
                            centers[j, 1] -= push
                            centers[i, 1] += push
                    moved = True
        if not moved:
            break
    return centers


def _try_grow_squares(centers, sides, tol=1e-7, max_iters=500):
    """
    Try to grow each square as much as possible while maintaining feasibility.
    Uses iterative bisection on each square's size.
    """
    n = len(sides)
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            # Try to grow square i
            lo = sides[i]
            hi = 1.0  # maximum possible side
            # Binary search for max size
            for _ in range(30):
                mid = (lo + hi) / 2.0
                test_sides = sides.copy()
                test_sides[i] = mid
                # Check if square i fits inside unit square
                half = mid / 2.0
                if centers[i, 0] - half < -tol or centers[i, 0] + half > 1.0 + tol:
                    hi = mid
                    continue
                if centers[i, 1] - half < -tol or centers[i, 1] + half > 1.0 + tol:
                    hi = mid
                    continue
                # Check non-overlap with all others
                feasible = True
                for j in range(n):
                    if j == i:
                        continue
                    t = (mid + sides[j]) / 2.0 - tol
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    if dx < t and dy < t:
                        feasible = False
                        break
                if feasible:
                    lo = mid
                else:
                    hi = mid
            # If we can grow
            if lo > sides[i] + 1e-9:
                sides[i] = lo
                improved = True
        if not improved:
            break
    return sides


def _grow_square_max(centers, sides, i, tol=1e-7):
    """Find the maximum side length for square i given all other positions/sizes (analytic)."""
    cx, cy = centers[i]
    # Bound from unit square: s <= 2*min(cx, 1-cx, cy, 1-cy)
    max_s = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
    if max_s < 0:
        max_s = 0.0
    # For each other square j, constraint: NOT(|cx-cxj| < (s+sj)/2 - tol AND |cy-cyj| < (s+sj)/2 - tol)
    # i.e., s <= 2*(dx+tol) - sj OR s <= 2*(dy+tol) - sj
    # The binding constraint for square i is: s <= min over j of max(2*(dx_j+tol)-sj, 2*(dy_j+tol)-sj)
    # But we also need s >= current sides[i]
    n = len(sides)
    for j in range(n):
        if j == i:
            continue
        dx = abs(cx - centers[j, 0])
        dy = abs(cy - centers[j, 1])
        # To avoid overlap with j: s <= 2*(dx + tol) - sides[j] OR s <= 2*(dy + tol) - sides[j]
        # The tighter of these two is the constraint
        c1 = 2.0 * (dx + tol) - sides[j]
        c2 = 2.0 * (dy + tol) - sides[j]
        c = max(c1, c2)
        if c < max_s:
            max_s = c
    if max_s < sides[i]:
        return sides[i]
    return max_s


def _try_push_to_corner(centers, sides, i, tol=1e-7):
    """
    Try to push square i toward each of the 4 corners and edges.
    Returns (new_centers, new_sides, improved) if a better packing is found.
    """
    n = len(sides)
    best_obj = _objective(centers, sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    half = sides[i] / 2.0
    
    # Candidate target positions: corners and edge midpoints
    targets = [
        (half, half),              # bottom-left corner
        (1.0 - half, half),        # bottom-right
        (half, 1.0 - half),        # top-left
        (1.0 - half, 1.0 - half),  # top-right
        (0.5, half),               # bottom edge mid
        (0.5, 1.0 - half),         # top edge mid
        (half, 0.5),               # left edge mid
        (1.0 - half, 0.5),         # right edge mid
    ]
    
    for tx, ty in targets:
        test_c = centers.copy()
        test_c[i, 0] = tx
        test_c[i, 1] = ty
        if not _is_valid(test_c, sides, tol):
            continue
        # Try growing all squares
        trial_s = sides.copy()
        for k in range(n):
            new_s = _grow_square_max(test_c, trial_s, k, tol)
            if new_s > trial_s[k] + 1e-9:
                trial_s[k] = new_s
        if _is_valid(test_c, trial_s, tol):
            obj = _objective(test_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = test_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s


def _grow_all_to_convergence(centers, sides, tol=1e-7, max_iters=200):
    """Iteratively grow all squares until convergence (simultaneous growth)."""
    n = len(sides)
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            new_s = _grow_square_max(centers, sides, i, tol)
            if new_s > sides[i] + 1e-10:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _smooth_penalty_and_grad(centers, sides, tol=1e-7):
    """
    Compute a smooth penalty for overlaps and boundary violations, plus gradients.
    Returns (penalty, grad_centers, grad_sides)
    """
    n = len(sides)
    penalty = 0.0
    grad_c = np.zeros_like(centers)
    grad_s = np.zeros_like(sides)
    
    # Boundary penalties
    for i in range(n):
        s = sides[i]
        half = s / 2.0
        cx, cy = centers[i]
        
        viol_l = max(0.0, -cx + half)
        if viol_l > 0:
            penalty += viol_l ** 2
            grad_c[i, 0] += 2.0 * viol_l
            grad_s[i] += viol_l
        
        viol_r = max(0.0, cx + half - 1.0)
        if viol_r > 0:
            penalty += viol_r ** 2
            grad_c[i, 0] -= 2.0 * viol_r
            grad_s[i] += viol_r
        
        viol_b = max(0.0, -cy + half)
        if viol_b > 0:
            penalty += viol_b ** 2
            grad_c[i, 1] += 2.0 * viol_b
            grad_s[i] += viol_b
        
        viol_t = max(0.0, cy + half - 1.0)
        if viol_t > 0:
            penalty += viol_t ** 2
            grad_c[i, 1] -= 2.0 * viol_t
            grad_s[i] += viol_t
    
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            adx = abs(dx)
            ady = abs(dy)
            
            viol_x = max(0.0, t - adx)
            viol_y = max(0.0, t - ady)
            
            if viol_x > 0 and viol_y > 0:
                pen = viol_x * viol_y
                penalty += pen ** 2
                
                if dx >= 0:
                    grad_c[i, 0] += 2.0 * pen * viol_y
                    grad_c[j, 0] -= 2.0 * pen * viol_y
                else:
                    grad_c[i, 0] -= 2.0 * pen * viol_y
                    grad_c[j, 0] += 2.0 * pen * viol_y
                
                if dy >= 0:
                    grad_c[i, 1] += 2.0 * pen * viol_x
                    grad_c[j, 1] -= 2.0 * pen * viol_x
                else:
                    grad_c[i, 1] -= 2.0 * pen * viol_x
                    grad_c[j, 1] += 2.0 * pen * viol_x
                
                dpen_di = 0.5 * (viol_y + viol_x)
                dpen_dj = 0.5 * (viol_y + viol_x)
                grad_s[i] += 2.0 * pen * dpen_di
                grad_s[j] += 2.0 * pen * dpen_dj
    
    return penalty, grad_c, grad_s


def _gradient_optimize(centers, sides, tol=1e-7, max_iters=300, lr_init=0.01, lr_decay=0.998, deadline=None):
    """
    Gradient-based optimization: minimize penalty while maximizing sum of sides.
    """
    n = len(sides)
    centers = centers.copy()
    sides = sides.copy()
    lr = lr_init
    best_obj = _objective(centers, sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        pen, grad_c, grad_s = _smooth_penalty_and_grad(centers, sides, tol)
        lam = 10.0
        
        grad_obj_c = -lam * grad_c
        grad_obj_s = np.ones(n) - lam * grad_s
        
        for i in range(n):
            centers[i, 0] += lr * grad_obj_c[i, 0]
            centers[i, 1] += lr * grad_obj_c[i, 1]
            half = sides[i] / 2.0
            centers[i, 0] = np.clip(centers[i, 0], half, 1.0 - half)
            centers[i, 1] = np.clip(centers[i, 1], half, 1.0 - half)
            
            sides[i] += lr * grad_obj_s[i]
            sides[i] = max(0.01, sides[i])
            max_s = min(2.0 * centers[i, 0], 2.0 * (1.0 - centers[i, 0]),
                        2.0 * centers[i, 1], 2.0 * (1.0 - centers[i, 1]))
            sides[i] = min(sides[i], max_s)
        
        lr *= lr_decay
        
        if _is_valid(centers, sides, tol):
            obj = _objective(centers, sides)
            if obj > best_obj:
                best_obj = obj
                best_c = centers.copy()
                best_s = sides.copy()
    
    return best_c, best_s


def _local_optimize(centers, sides, tol=1e-7, max_rounds=50, deadline=None):
    """
    Local optimization: alternate between simultaneous growth, repositioning, gradient steps, and perturbations.
    """
    n = len(sides)
    best_sides = sides.copy()
    best_centers = centers.copy()
    best_obj = _objective(best_centers, best_sides)
    
    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        
        # Step 1: Simultaneous growth to convergence
        new_sides = sides.copy()
        new_sides = _grow_all_to_convergence(centers, new_sides, tol)
        
        if _is_valid(centers, new_sides, tol):
            sides = new_sides
            obj = _objective(centers, sides)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_sides = sides.copy()
                best_centers = centers.copy()
        
        # Step 2: Try pushing each square to corners/edges
        for i in range(n):
            new_c, new_s = _try_push_to_corner(centers, sides, i, tol)
            if _is_valid(new_c, new_s, tol):
                obj = _objective(new_c, new_s)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_centers = new_c.copy()
                    best_sides = new_s.copy()
                    centers = new_c.copy()
                    sides = new_s.copy()
        
        # Step 3: Gradient-based optimization to escape local minima
        grad_c, grad_s = _gradient_optimize(centers, sides, tol, max_iters=100, deadline=deadline)
        if _is_valid(grad_c, grad_s, tol):
            obj = _objective(grad_c, grad_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_centers = grad_c.copy()
                best_sides = grad_s.copy()
                centers = grad_c.copy()
                sides = grad_s.copy()
        
        # Step 4: Small random perturbations with growth
        rng = np.random.default_rng(seed=rnd * 1000 + 42)
        for i in range(n):
            for _ in range(5):
                dx = rng.uniform(-0.04, 0.04)
                dy = rng.uniform(-0.04, 0.04)
                new_c = centers.copy()
                new_c[i, 0] = np.clip(centers[i, 0] + dx, sides[i]/2, 1.0 - sides[i]/2)
                new_c[i, 1] = np.clip(centers[i, 1] + dy, sides[i]/2, 1.0 - sides[i]/2)
                if _is_valid(new_c, sides, tol):
                    trial_s = sides.copy()
                    trial_s = _grow_all_to_convergence(new_c, trial_s, tol)
                    if _is_valid(new_c, trial_s, tol):
                        obj = _objective(new_c, trial_s)
                        if obj > best_obj + 1e-9:
                            best_obj = obj
                            best_centers = new_c.copy()
                            best_sides = trial_s.copy()
                            centers = new_c.copy()
                            sides = trial_s.copy()
        
        # Convergence check
        if rnd > 5:
            can_improve = False
            for i in range(n):
                max_s = _grow_square_max(centers, sides, i, tol)
                if max_s > sides[i] + 1e-9:
                    can_improve = True
                    break
            if not can_improve:
                break
    
    return best_centers, best_sides


def _init_grid_packing():
    """Baseline 3x3 grid packing."""
    n = 10
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _init_heterogeneous_1():
    """
    One large square (side 0.5) in bottom-left, rest in top-right region.
    Large square: [0, 0.5] x [0, 0.5], center (0.25, 0.25)
    Remaining region: [0,1]x[0.5,1] and [0.5,1]x[0,0.5]
    Place 9 squares in the top strip [0,1]x[0.5,1] (height 0.5)
    and right strip [0.5,1]x[0,0.5] (width 0.5).
    Actually let's put all 9 in the top-right L-shape.
    Simpler: put 9 squares in the region above and to the right.
    Let's use a 3x3 grid in [0,1]x[0.5,1] for 6, and [0.5,1]x[0,0.5] for 3.
    Top strip: 3 cols x 2 rows = 6 squares, side = 0.5/3 = 1/6
    Right strip below: 3 cols x 1 row = 3 squares, side = 0.5/3 = 1/6
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    # Square 0: large
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    # Top strip [0,1]x[0.5,1]: 3 cols, 2 rows
    side_t = 1.0 / 3.0
    idx = 1
    for row in range(2):
        for col in range(3):
            centers[idx, 0] = (col + 0.5) * side_t
            centers[idx, 1] = 0.5 + (row + 0.5) * side_t
            sides[idx] = side_t
            idx += 1
    # Right strip [0.5,1]x[0,0.5]: 3 cols, 1 row
    side_r = 1.0 / 3.0
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_r
        centers[idx, 1] = 0.25
        sides[idx] = side_r
        idx += 1
    return centers, sides


def _init_heterogeneous_2():
    """
    Two large squares (side 0.4) and 8 small ones.
    Square 0: [0, 0.4] x [0, 0.4], center (0.2, 0.2)
    Square 1: [0.6, 1.0] x [0, 0.4], center (0.8, 0.2)
    Remaining: top strip [0,1]x[0.4,1] (height 0.6) and middle [0.4,0.6]x[0,0.4]
    8 squares in top strip: 4 cols x 2 rows, side = 0.6/4 = 0.15... 
    Let's do 4x2 in top: side = 1/4 = 0.25? No, height is 0.6, so 2 rows of 0.3.
    4 cols of 0.25, 2 rows of 0.3 -> but squares must be square.
    Use side = 0.25: 4 cols fit in width 1.0, 2 rows need height 0.5 <= 0.6. OK.
    That's 8 squares of side 0.25 in top strip.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    # 8 squares of side 0.25 in top strip [0,1]x[0.4, 0.9]
    side_s = 0.25
    idx = 2
    for row in range(2):
        for col in range(4):
            centers[idx, 0] = (col + 0.5) * side_s
            centers[idx, 1] = 0.4 + (row + 0.5) * side_s
            sides[idx] = side_s
            idx += 1
    return centers, sides


def _init_heterogeneous_3():
    """
    One very large square (side 0.5) and 9 small squares filling the rest.
    Large: [0,0.5]x[0,0.5]
    Rest: L-shape. Fill with 9 squares of side ~0.167 in a 3x3 grid in [0,1]x[0.5,1]
    plus [0.5,1]x[0,0.5].
    Actually same as heterogeneous_1 but let's try different sizes.
    9 squares in the L: top strip [0,1]x[0.5,1] has area 0.5, right strip [0.5,1]x[0,0.5] area 0.25.
    Total L area = 0.75. 9 squares, avg area = 0.75/9 = 0.0833, side ~ 0.289.
    But they must fit geometrically. Let's just use 1/6 side (0.167) for all 9.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    # Top strip [0,1]x[0.5,1]: 6 cols x 1 row? side=1/6, 6 cols = 1.0, 1 row height 1/6 <= 0.5
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    # Right strip [0.5,1]x[0,0.5]: 3 cols x 1 row? side=1/6, 3 cols = 0.5, fits in width 0.5
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_4():
    """
    1 large (0.5) + 2 medium (0.25) + 7 small, placed in feasible positions.
    Large: [0,0.5]x[0,0.5]
    Medium 1: [0,0.25]x[0.5,0.75] -> center (0.125, 0.625)
    Medium 2: [0.75,1.0]x[0.5,0.75] -> center (0.875, 0.625)
    Small squares fill remaining gaps.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.125, 0.625]
    sides[1] = 0.25
    centers[2] = [0.875, 0.625]
    sides[2] = 0.25
    # Remaining regions to fill with 7 small squares:
    # [0.25,0.75]x[0.5,0.75] (width 0.5, height 0.25)
    # [0,1]x[0.75,1.0] (width 1, height 0.25)
    # [0.5,1]x[0,0.5] (width 0.5, height 0.5)
    # Place small squares of side ~0.125
    side_s = 0.125
    idx = 3
    positions = [
        [0.375, 0.625], [0.625, 0.625],  # in middle gap
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],  # top strip
        [0.75, 0.25],  # right-bottom
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_5():
    """
    Two large squares (0.45) in opposite corners, 8 smaller filling the rest.
    Square 0: [0,0.45]x[0,0.45], center (0.225, 0.225)
    Square 1: [0.55,1.0]x[0.55,1.0], center (0.775, 0.775)
    8 squares of side ~0.15 in remaining L-shape + cross.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.225, 0.225]
    sides[0] = 0.45
    centers[1] = [0.775, 0.775]
    sides[1] = 0.45
    # Fill with 8 squares of side 0.15
    side_s = 0.15
    idx = 2
    positions = [
        [0.225, 0.575], [0.225, 0.875],  # left of top-right large
        [0.525, 0.225], [0.825, 0.225],  # below top-right large / right of bottom-left
        [0.525, 0.525],  # center
        [0.05, 0.575], [0.05, 0.875],    # far left top
        [0.825, 0.525],                  # right middle
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_6():
    """
    One large (0.5) in corner, 3 medium (0.25) in other corners/edges, 6 small.
    Large: [0,0.5]x[0,0.5]
    Medium: [0.5,0.75]x[0,0.25], [0.75,1]x[0.5,0.75], [0,0.25]x[0.75,1]
    Small fill remaining gaps.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.625, 0.125]
    sides[1] = 0.25
    centers[2] = [0.875, 0.625]
    sides[2] = 0.25
    centers[3] = [0.125, 0.875]
    sides[3] = 0.25
    # 6 small squares of side ~0.125
    side_s = 0.125
    idx = 4
    positions = [
        [0.625, 0.375], [0.875, 0.375],
        [0.375, 0.625], [0.375, 0.875],
        [0.625, 0.875], [0.875, 0.875],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_7():
    """
    Three large squares (0.4) in three corners, 7 smaller filling the rest.
    Square 0: [0,0.4]x[0,0.4]
    Square 1: [0.6,1.0]x[0,0.4]
    Square 2: [0,0.4]x[0.6,1.0]
    7 squares fill the remaining L-shape in top-right.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    centers[2] = [0.2, 0.8]
    sides[2] = 0.4
    # Remaining region: [0.4,1]x[0.4,1] minus overlaps
    # Top-right area [0.4,1]x[0.4,1] is 0.6x0.6
    # Place 7 squares of side ~0.2 in this region
    side_s = 0.2
    idx = 3
    positions = [
        [0.5, 0.5], [0.8, 0.5],
        [0.5, 0.8], [0.8, 0.8],
        [0.65, 0.65], [0.65, 0.35],
        [0.35, 0.65],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 580s safety margin
    
    # Generate multiple initial packings
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
        _init_heterogeneous_7(),
    ]
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for centers, sides in inits:
        if time.time() > deadline:
            break
        
        # First, resolve any overlaps
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        centers = _resolve_overlap(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            # Try harder to fix
            for _ in range(10):
                centers = _resolve_overlap(centers, sides)
                centers = _clamp_to_unit(centers, sides)
                if _is_valid(centers, sides):
                    break
        
        if not _is_valid(centers, sides):
            continue
        
        # Local optimization
        opt_centers, opt_sides = _local_optimize(centers, sides, deadline=deadline)
        
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
        
        # Also try the original (pre-optimization) in case optimizer degraded
        orig_obj = _objective(centers, sides)
        if orig_obj > best_obj:
            best_obj = orig_obj
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # If no valid solution found, fall back to grid
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        # Final validation
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        
        # Final polish: high-precision simultaneous growth
        final_sides = best_sides.copy()
        final_sides = _grow_all_to_convergence(best_centers, final_sides, tol=1e-7, max_iters=500)
        if _is_valid(best_centers, final_sides):
            if _objective(best_centers, final_sides) > _objective(best_centers, best_sides):
                best_sides = final_sides
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
