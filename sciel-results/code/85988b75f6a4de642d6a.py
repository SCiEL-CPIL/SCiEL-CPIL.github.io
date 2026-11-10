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


def _grow_square_max(centers, sides, i, tol=1e-7):
    """Find the maximum side length for square i given all other positions/sizes."""
    lo = sides[i]
    hi = 1.0
    cx, cy = centers[i]
    for _ in range(50):
        mid = (lo + hi) / 2.0
        half = mid / 2.0
        if cx - half < -tol or cx + half > 1.0 + tol:
            hi = mid
            continue
        if cy - half < -tol or cy + half > 1.0 + tol:
            hi = mid
            continue
        feasible = True
        for j in range(len(sides)):
            if j == i:
                continue
            t = (mid + sides[j]) / 2.0 - tol
            dx = abs(cx - centers[j, 0])
            dy = abs(cy - centers[j, 1])
            if dx < t and dy < t:
                feasible = False
                break
        if feasible:
            lo = mid
        else:
            hi = mid
    return lo


def _grow_all_iteratively(centers, sides, tol=1e-7, max_iters=300):
    """
    Iteratively grow each square to its maximum given others, repeating until convergence.
    """
    n = len(sides)
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            new_s = _grow_square_max(centers, sides, i, tol)
            if new_s > sides[i] + 1e-12:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _force_based_separation(centers, sides, tol=1e-7, max_iters=200, lr=0.05):
    """
    Use smooth repulsion forces to separate overlapping squares.
    This provides a differentiable landscape for optimization.
    """
    n = len(sides)
    for it in range(max_iters):
        forces = np.zeros((n, 2))
        max_overlap = 0.0
        
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                ax = abs(dx)
                ay = abs(dy)
                
                # Smooth overlap measure
                if ax < t and ay < t:
                    # Overlap exists - compute repulsion
                    overlap_x = max(0, t - ax)
                    overlap_y = max(0, t - ay)
                    max_overlap = max(max_overlap, overlap_x, overlap_y)
                    
                    # Force direction: push apart along axis of least overlap
                    if overlap_x < overlap_y:
                        # Push apart in x
                        sign = 1 if dx >= 0 else -1
                        forces[i, 0] += sign * overlap_x * 10
                        forces[j, 0] -= sign * overlap_x * 10
                    else:
                        # Push apart in y
                        sign = 1 if dy >= 0 else -1
                        forces[i, 1] += sign * overlap_y * 10
                        forces[j, 1] -= sign * overlap_y * 10
        
        if max_overlap < 1e-10:
            break
        
        # Apply forces with learning rate
        for i in range(n):
            centers[i, 0] += lr * forces[i, 0]
            centers[i, 1] += lr * forces[i, 1]
        
        # Clamp to unit square
        centers = _clamp_to_unit(centers, sides)
    
    return centers


def _push_to_corner(centers, sides, i, tol=1e-7):
    """Push square i to corners, then regrow."""
    n = len(sides)
    best_obj = _objective(centers, sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    half = sides[i] / 2.0
    targets = [
        (half, half),
        (1.0 - half, half),
        (half, 1.0 - half),
        (1.0 - half, 1.0 - half),
        (0.5, half),
        (0.5, 1.0 - half),
        (half, 0.5),
        (1.0 - half, 0.5),
    ]
    
    for tx, ty in targets:
        test_c = centers.copy()
        test_c[i, 0] = tx
        test_c[i, 1] = ty
        if not _is_valid(test_c, sides, tol):
            continue
        trial_s = sides.copy()
        for k in range(n):
            new_s = _grow_square_max(test_c, trial_s, k, tol)
            if new_s > trial_s[k] + 1e-12:
                trial_s[k] = new_s
        if _is_valid(test_c, trial_s, tol):
            obj = _objective(test_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = test_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s


def _shake_and_regrow(centers, sides, tol=1e-7, rng=None):
    """
    Perturb all squares simultaneously, then resolve and regrow.
    Helps escape local minima.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(sides)
    new_centers = centers.copy()
    for i in range(n):
        dx = rng.uniform(-0.05, 0.05)
        dy = rng.uniform(-0.05, 0.05)
        new_centers[i, 0] = np.clip(centers[i, 0] + dx, sides[i]/2, 1.0 - sides[i]/2)
        new_centers[i, 1] = np.clip(centers[i, 1] + dy, sides[i]/2, 1.0 - sides[i]/2)
    
    # Force-based separation
    new_centers = _force_based_separation(new_centers, sides, tol)
    
    # Check validity
    if not _is_valid(new_centers, sides, tol):
        return None
    
    # Regrow
    new_sides = _grow_all_iteratively(new_centers.copy(), sides.copy(), tol)
    
    if _is_valid(new_centers, new_sides, tol):
        return new_centers, new_sides
    
    return None


def _local_optimize(centers, sides, tol=1e-7, max_rounds=80, seed=42):
    """
    Enhanced local optimization with force-based methods.
    """
    n = len(sides)
    best_sides = sides.copy()
    best_centers = centers.copy()
    best_obj = _objective(best_centers, best_sides)
    
    rng = np.random.default_rng(seed=seed)
    
    for rnd in range(max_rounds):
        # Step 1: Grow all squares iteratively
        new_sides = _grow_all_iteratively(centers.copy(), sides.copy(), tol)
        if _is_valid(centers, new_sides, tol):
            obj = _objective(centers, new_sides)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_sides = new_sides.copy()
                best_centers = centers.copy()
                sides = new_sides.copy()
        
        # Step 2: Force-based separation + regrow
        if rnd % 3 == 0:
            sep_centers = _force_based_separation(centers.copy(), sides, tol)
            if _is_valid(sep_centers, sides, tol):
                sep_sides = _grow_all_iteratively(sep_centers.copy(), sides.copy(), tol)
                if _is_valid(sep_centers, sep_sides, tol):
                    obj = _objective(sep_centers, sep_sides)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_sides = sep_sides.copy()
                        best_centers = sep_centers.copy()
                        centers = sep_centers.copy()
                        sides = sep_sides.copy()
        
        # Step 3: Try pushing each square to corners/edges
        for i in range(n):
            new_c, new_s = _push_to_corner(centers, sides, i, tol)
            if _objective(new_c, new_s) > best_obj + 1e-9:
                best_obj = _objective(new_c, new_s)
                best_centers = new_c
                best_sides = new_s
                centers = new_c.copy()
                sides = new_s.copy()
                sides = _grow_all_iteratively(centers.copy(), sides.copy(), tol)
                if _is_valid(centers, sides, tol):
                    obj = _objective(centers, sides)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_sides = sides.copy()
                        best_centers = centers.copy()
        
        # Step 4: Shake and regrow (escape local minima)
        if rnd % 5 == 2:
            result = _shake_and_regrow(centers, sides, tol, rng)
            if result is not None:
                new_c, new_s = result
                obj = _objective(new_c, new_s)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_centers = new_c.copy()
                    best_sides = new_s.copy()
                    centers = new_c.copy()
                    sides = new_s.copy()
        
        # Step 5: Small targeted perturbations
        for i in range(n):
            for _ in range(3):
                dx = rng.uniform(-0.03, 0.03)
                dy = rng.uniform(-0.03, 0.03)
                new_c = centers.copy()
                new_c[i, 0] = np.clip(centers[i, 0] + dx, sides[i]/2, 1.0 - sides[i]/2)
                new_c[i, 1] = np.clip(centers[i, 1] + dy, sides[i]/2, 1.0 - sides[i]/2)
                if _is_valid(new_c, sides, tol):
                    trial_s = _grow_all_iteratively(new_c.copy(), sides.copy(), tol)
                    if _is_valid(new_c, trial_s, tol):
                        obj = _objective(new_c, trial_s)
                        if obj > best_obj + 1e-9:
                            best_obj = obj
                            best_centers = new_c.copy()
                            best_sides = trial_s.copy()
                            centers = new_c.copy()
                            sides = trial_s.copy()
        
        # Convergence check
        if rnd > 10 and abs(_objective(centers, sides) - best_obj) < 1e-14:
            can_improve = False
            for i in range(n):
                max_s = _grow_square_max(centers, sides, i, tol)
                if max_s > sides[i] + 1e-12:
                    can_improve = True
                    break
            if not can_improve:
                break
    
    return best_centers, best_sides


def _init_grid_packing():
    """Baseline 3x3 grid packing (9 squares of 1/3 + 1 extra)."""
    n = 10
    k = 3
    side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[:9] = np.stack([cx, cy], axis=1)
    sides[:9] = side
    centers[9] = [0.5, 0.5]
    sides[9] = 1.0/6.0
    return centers, sides


def _init_heterogeneous_1():
    """One large (0.5) bottom-left, 9 smaller filling rest."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_2():
    """Two large (0.4) and 8 small (0.25)."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
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
    """One very large (0.5) + 9 small in L-shape."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_4():
    """1 large (0.5) + 2 medium (0.25) + 7 small (0.125)."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.125, 0.625]
    sides[1] = 0.25
    centers[2] = [0.875, 0.625]
    sides[2] = 0.25
    side_s = 0.125
    idx = 3
    positions = [
        [0.375, 0.625], [0.625, 0.625],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
        [0.75, 0.25],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_5():
    """Two large (0.45) in opposite corners, 8 smaller filling rest."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.225, 0.225]
    sides[0] = 0.45
    centers[1] = [0.775, 0.775]
    sides[1] = 0.45
    side_s = 0.15
    idx = 2
    positions = [
        [0.225, 0.575], [0.225, 0.875],
        [0.525, 0.225], [0.825, 0.225],
        [0.525, 0.525],
        [0.05, 0.575], [0.05, 0.875],
        [0.825, 0.525],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_6():
    """Three medium (0.33) in corners + 7 small."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.165, 0.165]
    sides[0] = 0.33
    centers[1] = [0.835, 0.165]
    sides[1] = 0.33
    centers[2] = [0.165, 0.835]
    sides[2] = 0.33
    side_s = 0.15
    idx = 3
    positions = [
        [0.5, 0.165], [0.835, 0.5], [0.5, 0.5],
        [0.5, 0.835], [0.835, 0.835], [0.33, 0.5],
        [0.665, 0.33],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_random(seed=123):
    """Random initial configuration."""
    rng = np.random.default_rng(seed=seed)
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    for i in range(10):
        s = rng.uniform(0.1, 0.35)
        sides[i] = s
        centers[i, 0] = rng.uniform(s/2, 1.0 - s/2)
        centers[i, 1] = rng.uniform(s/2, 1.0 - s/2)
    return centers, sides


def _resolve_overlaps(centers, sides, tol=1e-7):
    """Resolve overlaps using force-based separation."""
    return _force_based_separation(centers, sides, tol, max_iters=500)


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
    ]
    
    # Add random inits
    for s in [123, 456, 789, 1011, 2020, 3030]:
        inits.append(_init_random(seed=s))
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for idx, (centers, sides) in enumerate(inits):
        if time.time() > deadline:
            break
        
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        
        # Resolve overlaps
        centers = _resolve_overlaps(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            # Try harder to resolve
            centers = _force_based_separation(centers, sides, max_iters=1000)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides):
                continue
        
        # Local optimization with unique seed per init
        opt_centers, opt_sides = _local_optimize(centers, sides, seed=42 + idx * 100)
        
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
        
        # If we have a good result, try to improve it further
        if best_obj > 1.15 and time.time() < deadline - 60:
            # Additional shake-and-reground on best
            rng = np.random.default_rng(seed=999 + idx)
            for _ in range(5):
                result = _shake_and_regrow(best_centers, best_sides, rng=rng)
                if result is not None:
                    new_c, new_s = result
                    obj = _objective(new_c, new_s)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_centers = new_c.copy()
                        best_sides = new_s.copy()
    
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        
        # Final high-precision polish
        for _ in range(3):
            final_sides = best_sides.copy()
            for i in range(10):
                ns = _grow_square_max(best_centers, final_sides, i, tol=1e-9)
                if ns > final_sides[i] + 1e-12:
                    final_sides[i] = ns
            if _is_valid(best_centers, final_sides, tol=1e-7):
                if _objective(best_centers, final_sides) > _objective(best_centers, best_sides):
                    best_sides = final_sides.copy()
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
