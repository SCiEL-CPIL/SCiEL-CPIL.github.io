# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 620.0  # safety margin before 630s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_feasible(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square and no overlaps."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside [-tol, 1+tol]
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    # No overlaps
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _score(centers, sides):
    return float(np.sum(sides))


def _pack_guillotine(splits, region):
    """
    Recursively pack squares using guillotine cuts.
    splits: list of (ratio, orientation) for internal nodes (pre-order).
    region: (x, y, w, h)
    Returns (centers, sides) or None if infeasible.
    """
    x, y, w, h = region
    if not splits:
        # Leaf: one square filling this region as much as possible
        s = min(w, h)
        if s <= 0:
            return None
        cx = x + w / 2
        cy = y + h / 2
        return np.array([[cx, cy]]), np.array([s])

    ratio, orient = splits[0]
    rest = splits[1:]

    if orient == 0:  # vertical cut: left and right
        w1 = w * ratio
        w2 = w - w1
        if w1 <= 0 or w2 <= 0:
            return None
        res1 = _pack_guillotine(rest[:len(rest) // 2], (x, y, w1, h))
        res2 = _pack_guillotine(rest[len(rest) // 2:], (x + w1, y, w2, h))
        if res1 is None or res2 is None:
            return None
        c1, s1 = res1
        c2, s2 = res2
        return np.vstack([c1, c2]), np.concatenate([s1, s2])
    else:  # horizontal cut: bottom and top
        h1 = h * ratio
        h2 = h - h1
        if h1 <= 0 or h2 <= 0:
            return None
        res1 = _pack_guillotine(rest[:len(rest) // 2], (x, y, w, h1))
        res2 = _pack_guillotine(rest[len(rest) // 2:], (x, y + h1, w, h2))
        if res1 is None or res2 is None:
            return None
        c1, s1 = res1
        c2, s2 = res2
        return np.vstack([c1, c2]), np.concatenate([s1, s2])


def _try_packing(splits, n=10):
    """Try to pack n squares with given guillotine splits."""
    if len(splits) != n - 1:
        return None
    result = _pack_guillotine(splits, (0, 0, 1, 1))
    if result is None:
        return None
    centers, sides = result
    if len(sides) != n:
        return None
    if _is_feasible(centers, sides):
        return centers, sides
    return None


def _local_search(centers, sides, max_iters=5000):
    """Local search: perturb positions and sizes, accept if feasible and better."""
    n = len(sides)
    best_score = _score(centers, sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    
    lr = 0.02
    for it in range(max_iters):
        if _time_left() < 10:
            break
        if it % 200 == 0 and it > 0:
            lr *= 0.95
        
        # Random perturbation
        dc = np.random.randn(n, 2) * lr
        ds = np.random.randn(n) * lr * 0.5
        new_centers = centers + dc
        new_sides = np.maximum(sides + ds, 0.01)
        
        if _is_feasible(new_centers, new_sides):
            sc = _score(new_centers, new_sides)
            if sc > best_score:
                best_score = sc
                best_centers = new_centers.copy()
                best_sides = new_sides.copy()
                centers = new_centers
                sides = new_sides
    return best_centers, best_sides, best_score


def _grid_baseline(n=10):
    """Grid packing baseline."""
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _generate_random_splits(n=10, num_ratios=8):
    """Generate random guillotine split trees."""
    splits = []
    # Random binary tree structure via random split orientations
    # For n leaves, we need n-1 internal nodes
    # Use a random BST-like structure
    def _gen(count):
        if count == 1:
            return []
        # Split into two groups
        left_count = np.random.randint(1, count)
        right_count = count - left_count
        orient = np.random.randint(2)
        ratio = np.random.uniform(0.2, 0.8)
        left_splits = _gen(left_count)
        right_splits = _gen(right_count)
        return [(ratio, orient)] + left_splits + right_splits
    return _gen(n)


def _optimize_scales(centers, sides, max_iters=3000):
    """Try to scale up squares while maintaining feasibility."""
    n = len(sides)
    best = sides.copy()
    best_score = _score(centers, sides)
    
    for it in range(max_iters):
        if _time_left() < 10:
            break
        # Try scaling one square up
        i = np.random.randint(n)
        factor = np.random.uniform(1.001, 1.02)
        new_sides = sides.copy()
        new_sides[i] *= factor
        if _is_feasible(centers, new_sides):
            sc = _score(centers, new_sides)
            if sc > best_score:
                best_score = sc
                best = new_sides.copy()
                sides = new_sides
        else:
            # Try scaling down a random one and up another
            j = np.random.randint(n)
            if j != i:
                new_sides2 = sides.copy()
                new_sides2[j] *= np.random.uniform(0.98, 0.999)
                new_sides2[i] *= np.random.uniform(1.005, 1.02)
                if _is_feasible(centers, new_sides2):
                    sc = _score(centers, new_sides2)
                    if sc > best_score:
                        best_score = sc
                        best = new_sides2.copy()
                        sides = new_sides2
    return best, best_score


def squares_in_square10() -> tuple:
    """
    Pack 10 squares in the unit square maximizing total side length.
    Uses guillotine cut tree structure + local search optimization.
    """
    n = 10
    np.random.seed(42)
    
    # Start with grid baseline
    centers, sides = _grid_baseline(n)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _score(centers, sides)
    
    # Try several random guillotine packings
    num_attempts = 200
    for attempt in range(num_attempts):
        if _time_left() < 30:
            break
        
        splits = _generate_random_splits(n)
        result = _try_packing(splits, n)
        if result is None:
            continue
        c, s = result
        sc = _score(c, s)
        if sc > best_score:
            best_score = sc
            best_centers = c.copy()
            best_sides = s.copy()
            centers = c
            sides = s
    
    # Local search refinement
    if _time_left() > 30:
        best_centers, best_sides, best_score = _local_search(
            best_centers, best_sides, max_iters=20000
        )
    
    # Try to scale up
    if _time_left() > 15:
        best_sides, best_score = _optimize_scales(best_centers, best_sides, max_iters=10000)
    
    # Final feasibility check
    if not _is_feasible(best_centers, best_sides):
        # Fall back to grid
        best_centers, best_sides = _grid_baseline(n)
        best_score = _score(best_centers, best_sides)
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
