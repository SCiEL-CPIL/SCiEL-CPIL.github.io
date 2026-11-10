# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations


_TOL = 1e-7


def _is_valid(centers, sides, tol=_TOL):
    """Vectorized constraint check: inside unit square, positive sides, non-overlapping."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Boundary: each square must lie inside [-tol, 1+tol]^2
    if np.any(centers - sides[:, None] / 2.0 < -tol):
        return False
    if np.any(centers + sides[:, None] / 2.0 > 1.0 + tol):
        return False
    # Pairwise non-overlap
    iu = np.triu_indices(n, k=1)
    dx = np.abs(centers[iu[0], 0] - centers[iu[1], 0])
    dy = np.abs(centers[iu[0], 1] - centers[iu[1], 1])
    t = (sides[iu[0]] + sides[iu[1]]) / 2.0 - tol
    if np.any((dx < t) & (dy < t)):
        return False
    return True


def _objective(centers, sides):
    return float(np.sum(sides))


def _clamp_to_unit(centers, sides):
    half = sides / 2.0
    centers[:, 0] = np.clip(centers[:, 0], half, 1.0 - half)
    centers[:, 1] = np.clip(centers[:, 1], half, 1.0 - half)
    return centers


def _max_side_vectorized(centers, sides, tol=_TOL, max_iters=90):
    """Binary-search the max side for EVERY square simultaneously (vectorized feasibility)."""
    n = len(sides)
    iu = np.triu_indices(n, k=1)
    a, b = iu[0], iu[1]
    lo = sides.copy()
    hi = np.full(n, 1.0)
    # Precompute which indices each square is involved in
    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        # For each pair, test with the *proposed* side of both endpoints
        sA = mid[a]
        sB = mid[b]
        t = (sA + sB) / 2.0 - tol
        dx = np.abs(centers[a, 0] - centers[b, 0])
        dy = np.abs(centers[a, 1] - centers[b, 1])
        pair_bad = (dx < t) & (dy < t)
        # boundary feasibility for each square at its mid side
        boundary_ok = (centers[:, 0] - mid / 2.0 >= -tol) & (centers[:, 0] + mid / 2.0 <= 1.0 + tol) & \
                      (centers[:, 1] - mid / 2.0 >= -tol) & (centers[:, 1] + mid / 2.0 <= 1.0 + tol)
        # a square is feasible if boundary ok AND no incident pair is bad
        overlap_bad = np.zeros(n, dtype=bool)
        overlap_bad[a] |= pair_bad
        overlap_bad[b] |= pair_bad
        feasible = boundary_ok & (~overlap_bad)
        lo = np.where(feasible, mid, lo)
        hi = np.where(feasible, hi, mid)
    return lo


def _grow_all(centers, sides, tol=_TOL, max_rounds=60, max_iters=90):
    """Iteratively grow all squares to convergence (vectorized growth)."""
    for _ in range(max_rounds):
        new_s = _max_side_vectorized(centers, sides, tol, max_iters)
        if np.all(new_s <= sides + 1e-12):
            break
        sides = new_s
    return sides


def _penalty_energy(centers, sides, tol=_TOL, beta=100.0):
    """Smooth penalty: 0 when all constraints satisfied. log-sum-exp of overlap margins."""
    n = len(sides)
    iu = np.triu_indices(n, k=1)
    a, b = iu[0], iu[1]
    energy = 0.0
    # Boundary soft penalties
    d = np.minimum(
        np.minimum(centers - sides[:, None] / 2.0, 1.0 - centers - sides[:, None] / 2.0),
        1.0
    )
    d = np.min(d, axis=1)  # min distance to any of 4 walls
    under = np.clip(tol - d, 0.0, None)
    energy += beta * np.sum(under ** 2)
    # Overlap: margin m = min(t-dx, t-dy); penalty on positive part, smoothed via logsumexp
    t = (sides[a] + sides[b]) / 2.0 - tol
    dx = np.abs(centers[a, 0] - centers[b, 0])
    dy = np.abs(centers[a, 1] - centers[b, 1])
    mx = t - dx
    my = t - dy
    # smooth max(mx, my, 0)
    vmax = np.maximum(np.maximum(mx, my), 0.0)
    # logsumexp of [mx,my,0] minus vmax, only where vmax>0
    e = np.exp(np.vstack([mx, my, np.zeros_like(mx)]) - vmax[None, :])
    lse = vmax + np.log(np.sum(e, axis=0))
    pos = lse * (vmax > 0)
    energy += beta * np.sum(pos ** 2)
    return float(energy)


def _smooth_optimize(centers, sides, tol=_TOL, max_iters=1500, lr=0.02,
                     beta_start=10.0, beta_end=600.0, obj_w=0.05):
    """Smooth gradient descent on penalty + side-growth objective."""
    n = len(sides)
    c = centers.copy()
    s = sides.copy()
    best_obj = _objective(c, s)
    best_c = c.copy()
    best_s = s.copy()
    lr_decay = 0.9997
    for it in range(max_iters):
        frac = it / max_iters
        beta = beta_start + (beta_end - beta_start) * (frac ** 1.5)
        grad_c, grad_s = _grad_penalty(c, s, tol, beta)
        grad_s = grad_s - obj_w * np.ones(n)  # maximize sum of sides
        cur_lr = lr * (lr_decay ** it)
        c -= cur_lr * grad_c
        s -= cur_lr * grad_s
        s = np.clip(s, 0.01, None)
        c = _clamp_to_unit(c, s)
        if _is_valid(c, s, tol):
            obj = _objective(c, s)
            if obj > best_obj + 1e-10:
                best_obj = obj
                best_c = c.copy()
                best_s = s.copy()
    return best_c, best_s


def _grad_penalty(centers, sides, tol=_TOL, beta=100.0):
    """Analytic-ish gradient of penalty energy w.r.t. centers and sides."""
    n = len(sides)
    gc = np.zeros((n, 2))
    gs = np.zeros(n)
    # Boundary
    for i in range(n):
        cx, cy = centers[i]
        d = [cx - sides[i] / 2, 1.0 - cx - sides[i] / 2, cy - sides[i] / 2, 1.0 - cy - sides[i] / 2]
        # left
        if d[0] < tol:
            gc[i, 0] += -2 * beta * (tol - d[0]); gs[i] += beta * (tol - d[0])
        if d[1] < tol:
            gc[i, 0] += -2 * beta * (tol - d[1]); gs[i] += beta * (tol - d[1])
        if d[2] < tol:
            gc[i, 1] += -2 * beta * (tol - d[2]); gs[i] += beta * (tol - d[2])
        if d[3] < tol:
            gc[i, 1] += -2 * beta * (tol - d[3]); gs[i] += beta * (tol - d[3])
    # Overlap: push along axis of least penetration
    iu = np.triu_indices(n, k=1)
    a, b = iu[0], iu[1]
    t = (sides[a] + sides[b]) / 2.0 - tol
    dx = centers[a, 0] - centers[b, 0]
    dy = centers[a, 1] - centers[b, 1]
    ax = np.abs(dx)
    ay = np.abs(dy)
    ox = np.clip(t - ax, 0.0, None)
    oy = np.clip(t - ay, 0.0, None)
    both = (ox > 0) & (oy > 0)
    onlyx = (ox > 0) & (oy <= 0)
    onlyy = (oy > 0) & (ox <= 0)
    # both: use axis of least overlap
    use_x = both & (ox <= oy)
    use_y = both & (oy < ox)
    for idx in np.where(use_x)[0]:
        i, j = a[idx], b[idx]
        sgn = 1.0 if dx[idx] >= 0 else -1.0
        gc[i, 0] += sgn * 2 * beta * ox[idx]
        gc[j, 0] -= sgn * 2 * beta * ox[idx]
        gs[i] += beta * ox[idx]; gs[j] += beta * ox[idx]
    for idx in np.where(use_y)[0]:
        i, j = a[idx], b[idx]
        sgn = 1.0 if dy[idx] >= 0 else -1.0
        gc[i, 1] += sgn * 2 * beta * oy[idx]
        gc[j, 1] -= sgn * 2 * beta * oy[idx]
        gs[i] += beta * oy[idx]; gs[j] += beta * oy[idx]
    for idx in np.where(onlyx)[0]:
        i, j = a[idx], b[idx]
        sgn = 1.0 if dx[idx] >= 0 else -1.0
        gc[i, 0] += sgn * beta * ox[idx]
        gc[j, 0] -= sgn * beta * ox[idx]
        gs[i] += 0.5 * beta * ox[idx]; gs[j] += 0.5 * beta * ox[idx]
    for idx in np.where(onlyy)[0]:
        i, j = a[idx], b[idx]
        sgn = 1.0 if dy[idx] >= 0 else -1.0
        gc[i, 1] += sgn * beta * oy[idx]
        gc[j, 1] -= sgn * beta * oy[idx]
        gs[i] += 0.5 * beta * oy[idx]; gs[j] += 0.5 * beta * oy[idx]
    return gc, gs


def _force_separate(centers, sides, tol=_TOL, max_iters=400, lr=0.05):
    n = len(sides)
    c = centers.copy()
    for _ in range(max_iters):
        forces = np.zeros((n, 2))
        iu = np.triu_indices(n, k=1)
        a, b = iu[0], iu[1]
        t = (sides[a] + sides[b]) / 2.0 - tol
        dx = c[a, 0] - c[b, 0]
        dy = c[a, 1] - c[b, 1]
        ax = np.abs(dx); ay = np.abs(dy)
        bad = (ax < t) & (ay < t)
        ox = np.clip(t - ax, 0.0, None)
        oy = np.clip(t - ay, 0.0, None)
        if not np.any(bad):
            break
        for idx in np.where(bad)[0]:
            i, j = a[idx], b[idx]
            if ox[idx] <= oy[idx]:
                sgn = 1.0 if dx[idx] >= 0 else -1.0
                forces[i, 0] += sgn * ox[idx] * 10
                forces[j, 0] -= sgn * ox[idx] * 10
            else:
                sgn = 1.0 if dy[idx] >= 0 else -1.0
                forces[i, 1] += sgn * oy[idx] * 10
                forces[j, 1] -= sgn * oy[idx] * 10
        c += lr * forces
        c = _clamp_to_unit(c, sides)
    return c


def _quadtree_layout(depth, n_squares, deadline):
    """
    Generate a quadtree-based packing layout.
    Recursively partition the unit square into rectangular cells.
    Each leaf cell gets one square with side = min(cell_w, cell_h).
    Returns (centers, sides) or None if infeasible.
    """
    if time.time() > deadline:
        return None
    
    # Build quadtree: list of (x, y, w, h, is_leaf, square_idx)
    # Start with root
    nodes = [(0.0, 0.0, 1.0, 1.0, False, -1)]
    
    # We need exactly n_squares leaves
    # Strategy: recursively split until we have enough cells, then assign squares
    # Use a priority queue approach: split the most "square-like" or largest cell first
    
    import heapq
    
    # Priority: prefer splitting cells that are large and roughly square
    # to maximize min(w,h) of resulting cells
    def cell_priority(node):
        x, y, w, h, is_leaf, idx = node
        if is_leaf:
            return 0
        # Prefer splitting larger cells, and more square-like ones
        return min(w, h) * (min(w,h)/max(w,h))
    
    # We'll do a BFS-like expansion
    # Target: exactly n_squares leaves
    # Start with 1 node, need n_squares-1 splits
    
    splits_needed = n_squares - 1
    for split_i in range(splits_needed):
        if time.time() > deadline:
            return None
        # Find the best node to split (non-leaf with highest priority)
        best_idx = -1
        best_prio = -1
        for i, node in enumerate(nodes):
            if not node[4]:  # not a leaf
                p = cell_priority(node)
                if p > best_prio:
                    best_prio = p
                    best_idx = i
        
        if best_idx == -1:
            break
        
        x, y, w, h, _, _ = nodes[best_idx]
        
        # Choose split direction: split the longer dimension
        if w >= h:
            # Split vertically
            split_frac = 0.5
            # Try to make the split more balanced if possible
            nodes[best_idx] = (x, y, w * split_frac, h, False, -1)
            nodes.append((x + w * split_frac, y, w * (1 - split_frac), h, False, -1))
        else:
            # Split horizontally
            split_frac = 0.5
            nodes[best_idx] = (x, y, w, h * split_frac, False, -1)
            nodes.append((x, y + h * split_frac, w, h * (1 - split_frac), False, -1))
    
    # Now we should have n_squares nodes (all should be leaves conceptually)
    if len(nodes) != n_squares:
        # If we have more, merge some; if fewer, we can't proceed cleanly
        # For now, just use what we have
        pass
    
    # Assign squares to cells
    centers = np.zeros((n_squares, 2))
    sides = np.zeros(n_squares)
    
    for i, (x, y, w, h, _, _) in enumerate(nodes[:n_squares]):
        side = min(w, h)
        # Center the square in the cell
        cx = x + w / 2.0
        cy = y + h / 2.0
        centers[i] = [cx, cy]
        sides[i] = side
    
    return centers, sides


def _quadtree_optimize(centers, sides, tol=_TOL, max_rounds=50, deadline=None):
    """
    Optimize a quadtree-based layout by adjusting split positions.
    Uses local search on split parameters.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    # Represent the layout as a sequence of splits
    # For simplicity, we'll do local perturbations of centers
    
    rng = np.random.default_rng(12345)
    c = centers.copy()
    s = sides.copy()
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Try to grow all squares
        ns = _grow_all(c.copy(), s.copy(), tol)
        if _is_valid(c, ns, tol):
            obj = _objective(c, ns)
            if obj > best_obj + 1e-10:
                best_obj = obj
                best_c = c.copy()
                best_s = ns.copy()
                s = ns.copy()
        
        # Perturb centers slightly
        nc = c.copy()
        for i in range(n):
            dx = rng.uniform(-0.02, 0.02)
            dy = rng.uniform(-0.02, 0.02)
            nc[i, 0] = np.clip(c[i, 0] + dx, s[i] / 2, 1.0 - s[i] / 2)
            nc[i, 1] = np.clip(c[i, 1] + dy, s[i] / 2, 1.0 - s[i] / 2)
        
        nc = _force_separate(nc, s, tol, max_iters=200)
        nc = _clamp_to_unit(nc, s)
        
        if _is_valid(nc, s, tol):
            tns = _grow_all(nc.copy(), s.copy(), tol)
            if _is_valid(nc, tns, tol):
                obj = _objective(nc, tns)
                if obj > best_obj + 1e-10:
                    best_obj = obj
                    best_c = nc.copy()
                    best_s = tns.copy()
                    c = nc.copy()
                    s = tns.copy()
    
    return best_c, best_s


def _local_optimize(centers, sides, tol=_TOL, max_rounds=60, seed=42):
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    rng = np.random.default_rng(seed)
    c = centers.copy(); s = sides.copy()
    for rnd in range(max_rounds):
        # grow
        ns = _grow_all(c.copy(), s.copy(), tol)
        if _is_valid(c, ns, tol) and _objective(c, ns) > best_obj + 1e-10:
            best_obj = _objective(c, ns); best_c = c.copy(); best_s = ns.copy(); s = ns.copy()
        # smooth optimize
        if rnd % 2 == 0:
            sc, ss = _smooth_optimize(c.copy(), s.copy(), tol, max_iters=900, lr=0.012,
                                      beta_start=5.0, beta_end=500.0)
            if _is_valid(sc, ss, tol) and _objective(sc, ss) > best_obj + 1e-10:
                best_obj = _objective(sc, ss); best_c = sc.copy(); best_s = ss.copy()
                c = sc.copy(); s = ss.copy()
        # force separate + regrow
        if rnd % 3 == 0:
            fc = _force_separate(c.copy(), s, tol)
            if _is_valid(fc, s, tol):
                fs = _grow_all(fc.copy(), s.copy(), tol)
                if _is_valid(fc, fs, tol) and _objective(fc, fs) > best_obj + 1e-10:
                    best_obj = _objective(fc, fs); best_c = fc.copy(); best_s = fs.copy()
                    c = fc.copy(); s = fs.copy()
        # targeted perturbations
        for i in range(n):
            for _ in range(4):
                dx = rng.uniform(-0.03, 0.03); dy = rng.uniform(-0.03, 0.03)
                nc = c.copy()
                nc[i, 0] = np.clip(c[i, 0] + dx, s[i] / 2, 1.0 - s[i] / 2)
                nc[i, 1] = np.clip(c[i, 1] + dy, s[i] / 2, 1.0 - s[i] / 2)
                if _is_valid(nc, s, tol):
                    tns = _grow_all(nc.copy(), s.copy(), tol)
                    if _is_valid(nc, tns, tol) and _objective(nc, tns) > best_obj + 1e-10:
                        best_obj = _objective(nc, tns); best_c = nc.copy(); best_s = tns.copy()
                        c = nc.copy(); s = tns.copy()
        # shake all
        if rnd % 5 == 1:
            nc = c.copy()
            for i in range(n):
                nc[i, 0] = np.clip(c[i, 0] + rng.uniform(-0.05, 0.05), s[i] / 2, 1.0 - s[i] / 2)
                nc[i, 1] = np.clip(c[i, 1] + rng.uniform(-0.05, 0.05), s[i] / 2, 1.0 - s[i] / 2)
            nc = _force_separate(nc, s, tol)
            if _is_valid(nc, s, tol):
                tns = _grow_all(nc.copy(), s.copy(), tol)
                if _is_valid(nc, tns, tol) and _objective(nc, tns) > best_obj + 1e-10:
                    best_obj = _objective(nc, tns); best_c = nc.copy(); best_s = tns.copy()
                    c = nc.copy(); s = tns.copy()
    return best_c, best_s


def _init_grid():
    n = 10
    centers = np.zeros((n, 2)); sides = np.zeros(n)
    k = 3; side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k; rows = idx // k
    centers[:9] = np.stack([(cols + 0.5) * side, (rows + 0.5) * side], axis=1)
    sides[:9] = side
    centers[9] = [0.5, 0.5]; sides[9] = 1.0 / 6.0
    return centers, sides


def _init_het1():
    centers = np.zeros((10, 2)); sides = np.zeros(10)
    centers[0] = [0.25, 0.25]; sides[0] = 0.5
    side_s = 1.0 / 6.0
    idx = 1
    for col in range(6):
        centers[idx] = [(col + 0.5) * side_s, 0.5 + 0.5 * side_s]; sides[idx] = side_s; idx += 1
    for col in range(3):
        centers[idx] = [0.5 + (col + 0.5) * side_s, 0.5 * side_s]; sides[idx] = side_s; idx += 1
    return centers, sides


def _init_het2():
    centers = np.zeros((10, 2)); sides = np.zeros(10)
    centers[0] = [0.2, 0.2]; sides[0] = 0.4
    centers[1] = [0.8, 0.2]; sides[1] = 0.4
    side_s = 0.25
    idx = 2
    for row in range(2):
        for col in range(4):
            centers[idx] = [(col + 0.5) * side_s, 0.4 + (row + 0.5) * side_s]; sides[idx] = side_s; idx += 1
    return centers, sides


def _init_het3():
    centers = np.zeros((10, 2)); sides = np.zeros(10)
    centers[0] = [0.225, 0.225]; sides[0] = 0.45
    centers[1] = [0.775, 0.775]; sides[1] = 0.45
    side_s = 0.15
    idx = 2
    pos = [[0.225, 0.575], [0.225, 0.875], [0.525, 0.225], [0.825, 0.225],
           [0.525, 0.525], [0.05, 0.575], [0.05, 0.875], [0.825, 0.525]]
    for p in pos:
        centers[idx] = p; sides[idx] = side_s; idx += 1
    return centers, sides


def _init_het4():
    centers = np.zeros((10, 2)); sides = np.zeros(10)
    centers[0] = [0.2, 0.2]; sides[0] = 0.4
    centers[1] = [0.825, 0.825]; sides[1] = 0.35
    side_s = 0.15
    idx = 2
    pos = [[0.55, 0.2], [0.8, 0.2], [0.2, 0.55], [0.2, 0.8],
           [0.55, 0.5], [0.55, 0.8], [0.2, 0.35], [0.55, 0.35]]
    for p in pos:
        centers[idx] = p; sides[idx] = side_s; idx += 1
    return centers, sides


def _init_random(seed):
    rng = np.random.default_rng(seed)
    centers = np.zeros((10, 2)); sides = np.zeros(10)
    for i in range(10):
        s = rng.uniform(0.1, 0.35); sides[i] = s
        centers[i, 0] = rng.uniform(s / 2, 1.0 - s / 2)
        centers[i, 1] = rng.uniform(s / 2, 1.0 - s / 2)
    return centers, sides


def squares_in_square10() -> tuple:
    start = time.time()
    deadline = start + 570.0  # Safety margin before 630s limit
    
    best_c = None
    best_s = None
    best_obj = -1.0
    
    # Phase 1: Try quadtree-based layouts
    for depth_seed in [0, 1, 2, 3, 4]:
        if time.time() > deadline:
            break
        
        # Generate quadtree layout with different random seeds for split decisions
        rng = np.random.default_rng(depth_seed)
        centers, sides = _quadtree_layout(depth_seed, 10, deadline)
        
        if centers is None:
            continue
        
        # Ensure validity
        centers = _clamp_to_unit(centers.copy(), sides.copy())
        if not _is_valid(centers, sides):
            centers = _force_separate(centers, sides, max_iters=1000)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides):
                continue
        
        # Optimize the quadtree layout
        oc, os_ = _quadtree_optimize(centers, sides, tol=_TOL, max_rounds=40, deadline=deadline)
        
        if _is_valid(oc, os_):
            obj = _objective(oc, os_)
            if obj > best_obj:
                best_obj = obj
                best_c = oc.copy()
                best_s = os_.copy()
    
    # Phase 2: Try traditional initializations
    inits = [_init_grid(), _init_het1(), _init_het2(), _init_het3(), _init_het4()]
    for s in [123, 456, 789, 1011, 2020, 3030, 4040, 5050, 6060, 7070]:
        inits.append(_init_random(seed=s))
    
    for idx, (centers, sides) in enumerate(inits):
        if time.time() > deadline:
            break
        
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        centers = _force_separate(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            centers = _force_separate(centers, sides, max_iters=1500)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides):
                continue
        
        # Use remaining time for local optimization
        remaining = deadline - time.time()
        max_rounds = max(10, int(remaining / 5))  # ~5s per round
        
        oc, os_ = _local_optimize(centers, sides, seed=42 + idx * 100, max_rounds=max_rounds)
        
        if _is_valid(oc, os_):
            obj = _objective(oc, os_)
            if obj > best_obj:
                best_obj = obj
                best_c = oc.copy()
                best_s = os_.copy()
    
    # Fallback
    if best_c is None:
        best_c, best_s = _init_grid()
    elif not _is_valid(best_c, best_s):
        best_c, best_s = _init_grid()
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            fs = _grow_all(best_c.copy(), best_s.copy(), tol=1e-10, max_rounds=40, max_iters=100)
            if _is_valid(best_c, fs, _TOL) and _objective(best_c, fs) > _objective(best_c, best_s) + 1e-12:
                best_s = fs.copy()
            else:
                break
    
    return best_c, best_s


# EVOLVE-BLOCK-END
