"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _radius(c):
    """Compute the maximum common radius for centers c of shape (N,2)."""
    n = c.shape[0]
    # Pairwise distances (upper triangle only for efficiency)
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    min_pair = np.min(dists) / 2.0
    # Boundary: distance to nearest edge
    boundary = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
    min_boundary = np.min(boundary)
    return min(min_pair, min_boundary)


def _relax(c, r_target, max_iter=200):
    """Relaxation: push overlapping circles apart and keep them inside the square.
    Returns updated centers."""
    c = c.copy()
    n = c.shape[0]
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 2 * r_target and d > 1e-12:
                    # Push apart
                    push = (2 * r_target - d) / d * 0.5
                    c[i] += push * diff
                    c[j] -= push * diff
                    moved = True
                elif d <= 1e-12:
                    # Coincident: separate randomly
                    c[i] += np.random.randn(2) * 1e-4
                    c[j] -= np.random.randn(2) * 1e-4
                    moved = True
        # Keep inside square
        c[:, 0] = np.clip(c[:, 0], -0.5 + r_target, 0.5 - r_target)
        c[:, 1] = np.clip(c[:, 1], -0.5 + r_target, 0.5 - r_target)
        if not moved:
            break
    return c


def _grow(c, r, growth_rate=0.0005, max_iters=5000, deadline=None):
    """Gradually grow the radius, relaxing at each step."""
    c = c.copy()
    current_r = r
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        current_r += growth_rate
        c = _relax(c, current_r, max_iter=50)
        # Check if we can still satisfy the constraint
        actual_r = _radius(c)
        if actual_r < current_r - 1e-9:
            # Couldn't maintain, step back
            current_r -= growth_rate * 2
            break
    return c, current_r


def _init_7x7_plus():
    """7x7 grid (49 pts) plus one extra, slightly perturbed."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Add 50th point at center of one cell, then perturb all slightly
    pts = np.vstack([grid, np.array([0.0, 0.0])])
    rng = np.random.RandomState(123)
    pts += rng.randn(50, 2) * 0.001
    return pts


def _init_jitter8():
    """8x8 grid with 14 points removed, jittered."""
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Remove 14 random points to get 50
    rng = np.random.RandomState(456)
    remove_idx = rng.choice(64, size=14, replace=False)
    keep_idx = np.setdiff1d(np.arange(64), remove_idx)
    pts = grid[keep_idx].copy()
    pts += rng.randn(50, 2) * 0.002
    return pts


def _init_rand():
    """Random initialization with minimum spacing."""
    rng = np.random.RandomState(789)
    pts = rng.rand(N, 2) - 0.5
    return pts


def _init_hexagonal():
    """Hexagonal packing initialization."""
    # Hexagonal lattice with spacing s
    s = 0.14
    pts = []
    row = 0
    while len(pts) < N:
        y = -0.5 + s * 0.866 * (row + 1)
        if y > 0.5 - s * 0.866:
            break
        offset = s * 0.5 if row % 2 == 1 else 0.0
        x = -0.5 + s * 0.5 + offset
        while x < 0.5 - s * 0.5:
            if len(pts) < N:
                pts.append([x, y])
            x += s
        row += 1
    # Pad if needed
    while len(pts) < N:
        pts.append([rng.rand() - 0.5, rng.rand() - 0.5])
    pts = np.array(pts[:N])
    rng = np.random.RandomState(321)
    pts += rng.randn(N, 2) * 0.001
    return pts


def _local_optimize(c, r_init, deadline):
    """Iterative local optimization: grow + relax cycles with multiple strategies."""
    c = c.copy()
    best_r = _radius(c)
    best_c = c.copy()
    
    # Phase 1: Grow from initial radius
    c, r_grown = _grow(c, r_init, growth_rate=0.0003, max_iters=3000, deadline=deadline)
    r = _radius(c)
    if r > best_r:
        best_r = r
        best_c = c.copy()
    
    # Phase 2: Fine-tune with smaller growth steps
    for attempt in range(5):
        if time.time() > deadline:
            break
        c2 = best_c + np.random.randn(N, 2) * 0.0005
        c2, r2 = _grow(c2, best_r * 0.99, growth_rate=0.0002, max_iters=2000, deadline=deadline)
        r2 = _radius(c2)
        if r2 > best_r:
            best_r = r2
            best_c = c2.copy()
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple initialization strategies
    initializers = [
        ('grid7', _init_7x7_plus),
        ('jitter8', _init_jitter8),
        ('hex', _init_hexagonal),
        ('rand', _init_rand),
    ]
    
    best_c = None
    best_r = -1
    
    for name, init_fn in initializers:
        if time.time() > deadline:
            break
        c = init_fn()
        r = _radius(c)
        
        # Do local optimization
        c_opt, r_opt = _local_optimize(c, r, deadline)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # If we still have time, do more refinement on the best
    if best_c is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline:
                break
            c_pert = best_c + np.random.randn(N, 2) * 0.0003
            c_opt, r_opt = _local_optimize(c_pert, best_r * 0.995, deadline)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
    
    # Ensure validity: clip to square with margin
    if best_c is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_r = _radius(best_c)
    
    # Final clip to ensure inside square
    margin = 1e-12
    best_c = np.clip(best_c, -0.5 + margin, 0.5 - margin)
    
    # Final validity check
    final_r = _radius(best_c)
    if final_r <= 0 or not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_c
# EVOLVE-BLOCK-END
