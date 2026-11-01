"""Packing of 50 equal circles in a unit square via Simulated Annealing and L-BFGS-B polish."""
import numpy as np
import time

N = 50
HALF = 0.5

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r

def _softmin_score(pts, t):
    """Smooth objective to maximize: soft-min of all constraints (pairwise/2 and boundary)."""
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    vals = np.concatenate([d, pair])
    # soft min = -logsumexp(-vals)/t
    m = np.min(vals)
    s = np.sum(np.exp(-t * (vals - m)))
    return m - np.log(s) / t

def _optimize_from(pts0, t, maxiter, deadline):
    """Run L-BFGS-B to maximize the soft-min score. Returns (pts, score)."""
    from scipy.optimize import minimize
    x0 = pts0.ravel()

    def neg_obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return -_softmin_score(p, t)

    def neg_grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        # numerical gradient
        eps = 1e-5
        g = np.zeros_like(x)
        base = neg_obj(x)
        for i in range(x.size):
            xp = x.copy(); xp[i] += eps
            g[i] = (neg_obj(xp) - base) / eps
        return g

    res = minimize(neg_obj, x0, method='L-BFGS-B',
                   jac=neg_grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
    return pts, float(-res.fun)

def _random_init(rng):
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))

def _hex_init(rng, jitter=0.0):
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        col = 0
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
            col += 1
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF + 1e-9, HALF - 1e-9)

def _grid_init(rng, jitter=0.0):
    m = 7  # 7x7 = 49, add one
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF + 1e-9, HALF - 1e-9)

def _simulated_annealing(pts0, deadline, rng, t_start=0.1, t_end=1e-5, alpha=0.9995):
    """
    Run Simulated Annealing to maximize the minimum distance.
    Returns the best configuration found.
    """
    current_pts = pts0.copy()
    current_r = _compute_r(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    t = t_start
    step_scale = 0.05 # Initial step size scale
    
    # Pre-compute indices for efficient distance calculation? 
    # For N=50, 1225 pairs. Calculating all distances is fast enough in NumPy.
    
    while t > t_end and time.time() < deadline:
        # Pick a random circle to move
        idx = rng.integers(0, N)
        
        # Generate a random move
        # Scale step size with temperature to explore more at high T
        move = rng.normal(0, step_scale * t / t_start, size=2)
        
        new_pts = current_pts.copy()
        new_pts[idx] += move
        
        # Keep inside the square (reflect or clip? Clipping might create overlaps with walls)
        # Better to just check validity. If outside, reject immediately or reflect.
        # Let's clip to ensure validity, but this might bias towards walls.
        # Actually, the radius calculation handles walls. If a point goes outside, 
        # the distance to wall becomes negative, reducing r significantly.
        # So we don't need to clip for the objective function to work, 
        # but we should ensure the final output is valid.
        
        new_r = _compute_r(new_pts)
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r > 0:
            # Always accept improvements
            current_pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        else:
            # Accept worse moves with probability exp(delta_r / t)
            # Note: delta_r is negative. 
            # We want to accept if random() < exp(delta_r / t)
            # Since delta_r is negative, exp(delta_r/t) is small.
            # This is standard SA for maximization.
            if delta_r / t > np.log(rng.random()):
                current_pts = new_pts
                current_r = new_r
        
        # Cool down
        t *= alpha
        # Optionally adjust step size
        # step_scale could be kept constant or reduced. 
        # Using t/t_start in move generation already scales down moves.

    return best_pts, best_r

def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin: 120s limit, leave 10s for polish/return
    rng = np.random.default_rng(12345)

    best_pts = None
    best_r = 0.0

    def update_best(pts):
        nonlocal best_pts, best_r
        r = _compute_r(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 1: Simulated Annealing from multiple starts
    # We will run SA for a fixed number of iterations or time slice per start
    # Then pick the best and polish.
    
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.05),
        _random_init(rng),
        _random_init(rng),
        _random_init(rng),
        _random_init(rng)
    ]
    
    # Allocate time for SA: ~80 seconds
    sa_deadline = start + 80.0
    
    for i, init in enumerate(inits):
        if time.time() > sa_deadline:
            break
        
        # Run SA with a shorter internal deadline to allow multiple starts
        # Let's give each SA run about 8-10 seconds if we have 8 starts
        # But we can just let it run until the global SA deadline, 
        # and rely on the first few starts being good.
        # Actually, it's better to run SA for a fixed number of iterations 
        # or a time slice, then move to the next start.
        
        # Let's use a time slice per start
        slice_deadline = min(sa_deadline, time.time() + 8.0)
        
        try:
            pts_sa, r_sa = _simulated_annealing(init, slice_deadline, rng)
            update_best(pts_sa)
        except Exception:
            pass

    # Phase 2: Polish the best result with L-BFGS-B
    if best_pts is not None and time.time() < deadline - 5.0:
        # Use a sharp soft-min for polishing
        t_sharp = 100.0
        try:
            pts_polished, score = _optimize_from(best_pts, t_sharp, 500, deadline)
            update_best(pts_polished)
        except Exception:
            pass
        
        # Try another polish with even sharper t if time allows
        if time.time() < deadline - 2.0:
            t_sharper = 300.0
            try:
                pts_polished2, score2 = _optimize_from(best_pts, t_sharper, 200, deadline)
                update_best(pts_polished2)
            except Exception:
                pass

    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Final safety check
    final_r = _compute_r(best_pts)
    if final_r <= 0:
        # If somehow invalid, return grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_pts
