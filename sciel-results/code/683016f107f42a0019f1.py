"""Improved local-search for 50 equal circles in a unit square."""
import time
import numpy as np

N = 50
HALF = 0.5

def _exact_radius(pts):
    """Compute the exact packing radius in double precision."""
    pts = np.asarray(pts, dtype=np.float64)
    # margins to the four sides
    margins = np.minimum(np.minimum(HALF - pts[:, 0], HALF + pts[:, 0]),
                         np.minimum(HALF - pts[:, 1], HALF + pts[:, 1]))
    r_margin = float(np.min(margins))
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = float(np.min(dist) / 2.0)
    return min(r_margin, r_pair)

def _smooth_obj(pts, beta):
    """
    Smooth surrogate to maximize. We maximize a soft-min of the "constraints":
      g_ij = |c_i - c_j| - 2r_target  (we want g_ij >= 0)
      g_i  = HALF - |c_ik| - r_target
    Instead of fixing r_target, we optimize a smooth function that rewards large r:
      We use a log-sum-exp trick to get a smooth approximation of -min(g).
    Here we define for a given r:
      S = logsumexp(-beta * g)  (smaller is better; g large positive -> -beta*g very negative -> S small)
    We want to maximize r such that S is small. Simpler: use a direct smooth objective:
      f = - (1/beta) * log( sum_{ij} exp(-beta*|c_i-c_j|/2) + sum_i exp(-beta*margin_i) )
    This is a smooth approximation of min(min_pair/2, min_margin).
    """
    pts = np.asarray(pts, dtype=np.float64)
    # margins
    m = np.minimum(np.minimum(HALF - pts[:, 0], HALF + pts[:, 0]),
                   np.minimum(HALF - pts[:, 1], HALF + pts[:, 1]))
    # pairwise distances / 2
    diff = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    # avoid sqrt for speed? we need /2 of distance
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, np.inf)
    vals = np.concatenate([dist / 2.0, m])
    # log-sum-exp of -beta * vals
    z = -beta * vals
    zmax = np.max(z)
    s = np.sum(np.exp(z - zmax))
    f = -(zmax + np.log(s)) / beta
    return f

def _smooth_grad(pts, beta):
    """Gradient of the smooth objective (to maximize)."""
    pts = np.asarray(pts, dtype=np.float64)
    m = np.minimum(np.minimum(HALF - pts[:, 0], HALF + pts[:, 0]),
                   np.minimum(HALF - pts[:, 1], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    vals = np.concatenate([dist / 2.0, m])
    z = -beta * vals
    zmax = np.max(z)
    exp_z = np.exp(z - zmax)
    w = exp_z / np.sum(exp_z)
    # gradient of f = -(zmax + log s)/beta
    # df/dval_k = w_k  (since d logsumexp/dz = w, and dz/dval = -beta, and f = -(zmax+logs)/beta)
    # Actually: f = -logsumexp(z)/beta, z = -beta*vals
    # df/dvals_k = w_k
    grad_vals = w.copy()
    # Now back-propagate to pts
    grad = np.zeros_like(pts)
    # pairwise part: vals[:N*(N-1)/2] are dist/2 for i<j
    # We'll reconstruct using the full matrix approach
    # dist/2 for all pairs (i,j), i!=j; but we only have upper triangle in vals.
    # Easier: compute gradient via the full pairwise matrix.
    # Let's recompute with full matrix for clarity (N=50 is small)
    # Rebuild: for each pair (i,j), i<j, contribution to grad[i] and grad[j]
    # grad of (dist_ij/2) w.r.t. pts[i] = (pts[i]-pts[j]) / dist_ij
    # grad of (dist_ij/2) w.r.t. pts[j] = (pts[j]-pts[i]) / dist_ij
    # We need the weights for each pair. The vals array has dist/2 in the first N*(N-1)/2 entries
    # in row-major order of the upper triangle.
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            w_ij = grad_vals[idx]
            d = pts[i] - pts[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij > 1e-12:
                unit = d / dist_ij
                grad[i] += w_ij * unit
                grad[j] -= w_ij * unit
            idx += 1
    # margins part: vals[N*(N-1)/2:] are m_i for each point
    for i in range(N):
        w_i = grad_vals[N * (N - 1) // 2 + i]
        # m_i = min(HALF - x, HALF + x, HALF - y, HALF + y)
        # gradient is the subgradient of the min; we use the active constraint
        x, y = pts[i]
        # find which margin is smallest
        cands = [HALF - x, HALF + x, HALF - y, HALF + y]
        k = int(np.argmin(cands))
        if k == 0:
            grad[i, 0] += w_i * (-1.0)
        elif k == 1:
            grad[i, 0] += w_i * (1.0)
        elif k == 2:
            grad[i, 1] += w_i * (-1.0)
        else:
            grad[i, 1] += w_i * (1.0)
    return grad

def _project(pts):
    """Project points into the square [-0.5, 0.5]^2."""
    pts = pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], -HALF, HALF)
    pts[:, 1] = np.clip(pts[:, 1], -HALF, HALF)
    return pts

def _hex_seed():
    """Hexagonal lattice seed, trimmed to N points, centered."""
    # Hexagonal lattice spacing s
    # We want to fit ~50 points. Try a few spacings and pick the one that fits.
    best_pts = None
    best_r = -1
    for s in np.linspace(0.05, 0.15, 40):
        # Generate hex lattice points in a large region
        dx = s
        dy = s * np.sqrt(3) / 2
        xs = np.arange(-1, 1 + 1e-9, dx)
        ys = np.arange(-1, 1 + 1e-9, dy)
        pts_list = []
        for j, y in enumerate(ys):
            x_off = 0 if j % 2 == 0 else dx / 2
            for x in xs:
                pts_list.append((x + x_off, y))
        pts = np.array(pts_list)
        # Keep only those inside the square
        mask = (pts[:, 0] >= -HALF) & (pts[:, 0] <= HALF) & (pts[:, 1] >= -HALF) & (pts[:, 1] <= HALF)
        pts = pts[mask]
        if len(pts) < N:
            continue
        # If more than N, we need to select N. Use a simple approach: take the first N after sorting by distance from center
        if len(pts) > N:
            d2 = np.sum(pts * pts, axis=1)
            order = np.argsort(d2)
            pts = pts[order[:N]]
        r = _exact_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    if best_pts is None:
        # fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
    return best_pts

def _random_seed():
    """Random seed inside the square."""
    return np.random.uniform(-HALF, HALF, size=(N, 2))

def _local_search(pts0, time_limit, beta_init=20.0, beta_end=500.0):
    """Local search with smooth gradient ascent, adaptive step, and projection."""
    pts = np.array(pts0, dtype=np.float64)
    best_pts = pts.copy()
    best_r = _exact_radius(pts)
    beta = beta_init
    step = 0.01
    last_improve = time.time()
    no_improve_time = 0.0
    max_no_improve = 5.0  # seconds without improvement -> reduce step or restart locally

    while time.time() < time_limit:
        # Compute gradient
        g = _smooth_grad(pts, beta)
        # Normalize gradient to control step size
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        g = g / gn
        # Take step
        new_pts = pts + step * g
        new_pts = _project(new_pts)
        # Evaluate exact radius
        new_r = _exact_radius(new_pts)
        if new_r > best_r + 1e-15:
            pts = new_pts
            best_pts = pts.copy()
            best_r = new_r
            last_improve = time.time()
            # Increase beta slowly to sharpen
            beta = min(beta * 1.05, beta_end)
            # Keep step or increase slightly
            step = min(step * 1.01, 0.05)
        else:
            # No improvement
            no_improve_time = time.time() - last_improve
            if no_improve_time > max_no_improve:
                # Reduce step and increase beta more
                step *= 0.7
                beta = min(beta * 1.5, beta_end)
                # Occasionally add a small random kick
                if np.random.rand() < 0.3:
                    pts = pts + 0.005 * np.random.randn(N, 2)
                    pts = _project(pts)
                last_improve = time.time()
                no_improve_time = 0.0
            else:
                # Small step reduction
                step *= 0.95
        # Check deadline
        if time.time() > time_limit - 0.5:
            break
    return best_pts, best_r

def construct():
    """
    Main entry point. Returns an array of shape (50, 2) with centers.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    rng = np.random.default_rng(42)
    np.random.seed(42)

    # Generate multiple seeds
    seeds = []
    # Hexagonal seed
    seeds.append(_hex_seed())
    # Square grid seed
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    seeds.append(grid[:N])
    # A few random seeds
    for _ in range(3):
        seeds.append(_random_seed())

    best_pts = None
    best_r = -1

    # Local search on each seed with a time budget
    n_seeds = len(seeds)
    time_per_seed = (deadline - start_time) / n_seeds

    for i, seed in enumerate(seeds):
        if time.time() > deadline - 5.0:
            break
        t_end = min(deadline - 2.0, start_time + (i + 1) * time_per_seed)
        pts, r = _local_search(seed, t_end)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Final polish pass on the best configuration
    if best_pts is not None and time.time() < deadline - 10.0:
        # Use a higher beta and smaller steps for polish
        polish_time = min(deadline - 2.0, time.time() + 20.0)
        # Temporarily increase beta_end for polish
        global beta_end
        old_beta_end = 500.0
        # We'll just call local search again with a tighter budget
        pts, r = _local_search(best_pts, polish_time, beta_init=100.0, beta_end=2000.0)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Ensure validity: project and verify
    if best_pts is None:
        # Should not happen, but fallback
        best_pts = seeds[0]

    best_pts = _project(best_pts)
    # Final exact radius check
    final_r = _exact_radius(best_pts)

    return best_pts
