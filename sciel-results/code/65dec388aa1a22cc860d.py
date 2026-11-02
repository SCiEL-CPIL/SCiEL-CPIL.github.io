"""Packing of 50 equal circles in a unit square via soft-min optimization + SA + L-BFGS polish."""
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


def _soft_min_obj(pts, r_target, beta=500.0):
    """
    Smooth surrogate for the packing problem.
    We want to maximize r such that all pairwise distances >= 2*r and wall distances >= r.
    Here we minimize a soft penalty that is smooth in r_target.
    
    Objective: sum of soft-violations for pairs and walls.
    For pairs: max(0, 2*r_target - dist_i) approximated by softplus(beta*(2*r_target - dist_i))/beta
    For walls: max(0, r_target - wall_dist_i) approximated by softplus(beta*(r_target - wall_dist_i))/beta
    """
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    pair_viol = 2.0 * r_target - pair_dist
    # Softplus: log(1 + exp(x)) / beta
    # Clip to avoid overflow
    pair_viol_clipped = np.clip(pair_viol, -50, 50)
    soft_pair = np.log1p(np.exp(beta * pair_viol_clipped)) / beta
    
    # Wall distances
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r_target - wall_dist
    wall_viol_clipped = np.clip(wall_viol, -50, 50)
    soft_wall = np.log1p(np.exp(beta * wall_viol_clipped)) / beta
    
    return float(np.sum(soft_pair) + np.sum(soft_wall))


def _soft_min_grad(pts, r_target, beta=500.0):
    """
    Gradient of the soft-min objective with respect to pts.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    # Avoid division by zero
    dist_safe = np.where(dist > 1e-12, dist, 1e-12)
    
    # Gradient of softplus(beta*(2*r - dist)) w.r.t. pts
    # d/dpts [log(1+exp(beta*(2r - dist)))/beta]
    # = (1/beta) * exp(beta*(2r-dist))/(1+exp(beta*(2r-dist))) * (-beta) * d(dist)/dpts
    # = -sigmoid(beta*(2r-dist)) * d(dist)/dpts
    # where d(dist)/dpts_i = (pts_i - pts_j) / dist_ij
    
    pair_viol = 2.0 * r_target - dist
    pair_viol_clipped = np.clip(pair_viol, -50, 50)
    # sigmoid(x) = 1/(1+exp(-x))
    sig = 1.0 / (1.0 + np.exp(-beta * pair_viol_clipped))
    
    # Gradient contribution from pairs
    # For pair (i,j), d(dist_ij)/d(pts_i) = (pts_i - pts_j)/dist_ij
    # So grad_obj w.r.t. pts_i from pair (i,j) = -sig_ij * (pts_i - pts_j)/dist_ij
    # And w.r.t. pts_j = -sig_ij * (pts_j - pts_i)/dist_ij = sig_ij * (pts_i - pts_j)/dist_ij
    
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if d < 1e-12:
                continue
            s = sig[i, j]
            diff_ij = (pts[i] - pts[j]) / d
            grad[i] -= s * diff_ij
            grad[j] += s * diff_ij
    
    # Wall gradients
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r_target - wall_dist
    wall_viol_clipped = np.clip(wall_viol, -50, 50)
    sig_w = 1.0 / (1.0 + np.exp(-beta * wall_viol_clipped))
    
    # d(wall_dist)/d(pts_i) depends on which wall is closest
    for i in range(N):
        dl = d_left[i]
        dr = d_right[i]
        db = d_bottom[i]
        dt = d_top[i]
        wmin = wall_dist[i]
        s = sig_w[i]
        if abs(dl - wmin) < 1e-10:
            grad[i, 0] -= s  # d(wall_dist)/d(x) = 1 for left wall (x + 0.5)
        if abs(dr - wmin) < 1e-10:
            grad[i, 0] += s  # d(wall_dist)/d(x) = -1 for right wall (0.5 - x)
        if abs(db - wmin) < 1e-10:
            grad[i, 1] -= s  # d(wall_dist)/d(y) = 1 for bottom wall (y + 0.5)
        if abs(dt - wmin) < 1e-10:
            grad[i, 1] += s  # d(wall_dist)/d(y) = -1 for top wall (0.5 - y)
    
    return grad


def _violation(pts, r_target):
    """Total violation: sum of pairwise overlaps + wall penetrations."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist)
    iu = np.triu_indices(N, k=1)
    pair_viol = pair_viol[iu]
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d)
    
    return float(np.sum(pair_viol) + np.sum(wall_viol))


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=3000):
    """Simulated annealing to minimize violation for target radius r_target."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    # Temperature schedule
    t_start = 0.04
    t_end = 0.00005
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        # Random move: pick 1-3 circles and perturb
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    
    return best_pts, best_viol


def _lbfgs_polish(pts0, r_target, deadline, beta=500.0, maxiter=500):
    """Use L-BFGS-B to minimize the soft-min objective for target radius."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _soft_min_obj(p, r_target, beta)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        g = _soft_min_grad(p, r_target, beta)
        return g.ravel()
    
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-10})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts, float(res.fun)


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _binary_search_radius(pts_start, deadline, rng):
    """Binary search on radius with soft-min L-BFGS minimization."""
    best_pts = pts_start.copy()
    best_r = _compute_r(pts_start)
    
    r_lo = best_r
    r_hi = 0.07138
    
    for _ in range(12):
        if time.time() > deadline - 5.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        
        # Try to find feasible configuration at r_mid
        # First SA to get into basin
        pts_sa, viol_sa = _simulated_annealing(best_pts, r_mid, deadline - 5.0, rng, n_steps=1500)
        
        # Polish with L-BFGS on soft-min objective
        try:
            pts_lbfgs, obj_lbfgs = _lbfgs_polish(pts_sa, r_mid, deadline - 5.0, beta=500.0, maxiter=300)
        except Exception:
            pts_lbfgs = pts_sa
            obj_lbfgs = _soft_min_obj(pts_sa, r_mid, 500.0)
        
        # Check if feasible
        r_achieved = _compute_r(pts_lbfgs)
        viol = _violation(pts_lbfgs, r_mid)
        
        if viol < 1e-8 and r_achieved >= r_mid - 1e-9:
            r_lo = r_mid
            best_pts = pts_lbfgs.copy()
            best_r = r_achieved
        else:
            r_hi = r_mid
    
    return best_pts, best_r


def _sdp_relaxation_init(rng):
    """
    Attempt an SDP-inspired initialization by solving a relaxed version
    where we maximize the minimum pairwise distance using a smooth
    log-sum-exp surrogate, then use that as a starting point.
    """
    from scipy.optimize import minimize
    
    # Objective: minimize -log-sum-exp of pairwise distances (smooth max)
    # This encourages spreading points apart
    def obj(x):
        p = x.reshape(N, 2)
        diff = p[:, None, :] - p[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 0.0)
        dist = np.sqrt(dist_sq)
        iu = np.triu_indices(N, k=1)
        d = dist[iu]
        # log-sum-exp for numerical stability
        max_d = np.max(d)
        lse = max_d + np.log(np.sum(np.exp(d - max_d)))
        # Also penalize wall violations softly
        d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                           np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
        wall_pen = np.sum(np.exp(-10 * d_wall))
        return -lse + wall_pen
    
    def grad(x):
        p = x.reshape(N, 2)
        diff = p[:, None, :] - p[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 0.0)
        dist = np.sqrt(dist_sq)
        dist_safe = np.where(dist > 1e-12, dist, 1e-12)
        iu = np.triu_indices(N, k=1)
        d = dist[iu]
        max_d = np.max(d)
        exp_d = np.exp(d - max_d)
        sum_exp = np.sum(exp_d)
        weights = exp_d / sum_exp  # softmax weights
        
        g = np.zeros_like(p)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = weights[idx]
            diff_ij = p[i] - p[j]
            d_ij = dist_safe[i, j]
            g[i] += w * diff_ij / d_ij
            g[j] -= w * diff_ij / d_ij
        
        # Wall gradient
        d_left = p[:, 0] + HALF
        d_right = HALF - p[:, 0]
        d_bottom = p[:, 1] + HALF
        d_top = HALF - p[:, 1]
        d_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        for i in range(N):
            pen = np.exp(-10 * d_wall[i])
            if abs(d_left[i] - d_wall[i]) < 1e-10:
                g[i, 0] += 10 * pen
            if abs(d_right[i] - d_wall[i]) < 1e-10:
                g[i, 0] -= 10 * pen
            if abs(d_bottom[i] - d_wall[i]) < 1e-10:
                g[i, 1] += 10 * pen
            if abs(d_top[i] - d_wall[i]) < 1e-10:
                g[i, 1] -= 10 * pen
        
        return g.ravel()
    
    # Start from a hexagonal lattice
    x0 = _hex_init(rng, jitter=0.02).ravel()
    try:
        res = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    except Exception:
        return _hex_init(rng, jitter=0.02)


def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin before 150s limit
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    def try_improve(pts0, r_target, n_sa_steps, beta=500.0, maxiter=400):
        nonlocal best_pts, best_r
        if time.time() > deadline - 10.0:
            return
        
        # SA phase
        pts_sa, viol_sa = _simulated_annealing(pts0, r_target, deadline - 10.0, rng, n_steps=n_sa_steps)
        
        # L-BFGS polish on soft-min
        try:
            pts_lbfgs, obj_lbfgs = _lbfgs_polish(pts_sa, r_target, deadline - 10.0, beta=beta, maxiter=maxiter)
        except Exception:
            pts_lbfgs = pts_sa
        
        r = _compute_r(pts_lbfgs)
        if r > best_r:
            best_r = r
            best_pts = pts_lbfgs.copy()
    
    # Phase 0: SDP-inspired initialization
    if time.time() < deadline - 60.0:
        sdp_pts = _sdp_relaxation_init(rng)
        r_sdp = _compute_r(sdp_pts)
        if r_sdp > best_r:
            best_r = r_sdp
            best_pts = sdp_pts.copy()
    
    # Phase 1: Multi-start with moderate target
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.05),
        _random_init(rng),
        _random_init(rng),
    ]
    if best_pts is not None:
        inits.insert(0, best_pts.copy())
    
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        try_improve(init, 0.07, 1500, beta=500.0, maxiter=300)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Binary search on radius
    if time.time() < deadline - 25.0:
        best_pts, best_r = _binary_search_radius(best_pts, deadline, rng)
    
    # Phase 3: Final polish with tighter tolerance and higher beta
    # We try a few different beta values to ensure we escape soft-min plateaus
    if time.time() < deadline - 10.0:
        r_target = best_r * 1.0005
        # Use a higher beta for sharper approximation
        try_improve(best_pts, r_target, 1000, beta=2000.0, maxiter=500)
        
    # Phase 4: Very final polish with very high beta
    if time.time() < deadline - 3.0:
        r_target = best_r * 1.0002
        # Use an extremely high beta for near-exact min approximation
        try_improve(best_pts, r_target, 500, beta=5000.0, maxiter=800)
        
    # Phase 5: Final micro-polish
    if time.time() < deadline - 1.0:
        r_target = best_r * 1.0001
        try_improve(best_pts, r_target, 200, beta=10000.0, maxiter=1000)
    
    return best_pts
