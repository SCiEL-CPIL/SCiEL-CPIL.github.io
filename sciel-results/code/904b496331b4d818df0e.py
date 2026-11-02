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


def _soft_min_obj(pts, r_target, beta=50.0):
    """
    Soft-min surrogate: we want to maximize the minimum of all distances (pairwise/2 and wall).
    Equivalently, minimize -soft_min of distances.
    We use a log-sum-exp based soft-min: soft_min(x) = -logsumexp(-x) / beta approx min(x) as beta->inf.
    But for numerical stability, we'll work with the objective: sum of soft violations.
    
    Actually, let's define a smooth objective that is 0 when all constraints are satisfied.
    For each constraint c_i (distance >= 2*r_target for pairs, distance >= r_target for walls),
    we use a smooth max(0, violation) = softplus(beta * violation) / beta.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Pair constraints: dist >= 2*r_target
    pair_viol = 2.0 * r_target - dist
    iu = np.triu_indices(N, k=1)
    pair_viol = pair_viol[iu]
    
    # Wall constraints
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = r_target - d
    
    all_viol = np.concatenate([pair_viol, wall_viol])
    
    # Soft-plus: max(0, x) approximated by log(1+exp(beta*x))/beta
    # For stability: softplus(z) = log(1+exp(z)) for z>0, z for z<0
    beta_val = beta
    z = beta_val * all_viol
    # Use stable softplus
    softplus = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    obj = np.sum(softplus) / beta_val
    
    return float(obj)


def _soft_min_grad(pts, r_target, beta=50.0):
    """Gradient of the soft-min objective."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    pair_viol = 2.0 * r_target - dist
    iu = np.triu_indices(N, k=1)
    pair_viol = pair_viol[iu]
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = r_target - d
    
    all_viol = np.concatenate([pair_viol, wall_viol])
    
    beta_val = beta
    z = beta_val * all_viol
    # sigmoid(z) = 1/(1+exp(-z))
    sigmoid = np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    
    # d(softplus)/d(viol) = sigmoid(z)
    # d(viol)/d(pts) for pairs:
    # pair_viol_ij = 2*r - dist_ij
    # d(pair_viol_ij)/d(pts_i) = (pts_i - pts_j) / dist_ij
    # d(pair_viol_ij)/d(pts_j) = (pts_j - pts_i) / dist_ij
    
    grad = np.zeros_like(pts)
    
    # Pair contributions
    pair_viol_full = 2.0 * r_target - dist
    np.fill_diagonal(pair_viol_full, 0)
    z_full = beta_val * pair_viol_full
    sigmoid_full = np.where(z_full > 0, 1.0 / (1.0 + np.exp(-z_full)), np.exp(z_full) / (1.0 + np.exp(z_full)))
    # Only upper triangle
    iu = np.triu_indices(N, k=1)
    sig_pair = sigmoid_full[iu]
    
    # For each pair (i,j), contribution to grad[i] += sig_pair * (pts[i]-pts[j])/dist[i,j]
    # and grad[j] -= sig_pair * (pts[i]-pts[j])/dist[i,j]
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d_ij = dist[i, j]
        if d_ij < 1e-12:
            continue
        coeff = sig_pair[idx] / d_ij
        diff_vec = pts[i] - pts[j]
        grad[i] += coeff * diff_vec
        grad[j] -= coeff * diff_vec
    
    # Wall contributions
    z_wall = beta_val * wall_viol
    sigmoid_wall = np.where(z_wall > 0, 1.0 / (1.0 + np.exp(-z_wall)), np.exp(z_wall) / (1.0 + np.exp(z_wall)))
    
    for i in range(N):
        # wall_viol_i = r_target - min(d_left, d_right, d_bottom, d_top)
        d_left = pts[i, 0] + HALF
        d_right = HALF - pts[i, 0]
        d_bottom = pts[i, 1] + HALF
        d_top = HALF - pts[i, 1]
        d_min = np.min([d_left, d_right, d_bottom, d_top])
        # Which wall is the minimum?
        if abs(d_left - d_min) < 1e-12:
            grad[i, 0] += sigmoid_wall[i]
        if abs(d_right - d_min) < 1e-12:
            grad[i, 0] -= sigmoid_wall[i]
        if abs(d_bottom - d_min) < 1e-12:
            grad[i, 1] += sigmoid_wall[i]
        if abs(d_top - d_min) < 1e-12:
            grad[i, 1] -= sigmoid_wall[i]
    
    return grad


def _violation(pts, r_target):
    """Total violation: sum of pairwise overlaps + wall penetrations (hard)."""
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


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=3000, beta=50.0):
    """Simulated annealing to minimize soft-min violation for target radius r_target."""
    pts = pts0.copy()
    current_obj = _soft_min_obj(pts, r_target, beta)
    best_pts = pts.copy()
    best_obj = current_obj
    
    t_start = 0.03
    t_end = 0.00005
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_obj = _soft_min_obj(new_pts, r_target, beta)
        
        delta = new_obj - current_obj
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
    
    return best_pts, best_obj


def _lbfgs_polish(pts0, r_target, deadline, beta=100.0):
    """Use L-BFGS-B to minimize soft-min violation for target radius."""
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
        return _soft_min_grad(p, r_target, beta)
    
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-12})
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
    """Binary search on radius with violation minimization."""
    best_pts = pts_start.copy()
    best_r = _compute_r(pts_start)
    
    r_lo = best_r
    r_hi = 0.07138
    
    for _ in range(12):
        if time.time() > deadline - 5.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        
        # SA phase
        pts_sa, obj_sa = _simulated_annealing(best_pts, r_mid, deadline - 5.0, rng, n_steps=2500, beta=50.0)
        
        # L-BFGS polish
        try:
            pts_lbfgs, obj_lbfgs = _lbfgs_polish(pts_sa, r_mid, deadline - 5.0, beta=100.0)
        except Exception:
            pts_lbfgs = pts_sa
            obj_lbfgs = obj_sa
        
        r_achieved = _compute_r(pts_lbfgs)
        
        if obj_lbfgs < 1e-10 and r_achieved >= r_mid - 1e-9:
            r_lo = r_mid
            best_pts = pts_lbfgs.copy()
            best_r = r_achieved
        else:
            r_hi = r_mid
    
    return best_pts, best_r


def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin before 150s limit
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    def try_improve(pts0, r_target, n_sa_steps, beta_sa=50.0, beta_lbfgs=100.0):
        nonlocal best_pts, best_r
        if time.time() > deadline - 10.0:
            return
        
        pts_sa, obj_sa = _simulated_annealing(pts0, r_target, deadline - 10.0, rng, n_steps=n_sa_steps, beta=beta_sa)
        
        try:
            pts_lbfgs, obj_lbfgs = _lbfgs_polish(pts_sa, r_target, deadline - 10.0, beta=beta_lbfgs)
        except Exception:
            pts_lbfgs = pts_sa
            obj_lbfgs = obj_sa
        
        r = _compute_r(pts_lbfgs)
        if r > best_r:
            best_r = r
            best_pts = pts_lbfgs.copy()
    
    # Phase 1: Multi-start with moderate target
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.05),
        _random_init(rng),
        _random_init(rng),
    ]
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        try_improve(init, 0.07, 2000)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Binary search on radius
    if time.time() < deadline - 20.0:
        best_pts, best_r = _binary_search_radius(best_pts, deadline, rng)
    
    # Phase 3: Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        r_target = best_r * 1.0005
        try_improve(best_pts, r_target, 3000, beta_sa=100.0, beta_lbfgs=200.0)
    
    # Phase 4: Very final polish
    if time.time() < deadline - 2.0:
        r_target = best_r * 1.0002
        try_improve(best_pts, r_target, 2000, beta_sa=200.0, beta_lbfgs=500.0)
    
    return best_pts
