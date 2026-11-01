"""Packing of 50 equal circles in a unit square via soft-min optimization + SA + L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
IU = np.triu_indices(N, k=1)
N_PAIRS = len(IU[0])


def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _soft_min_obj(x, beta):
    """Objective: minimize -soft_min of all constraint distances (maximize the min)."""
    p = x.reshape(N, 2)
    d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                        np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    
    all_d = np.concatenate([d_wall, pair_d])
    m = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - m))
    log_sum = np.log(np.sum(exp_terms))
    r_soft = m - log_sum / beta
    return -r_soft


def _soft_min_grad(x, beta):
    """Gradient of -soft_min_r."""
    p = x.reshape(N, 2)
    d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                        np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    
    all_d = np.concatenate([d_wall, pair_d])
    m = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - m))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros((N, 2))
    
    # Wall gradients
    d_left = p[:, 0] + HALF
    d_right = HALF - p[:, 0]
    d_bottom = p[:, 1] + HALF
    d_top = HALF - p[:, 1]
    walls = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
    min_idx = np.argmin(walls, axis=1)
    
    for i in range(N):
        w = weights[i]
        if w > 1e-12:
            if min_idx[i] == 0:
                grad[i, 0] += w
            elif min_idx[i] == 1:
                grad[i, 0] -= w
            elif min_idx[i] == 2:
                grad[i, 1] += w
            else:
                grad[i, 1] -= w
    
    # Pair gradients
    for k in range(N_PAIRS):
        i, j = IU[0][k], IU[1][k]
        w = weights[N + k]
        if w > 1e-12:
            dij = p[i] - p[j]
            dist_ij = np.linalg.norm(dij)
            if dist_ij > 1e-12:
                g = w * dij / (2.0 * dist_ij)
                grad[i] += g
                grad[j] -= g
    
    return -grad.ravel()


def _violation(pts, r_target):
    """Hard violation for SA acceptance."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist[IU])
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d)
    
    return float(np.sum(pair_viol) + np.sum(wall_viol))


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=3000):
    """SA to minimize hard violation for target radius."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    t_start = 0.04
    t_end = 0.00005
    cooling = (t_end / t_start) ** (1.0 / max(n_steps - 1, 1))
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        n_move = int(rng.integers(1, 4))
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    
    return best_pts, best_viol


def _lbfgs_polish(pts0, r_target, deadline, beta=500.0, maxiter=300):
    """L-BFGS-B on the smooth objective."""
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _soft_min_obj(x, beta)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        return _soft_min_grad(x, beta)
    
    try:
        res = minimize(obj, pts0.ravel(), method='L-BFGS-B', jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 30})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    except Exception:
        return pts0


def _hex_init(rng, jitter=0.0):
    """Hexagonal lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            y = -HALF + a + row * dy
            if y < HALF - a + 1e-9:
                pts.append([x, y])
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


def _sdp_relaxation_init(rng):
    """
    Simple initialization based on symmetric placement.
    Uses a 5x10 grid with some perturbation.
    """
    # Try a structured approach: 5 rows, 10 columns
    pts = []
    n_rows = 5
    n_cols = 10
    for i in range(n_rows):
        for j in range(n_cols):
            x = (j + 0.5) / n_cols - 0.5
            y = (i + 0.5) / n_rows - 0.5
            pts.append([x, y])
    pts = np.array(pts[:N])
    pts += rng.normal(0, 0.02, size=pts.shape)
    return np.clip(pts, -HALF, HALF)


def _binary_search_radius(best_pts, r_lo, r_hi, deadline, rng, n_iters=8):
    """Binary search to push radius higher."""
    best_pts = best_pts.copy()
    best_r = r_lo
    
    for i in range(n_iters):
        if time.time() > deadline - 10.0:
            break
        r_mid = (r_lo + r_hi) / 2.0
        
        # SA to try to achieve r_mid
        pts_sa, viol_sa = _simulated_annealing(best_pts, r_mid, deadline - 8.0, rng, n_steps=2000)
        
        # Polish with increasing beta
        pts_final = pts_sa
        for beta in [100.0, 300.0, 800.0]:
            if time.time() > deadline - 5.0:
                break
            pts_final = _lbfgs_polish(pts_final, r_mid, deadline - 5.0, beta=beta, maxiter=200)
        
        r_achieved = _compute_r(pts_final)
        if r_achieved >= r_mid - 1e-9:
            r_lo = r_mid
            best_pts = pts_final.copy()
            best_r = r_achieved
        else:
            r_hi = r_mid
    
    return best_pts, best_r


def construct():
    start = time.time()
    deadline = start + 145.0
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    def try_improve(pts0, r_target, deadline_offset=5.0):
        nonlocal best_pts, best_r
        if time.time() > deadline - deadline_offset:
            return
        
        # SA phase
        pts_sa, _ = _simulated_annealing(pts0, r_target, deadline - deadline_offset, rng, n_steps=2000)
        
        # Multi-beta L-BFGS polish
        pts_final = pts_sa
        for beta in [50.0, 200.0, 600.0]:
            if time.time() > deadline - deadline_offset:
                break
            pts_final = _lbfgs_polish(pts_final, r_target, deadline - deadline_offset, beta=beta, maxiter=200)
        
        r = _compute_r(pts_final)
        if r > best_r:
            best_r = r
            best_pts = pts_final.copy()
    
    # Phase 1: Multi-start with diverse initializations
    inits = [
        _hex_init(rng, jitter=0.0),
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.0),
        _grid_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.05),
        _random_init(rng),
        _random_init(rng),
        _sdp_relaxation_init(rng),
    ]
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        try_improve(init, 0.068, deadline_offset=5.0)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Push radius via binary search
    if time.time() < deadline - 20.0:
        r_lo = best_r
        r_hi = 0.07138
        best_pts, best_r = _binary_search_radius(best_pts, r_lo, r_hi, deadline, rng, n_iters=8)
    
    # Phase 3: Final high-precision polish using soft-min radius maximization
    if time.time() < deadline - 8.0:
        for beta in [50.0, 150.0, 400.0]:
            if time.time() > deadline - 4.0:
                break
            try:
                res = minimize(_soft_min_obj, best_pts.ravel(), method='L-BFGS-B',
                               jac=_soft_min_grad, args=(beta,),
                               bounds=[(-HALF, HALF)] * (2 * N),
                               options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-12})
                pts_soft = res.x.reshape(N, 2)
                pts_soft = np.clip(pts_soft, -HALF, HALF)
                r_soft = _compute_r(pts_soft)
                if r_soft > best_r:
                    best_r = r_soft
                    best_pts = pts_soft
            except Exception:
                pass
    
    # Phase 4: One more targeted push from the best point
    if time.time() < deadline - 5.0:
        try_improve(best_pts, best_r + 0.0001, deadline_offset=3.0)
    
    return best_pts
