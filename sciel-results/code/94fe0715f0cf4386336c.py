import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 140.0  # safety margin before 150s kill

def _score(x):
    """Compute the radius: min(min_pair_dist/2, min_wall_dist)."""
    x = np.asarray(x)
    if x.shape != (N, 2) or not np.all(np.isfinite(x)):
        return -1.0
    if np.any(x[:, 0] < -0.5 - 1e-12) or np.any(x[:, 0] > 0.5 + 1e-12) or \
       np.any(x[:, 1] < -0.5 - 1e-12) or np.any(x[:, 1] > 0.5 + 1e-12):
        return -1.0
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(np.min(dists_sq))
    wx = 0.5 - np.abs(x[:, 0])
    wy = 0.5 - np.abs(x[:, 1])
    min_wall = np.min(np.minimum(wx, wy))
    return min(min_dist / 2.0, min_wall)

def _hex_seed(r=0.071):
    """Generate hexagonal lattice points, return N of them."""
    dx = 2 * r
    dy = np.sqrt(3) * r
    points = []
    y = -0.5 + dy / 2
    row = 0
    while y <= 0.5:
        x_start = -0.5 + dx / 2
        if row % 2 == 1:
            x_start += dx / 2
        x = x_start
        while x <= 0.5:
            points.append([x, y])
            x += dx
        y += dy
        row += 1
    pts = np.array(points)
    if len(pts) < N:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N].copy()
    # Randomly select N
    idx = np.random.choice(len(pts), N, replace=False)
    return pts[idx].copy()

def _grid_seed():
    """Square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) < N:
        # Pad with small random offsets
        extra = np.random.uniform(-0.01, 0.01, (N - len(grid), 2))
        grid = np.vstack([grid, extra])
    return grid[:N].copy()

def _random_seed():
    """Random initialization in safe region."""
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _sa_optimize(x0, max_iters=50000, t0=0.01, t1=1e-5, deadline=None):
    """Simulated annealing with incremental scoring."""
    x = x0.copy()
    # Compute initial score and track min pair
    def _full_score_and_minpair(x):
        dx = x[:, 0:1] - x[:, 0:1].T
        dy = x[:, 1:2] - x[:, 1:2].T
        dists_sq = dx**2 + dy**2
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = np.min(dists_sq)
        min_dist = np.sqrt(min_dist_sq)
        wx = 0.5 - np.abs(x[:, 0])
        wy = 0.5 - np.abs(x[:, 1])
        min_wall = np.min(np.minimum(wx, wy))
        score = min(min_dist / 2.0, min_wall)
        # Find min pair index
        iu = np.triu_indices(N, k=1)
        pair_dists_sq = dists_sq[iu]
        min_pair_idx = np.argmin(pair_dists_sq)
        return score, min_pair_idx, min_dist, min_wall
    
    best_x = x.copy()
    best_score, _, _, _ = _full_score_and_minpair(x)
    
    # Precompute pair indices for efficiency
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature
        t = t0 * (t1 / t0) ** (it / max_iters)
        
        # Propose a move: pick random point, random direction
        i = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi)
        step = np.random.exponential(t) * 2.0
        dx = np.array([np.cos(angle), np.sin(angle)]) * step
        x_new = x.copy()
        x_new[i] += dx
        
        # Clamp to bounds
        x_new[i, 0] = np.clip(x_new[i, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        x_new[i, 1] = np.clip(x_new[i, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Compute new score (full computation for correctness)
        dx_mat = x_new[:, 0:1] - x_new[:, 0:1].T
        dy_mat = x_new[:, 1:2] - x_new[:, 1:2].T
        dists_sq = dx_mat**2 + dy_mat**2
        np.fill_diagonal(dists_sq, np.inf)
        min_dist = np.sqrt(np.min(dists_sq))
        wx = 0.5 - np.abs(x_new[:, 0])
        wy = 0.5 - np.abs(x_new[:, 1])
        min_wall = np.min(np.minimum(wx, wy))
        new_score = min(min_dist / 2.0, min_wall)
        
        delta = new_score - best_score
        if delta > 0 or np.random.rand() < np.exp(delta / t):
            x = x_new
            if new_score > best_score:
                best_score = new_score
                best_x = x.copy()
    
    return best_x, best_score

def _polish_lbfgs(x0, maxiter=100, deadline=None):
    """Polish with L-BFGS-B using a smooth objective."""
    def _obj(x):
        x = x.reshape(N, 2)
        dx = x[:, 0:1] - x[:, 0:1].T
        dy = x[:, 1:2] - x[:, 1:2].T
        dists_sq = dx**2 + dy**2
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        wx = 0.5 - np.abs(x[:, 0])
        wy = 0.5 - np.abs(x[:, 1])
        walls = np.minimum(wx, wy)
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        wall_vals = 2 * walls
        vals = np.concatenate([pair_dists, wall_vals])
        # Use log-sum-exp for stability
        m = np.max(vals)
        log_sum = np.log(np.sum(np.exp(-vals + m)))
        softmin = m - np.log(log_sum)
        return -softmin
    
    x_opt = x0.copy()
    try:
        res = minimize(
            _obj,
            x_opt.ravel(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res.success or res.fun < _obj(x_opt.ravel()):
            x_opt = res.x.reshape(N, 2)
    except Exception:
        pass
    
    # Clamp
    x_opt[:, 0] = np.clip(x_opt[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
    x_opt[:, 1] = np.clip(x_opt[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
    return x_opt

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_x = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations
    restarts = [
        _hex_seed(0.071),
        _hex_seed(0.068),
        _grid_seed(),
        _random_seed(),
        _hex_seed(0.075),
        _random_seed(),
    ]
    
    for k, x0 in enumerate(restarts):
        if time.time() > deadline:
            break
        
        # SA phase
        sa_deadline = min(deadline - 5.0, start_time + 20.0 * (k + 1))
        x_sa, score_sa = _sa_optimize(x0, max_iters=30000, t0=0.008, t1=1e-5, deadline=sa_deadline)
        
        if score_sa > best_score:
            best_score = score_sa
            best_x = x_sa.copy()
        
        # Polish phase
        if time.time() < deadline - 3.0:
            x_polished = _polish_lbfgs(x_sa, maxiter=80, deadline=deadline - 1.0)
            score_polished = _score(x_polished)
            if score_polished > best_score:
                best_score = score_polished
                best_x = x_polished.copy()
    
    if best_x is None:
        best_x = _grid_seed()
    
    # Final validity check
    if not np.all(np.isfinite(best_x)):
        best_x = _grid_seed()
    
    return best_x
