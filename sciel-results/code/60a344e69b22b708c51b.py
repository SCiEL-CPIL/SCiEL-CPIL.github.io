import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time

N = 300

def _score(pts):
    """Compute the packing radius (higher is better)."""
    if len(pts) != N or pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(np.abs(pts) > 1.0):
        return 0.0
    r = np.linalg.norm(pts, axis=1)
    if np.any(r > 1.0):
        return 0.0
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1.0)
    min_pair = np.inf
    for a, b in pairs:
        d = np.linalg.norm(pts[a] - pts[b])
        if d < min_pair:
            min_pair = d
    if min_pair == np.inf:
        min_pair = 2.0
    min_boundary = 1.0 - np.max(r)
    return min(min_pair / 2.0, min_boundary)

def _soft_min_obj(beta, pts, cache):
    """Smooth soft-min objective and its gradient via log-sum-exp."""
    x = pts.ravel()
    N_ = len(x) // 2
    p = x.reshape(N_, 2)
    
    # Pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Mask diagonal
    idx = np.triu_indices(N_, 1)
    d = dists[idx] / 2.0
    if len(d) == 0:
        return 0.0, np.zeros_like(x)
    
    # Boundary distances
    r_pts = np.linalg.norm(p, axis=1)
    b = 1.0 - r_pts
    
    # Combined constraint vector
    all_x = np.concatenate([d, b])
    
    # Stable soft-min: soft_min = min - (1/beta)*log(sum(exp(-beta*(x - min))))
    m = np.min(all_x)
    shifted = all_x - m
    # weights
    e = np.exp(-beta * shifted)
    s = np.sum(e)
    obj = m - (1.0 / beta) * np.log(s)
    
    # gradient of soft_min w.r.t. all_x
    w = e / s  # normalized weights
    gd = w[:len(d)]
    gb = w[len(d):]
    
    # grad w.r.t. points
    g = np.zeros((N_, 2))
    # pairwise: d_k = dists[i,j]/2, d(d_k)/d(p_i) = (p_i - p_j)/dists[i,j] = (p_i-p_j)/(2*d_k)
    # But careful: d_k = dists/2, so dd_k/d p = (p_i - p_j)/dists[i,j]
    ii, jj = idx
    for k in range(len(ii)):
        i, j = ii[k], jj[k]
        dij = dists[i, j]
        if dij < 1e-12:
            continue
        f = gd[k] / dij
        g[i] += f * (p[i] - p[j])
        g[j] += f * (p[j] - p[i])
    # boundary: b_i = 1 - |p_i|, db/d p = -p/|p|
    for i in range(N_):
        if r_pts[i] < 1e-12:
            continue
        g[i] += gb[i] * (-p[i] / r_pts[i])
    
    return obj, g.ravel()

def _init_hex(n):
    """Hexagonal lattice init, select n closest to origin."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    k = int(1.0 / spacing) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) < n:
        # fallback to random
        return _init_random(n)
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists)[:n]
    return points[idx].copy()

def _init_random(n, seed=0):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(0, 0.95, n)
    pts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    return pts

def _init_rings(n):
    """Ring init: concentric rings with increasing count."""
    pts = [[0.0, 0.0]]
    remaining = n - 1
    ring = 1
    while remaining > 0:
        # number of points on this ring
        count = min(remaining, 6 * ring)
        if count <= 0:
            count = 6
        radius = ring * 0.052
        if radius > 0.98:
            radius = 0.98
        ang = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in ang:
            pts.append([radius * np.cos(a), radius * np.sin(a)])
        remaining -= count
        ring += 1
    pts = np.array(pts[:n])
    return pts

def _optimize_from(pts, beta_start, beta_end, time_limit):
    """Run L-BFGS-B with increasing beta from a starting config."""
    cache = {}
    x0 = pts.ravel()
    
    def objective(x):
        obj, grad = _soft_min_obj(beta_start, x, cache)
        # We want to maximize obj, so minimize -obj
        return -obj, -grad
    
    # Multi-stage: increase beta over time
    n_stages = 4
    t0 = time.time()
    total_time = time_limit
    time_per_stage = total_time / n_stages
    cur = x0.copy()
    best_x = x0.copy()
    best_obj = -np.inf
    # track best radius separately (exact)
    best_radius = _score(pts)
    
    for stage in range(n_stages):
        t_stage = t0 + stage * time_per_stage
        t_stage_end = t0 + (stage + 1) * time_per_stage
        # beta for this stage
        frac = stage / max(1, n_stages - 1)
        beta = beta_start * (beta_end / beta_start) ** frac
        
        # closure with current beta
        b_cur = beta
        def obj_fn(x, b=b_cur):
            obj, grad = _soft_min_obj(b, x, cache)
            return -obj, -grad
        
        res = minimize(obj_fn, cur, method='L-BFGS-B',
                       jac=True, options={'maxiter': 200, 'maxfun': 2000, 'disp': False})
        
        cur = res.x.reshape(N, 2)
        # evaluate exact radius
        r = _score(cur)
        if r > best_radius:
            best_radius = r
            best_x = cur.copy()
        
        if time.time() > t_stage_end:
            break
    
    return best_x, best_radius

def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill
    
    # Diverse initial configurations
    inits = [
        _init_hex(N),
        _init_rings(N),
        _init_random(N, seed=1),
        _init_random(N, seed=42),
    ]
    
    # De-polarize random inits a bit by a few gradient steps at low beta
    best_overall = inits[0]
    best_radius = _score(best_overall)
    
    # Allocate time: first pass a quick run on each init to find the best,
    # then focus remaining time on the best one with more stages.
    n_inits = len(inits)
    time_per_init = 25.0
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        # quick optimization
        x_opt, r_opt = _optimize_from(init, beta_start=8.0, beta_end=60.0,
                                       time_limit=min(time_per_init, deadline - time.time() - 20.0))
        if r_opt > best_radius:
            best_radius = r_opt
            best_overall = x_opt
    
    # Final intensive optimization on the best, with more stages and higher beta
    remaining = deadline - time.time()
    if remaining > 10.0:
        x_opt, r_opt = _optimize_from(best_overall, beta_start=20.0, beta_end=400.0,
                                       time_limit=remaining - 3.0)
        if r_opt > best_radius:
            best_radius = r_opt
            best_overall = x_opt
    
    # Ensure valid output
    if not np.all(np.isfinite(best_overall)) or best_overall.shape != (N, 2):
        return inits[0]
    # Clip to be strictly inside
    r = np.linalg.norm(best_overall, axis=1)
    mask = r > 1.0
    if np.any(mask):
        best_overall[mask] = best_overall[mask] / r[mask, None] * 0.999
    
    return best_overall
