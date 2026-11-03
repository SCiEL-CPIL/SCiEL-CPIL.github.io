import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
HALF = 0.5
TIME_LIMIT = 150.0
SAFETY = 10.0  # Safety margin in seconds
START = time.time()

def _radius(pts):
    """Exact feasible common radius (vectorized, double precision)."""
    pts = np.asarray(pts, dtype=np.float64)
    if not (pts.shape == (N, 3) and np.all(np.isfinite(pts))):
        return 0.0
    
    # Pairwise distances
    # For N=100, cdist is efficient
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    # Wall distances
    min_wall = np.min(HALF - np.max(np.abs(pts), axis=1))
    
    r = 0.5 * min(min_pair, 2.0 * min_wall)
    if not np.isfinite(r) or r < 0:
        return 0.0
    return float(r)

def _bcc_points(n_max=4):
    """BCC lattice points centred at origin, nearest N kept."""
    pts = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                pts.append([i, j, k])
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=np.float64)
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:N]]

def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)

def _init_bcc_scaled():
    """Initialize BCC lattice scaled to fit roughly in the cube."""
    V = _bcc_points(4)
    ext = np.max(V.max(axis=0) - V.min(axis=0))
    # Scale so that the extent is slightly less than 1 to allow for radius
    # r ~ 0.11, so extent ~ 1 - 2r ~ 0.78
    # But BCC packing efficiency is ~0.68, so for 100 spheres, volume ~ 100 * 4/3 * pi * r^3.
    # Let's just scale to fit within [-0.4, 0.4] initially and let optimizer expand.
    s = 0.8 / ext
    C = s * V
    C -= C.mean(axis=0)
    return C

def _soft_surrogate(pts, beta):
    """
    Smooth objective: -r + penalty.
    Uses log-sum-exp to approximate min distance smoothly.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 0.0)
    ii, jj = np.triu_indices(N, 1)
    d2u = d2[ii, jj]
    d2u = np.clip(d2u, 1e-12, None)
    
    # Soft min of distances via log-sum-exp
    # We want to maximize min(d). Minimizing -logsumexp(d) approximates -min(d).
    # Actually, we want to maximize r.
    # Let's minimize -r_smooth + penalty.
    
    # d = sqrt(d2)
    # logsumexp(d) is a smooth approximation of max(d)? No, min(d) is what we care about.
    # min(d) = -max(-d).
    # So we can use -logsumexp(-d) / beta?
    # Standard soft min: (1/beta) * log(sum(exp(-beta * d_i))) approximates min(d_i) as beta->inf.
    # Let's use this.
    
    d = np.sqrt(d2u)
    beta_val = 1.0 / beta  # beta is the temperature, small beta = sharp min
    
    # Prevent overflow in exp
    d_min = np.min(d)
    # logsumexp(-beta_val * d) = -beta_val * d_min + log(sum(exp(-beta_val * (d - d_min))))
    # This is stable.
    exp_terms = np.exp(-beta_val * (d - d_min))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_d = d_min - log_sum / beta_val
    
    # Wall distances
    w = HALF - np.max(np.abs(pts), axis=1)
    w_min = np.min(w)
    exp_terms_w = np.exp(-beta_val * (w - w_min))
    log_sum_w = np.log(np.sum(exp_terms_w))
    soft_min_w = w_min - log_sum_w / beta_val
    
    # r = 0.5 * min(soft_min_d, 2 * soft_min_w)
    # To make it smooth, we can just use the two soft mins and take a smooth min of them?
    # Or just return -0.5 * min(soft_min_d, 2*soft_min_w) + penalty.
    # The min is still nonsmooth. 
    # Let's use a smooth min for the final combination too.
    
    val1 = soft_min_d
    val2 = 2.0 * soft_min_w
    
    # Smooth min of val1, val2
    # min(a,b) approx -log(exp(-beta*a) + exp(-beta*b))/beta
    m = np.minimum(val1, val2)
    exp1 = np.exp(-beta_val * (val1 - m))
    exp2 = np.exp(-beta_val * (val2 - m))
    soft_r = m - np.log(exp1 + exp2) / beta_val
    
    r_smooth = 0.5 * soft_r
    
    # Penalty for wall breach (hard constraint violation)
    breach = np.clip(-w, 0.0, None)
    pen = 100.0 * np.sum(breach) + 1000.0 * np.max(breach)
    
    return -(r_smooth) + pen

def _polish_smooth(pts, budget):
    """
    Polish using smooth surrogate and L-BFGS-B.
    """
    deadline = START + (TIME_LIMIT - SAFETY)
    best_pts = pts.copy()
    best_r = _radius(pts)
    
    # Try a few iterations with decreasing beta (increasing sharpness)
    betas = [0.05, 0.02, 0.01, 0.005]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        def obj(x):
            return _soft_surrogate(x.reshape(N, 3), beta)
        
        try:
            # Use L-BFGS-B with bounds
            bounds = [(-HALF, HALF)] * (N * 3)
            res = minimize(obj, best_pts.ravel(), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            cand = res.x.reshape(N, 3)
            r = _radius(cand)
            if np.all(np.isfinite(cand)) and r > best_r:
                best_r = r
                best_pts = cand.copy()
                
        except Exception:
            pass
            
    return best_pts, best_r

def construct():
    rng = np.random.default_rng(42)
    deadline = START + (TIME_LIMIT - SAFETY)
    
    # 1. Generate BCC lattice
    V = _bcc_points(4)
    
    # 2. Global Optimization of Scale and Translation
    # Parameters: [s, tx, ty, tz]
    # C = s * V + t
    
    def objective_global(params):
        s = params[0]
        t = params[1:4]
        
        # Ensure s is positive
        if s <= 0:
            return 1e6
            
        C = s * V + t
        
        # Wall check
        min_wall = np.min(HALF - np.max(np.abs(C), axis=1))
        if min_wall < 0:
            # Penalty for being outside
            return 100.0 * (-min_wall) + 10.0
        
        # Pairwise min distance
        D = cdist(C, C)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D)
        
        r = 0.5 * min(min_pair, 2.0 * min_wall)
        return -r

    # Initial guess for scale
    # Extent of V
    V_extent = np.max(V.max(axis=0) - V.min(axis=0))
    # We want s * V_extent ~ 0.8 (leaving room for radius)
    s_init = 0.8 / V_extent
    t_init = [0.0, 0.0, 0.0]
    x0 = np.concatenate(([s_init], t_init))
    
    best_pts = None
    best_r = -1.0
    
    # Run global optimization
    try:
        res = minimize(objective_global, x0, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        s_opt = res.x[0]
        t_opt = res.x[1:4]
        C_opt = s_opt * V + t_opt
        r_opt = _radius(C_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pts = C_opt.copy()
            
    except Exception:
        pass
        
    # Fallback to grid if global opt failed or is bad
    if best_pts is None or best_r < 0.1:
        grid_pts = _init_grid()
        r_grid = _radius(grid_pts)
        if r_grid > best_r:
            best_r = r_grid
            best_pts = grid_pts.copy()
    else:
        # Also check grid just in case
        grid_pts = _init_grid()
        r_grid = _radius(grid_pts)
        if r_grid > best_r:
            best_r = r_grid
            best_pts = grid_pts.copy()

    # 3. Local Polish with Smooth Surrogate
    if time.time() < deadline:
        pol_pts, pol_r = _polish_smooth(best_pts, deadline)
        if pol_r > best_r:
            best_r = pol_r
            best_pts = pol_pts

    # 4. Final Validity Guard
    pts = np.asarray(best_pts, dtype=np.float64)
    if not (pts.shape == (N, 3) and np.all(np.isfinite(pts))):
        pts = _init_grid()
        
    # Ensure we are inside the cube (clip if necessary, though optimizer should keep them in)
    # Clipping might reduce radius, so only do if invalid
    if np.any(np.abs(pts) > HALF + 1e-9):
        pts = np.clip(pts, -HALF, HALF)
        
    return pts
