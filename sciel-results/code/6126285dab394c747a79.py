import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def soft_min_max_ratio(points, beta):
    """
    Compute a smooth surrogate for log((d_min / d_max)^2).
    Returns 2*log(d_min_soft) - 2*log(d_max_soft) for optimization stability.
    """
    dists = spd.pdist(points)
    if len(dists) == 0:
        return -1e6
    
    x_max = dists / beta
    m_max = np.max(x_max)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(x_max - m_max)))
    d_max_soft = beta * log_sum_exp_max
    
    x_min = -dists / beta
    m_min = np.max(x_min)
    log_sum_exp_min = m_min + np.log(np.exp(x_min - m_min))
    d_min_soft = -beta * log_sum_exp_min
    
    if d_max_soft <= 0 or d_min_soft <= 0:
        return -1e6
        
    log_ratio = np.log(d_min_soft) - np.log(d_max_soft)
    return 2 * log_ratio

def points_from_params(params):
    """
    Convert 28-dim param vector to 16x2 points.
    Params: first 2 fixed at (0,0) and (1,0), remaining 28 are free coords.
    """
    n = 16
    pts = np.zeros((n, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    rest = params.reshape(14, 2)
    pts[2:] = rest
    return pts

def generate_hex_init(rng):
    """Generate a hexagonal lattice-based initialization."""
    # Hexagonal lattice points
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = np.sqrt(3) / 2 * j
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts -= np.mean(hex_pts, axis=0)
    d_center = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(d_center)[:16]
    hex_init = hex_pts[idx]
    hex_init -= np.mean(hex_init, axis=0)
    
    # Pick two closest points as P0 and P1
    dists_hex = spd.pdist(hex_init)
    # Find the pair with minimum distance
    min_idx = np.argmin(dists_hex)
    # Convert linear index to (i, j) pair
    n = 16
    # For pdist, index k corresponds to pair (i, j) where:
    # i = (k // (n-1)) if k < (n-1)*n/2 else ... 
    # Actually, let's just iterate
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    i_idx, j_idx = pairs[min_idx]
    
    # Reorder so that i_idx is at position 0 and j_idx is at position 1
    order = [i_idx, j_idx]
    for k in range(n):
        if k != i_idx and k != j_idx:
            order.append(k)
    hex_init = hex_init[order]
    
    # Translate and rotate so P0=(0,0), P1=(1,0)
    p0 = hex_init[0]
    p1 = hex_init[1]
    hex_init = hex_init - p0
    d = np.linalg.norm(p1)
    if d > 1e-9:
        angle = np.arctan2(p1[1], p1[0])
        rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        hex_init = hex_init @ rot.T
        hex_init = hex_init / d
    
    return hex_init[2:].flatten()

def generate_ring_init(rng):
    """Generate a ring-based initialization."""
    angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
    r = 1.5 + rng.randn() * 0.3
    pts = np.column_stack((r*np.cos(angles), r*np.sin(angles)))
    return pts.flatten()

def generate_random_init(rng):
    """Generate a random initialization."""
    return rng.randn(28) * 2.0

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    
    global_deadline = time.time() + 600  # Safety margin before 630s limit
    
    def objective_smooth(params, beta):
        try:
            pts = points_from_params(params)
            if not np.all(np.isfinite(pts)):
                return 1e6
            return -soft_min_max_ratio(pts, beta)
        except Exception:
            return 1e6
    
    def objective_exact(params):
        try:
            pts = points_from_params(params)
            if not np.all(np.isfinite(pts)):
                return 1e6
            score = calculate_score(pts)
            return -score
        except Exception:
            return 1e6
    
    best_params = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(generate_hex_init(rng))
    inits.append(generate_ring_init(rng))
    
    for _ in range(8):
        inits.append(generate_random_init(rng))
    
    start_time = time.time()
    
    # Phase 1: Smooth optimization with L-BFGS-B, multiple betas
    betas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02]
    
    current_params = inits[0].copy()
    
    for beta in betas:
        if time.time() > global_deadline - 120:
            break
        
        best_local_params = None
        best_local_score = 1e6
        
        for init in inits[:6]:  # Use top 6 initializations
            if time.time() > global_deadline - 120:
                break
                
            try:
                res = minimize(
                    objective_smooth, init, args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.fun < best_local_score:
                    best_local_score = res.fun
                    best_local_params = res.x.copy()
            except Exception:
                pass
        
        if best_local_params is not None:
            current_params = best_local_params
            pts = points_from_params(current_params)
            score = calculate_score(pts)
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
            
            # Add best result to initializations for next round
            inits.insert(0, best_local_params.copy())
            if len(inits) > 6:
                inits = inits[:6]
    
    # Phase 2: Exact objective polishing with Nelder-Mead
    if time.time() < global_deadline - 60:
        if best_params is not None:
            start_polish = best_params
        else:
            start_polish = inits[0]
        
        # Multiple Nelder-Mead polishing attempts with different initial simplex sizes
        nm_configs = [
            {'maxiter': 2000, 'fatol': 1e-12, 'xatol': 1e-12, 'initial_simplex_size': 0.1},
            {'maxiter': 2000, 'fatol': 1e-12, 'xatol': 1e-12, 'initial_simplex_size': 0.05},
            {'maxiter': 2000, 'fatol': 1e-14, 'xatol': 1e-14, 'initial_simplex_size': 0.01},
        ]
        
        for config in nm_configs:
            if time.time() > global_deadline - 30:
                break
            
            try:
                res = minimize(
                    objective_exact, start_polish,
                    method='Nelder-Mead',
                    options=config
                )
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
                start_polish = res.x.copy()
            except Exception:
                pass
    
    # Final check
    if best_params is None:
        best_params = rng.randn(28)
        
    final_points = points_from_params(best_params)
    
    if not np.all(np.isfinite(final_points)):
        final_points = np.zeros((n, d))
        
    return final_points
