import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 21
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def _compute_radii(x, y, W, H):
    """
    Compute the maximum radius for each circle given centers x, y and box W, H.
    r_i = min( dist_to_nearest_neighbor_i, dist_to_boundary_i )
    """
    # Distances to boundary
    db = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
    
    # Distances to nearest neighbor
    tree = cKDTree(np.column_stack((x, y)))
    dists, _ = tree.query(np.column_stack((x, y)), k=2)
    dn = dists[:, 1]
    
    # Radius is min of boundary distance and neighbor distance
    r = np.minimum(db, dn)
    return r

def _softmin(a, b, beta=50.0):
    """
    Smooth approximation of min(a, b) using log-sum-exp.
    As beta -> inf, this approaches min(a, b).
    """
    m = np.minimum(a, b)
    # Use the identity: min(a,b) = -logsumexp(-a, -b) / beta + correction
    # More stable: min(a,b) ≈ m - log(1 + exp(-beta*(a-m))) / beta - log(1 + exp(-beta*(b-m))) / beta
    # But simpler: use the fact that for large beta, min(a,b) ≈ (a*b - sqrt(a^2 + b^2 - 2*a*b*exp(-beta*|a-b|)))/2
    # Let's use a simpler smooth min:
    # min(a,b) ≈ (a + b - |a-b|)/2, but |a-b| is not smooth
    # Use: min(a,b) ≈ -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # This is stable if we subtract the max:
    max_ab = np.maximum(a, b)
    exp_a = np.exp(-beta * (a - max_ab))
    exp_b = np.exp(-beta * (b - max_ab))
    return max_ab - np.log(exp_a + exp_b) / beta

def _objective_smooth(v, W, H, beta=50.0):
    """
    Smooth objective: -sum(softmin(dist_to_boundary, dist_to_neighbor))
    """
    x = v[:N]
    y = v[N:2*N]
    
    # Distances to boundary
    db = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
    
    # Distances to nearest neighbor
    tree = cKDTree(np.column_stack((x, y)))
    dists, _ = tree.query(np.column_stack((x, y)), k=2)
    dn = dists[:, 1]
    
    # Soft-min of boundary and neighbor distances
    r_smooth = _softmin(db, dn, beta)
    
    return -np.sum(r_smooth)

def _objective(v, W, H):
    """
    Objective function to minimize: -sum(radii)
    """
    x = v[:N]
    y = v[N:2*N]
    r = _compute_radii(x, y, W, H)
    return -np.sum(r)

def _validate(x, y, W, H, tol=1e-6):
    """
    Check if the configuration is valid.
    """
    # Check boundaries
    if np.any(x < -tol) or np.any(x > W + tol):
        return False
    if np.any(y < -tol) or np.any(y > H + tol):
        return False
    
    # Check pairwise distances
    r = _compute_radii(x, y, W, H)
    
    # Check boundaries with radii
    if np.any(x - r < -tol) or np.any(x + r > W + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > H + tol):
        return False
        
    # Check pairwise
    tree = cKDTree(np.column_stack((x, y)))
    pairs = tree.query_pairs(r.max() + tol)
    for i, j in pairs:
        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
        if dist < r[i] + r[j] - tol:
            return False
    return True

def _initial_grid(W, H):
    """
    Generate a grid-like initial configuration.
    """
    cols = 5
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    k = 0
    for i in range(cols):
        for j in range(rows):
            if k >= N:
                break
            x[k] = (i + 0.5) * W / cols
            y[k] = (j + 0.5) * H / rows
            k += 1
    return np.concatenate([x, y])

def _initial_random(W, H, seed=42):
    """
    Generate a random initial configuration.
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, W, N)
    y = rng.uniform(0, H, N)
    return np.concatenate([x, y])

def _sa_optimize(v0, W, H, max_iter=3000, t0=1.0, t_end=0.001, alpha=0.9995):
    """
    Simulated Annealing to find a good local minimum.
    """
    v = v0.copy()
    curr_val = _objective(v, W, H)
    best_val = curr_val
    best_v = v.copy()
    
    T = t0
    scale = max(W, H) / 5.0  # Scale step size to problem size
    
    for i in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Adaptive step size
        step = scale * T
        idx = np.random.randint(0, 2*N)
        delta = np.random.uniform(-step, step)
        
        v_new = v.copy()
        v_new[idx] += delta
        
        # Clamp to valid box
        v_new[:N] = np.clip(v_new[:N], 0, W)
        v_new[N:] = np.clip(v_new[N:], 0, H)
        
        new_val = _objective(v_new, W, H)
        delta_val = new_val - curr_val
        
        if delta_val < 0 or np.random.rand() < np.exp(-delta_val / T):
            v = v_new
            curr_val = new_val
            if curr_val < best_val:
                best_val = curr_val
                best_v = v.copy()
                
        T *= alpha
        if T < t_end:
            break
            
    return best_v, best_val

def _polish(v, W, H, use_smooth=True, max_iter=1000):
    """
    Use L-BFGS-B to polish the solution found by SA.
    """
    bounds = []
    for i in range(N):
        bounds.append((0, W))
        bounds.append((0, H))
        
    if use_smooth:
        obj_func = lambda vv: _objective_smooth(vv, W, H, beta=50.0)
    else:
        obj_func = lambda vv: _objective(vv, W, H)
        
    try:
        res = minimize(obj_func, v, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10})
        # After smooth optimization, do a final polish with the exact objective
        res2 = minimize(_objective, res.x, method='L-BFGS-B', bounds=bounds, 
                        options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10})
        return res2.x, res2.fun
    except Exception:
        return v, _objective(v, W, H)

def _optimize_for_aspect(W, H):
    """
    Run SA + Polish for a specific aspect ratio.
    """
    best_val = np.inf
    best_v = None
    
    # Try multiple initializations
    inits = []
    inits.append(_initial_grid(W, H))
    for seed in [1, 2, 3, 4, 5]:
        inits.append(_initial_random(W, H, seed))
        
    for v0 in inits:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # SA
        v_sa, val_sa = _sa_optimize(v0, W, H)
        
        # Polish with smooth objective, then exact
        v_pol, val_pol = _polish(v_sa, W, H, use_smooth=True, max_iter=500)
        
        if val_pol < best_val:
            best_val = val_pol
            best_v = v_pol
            
    return best_v, -best_val

def construct():
    best_val = -1
    best_res = None
    
    # Focus on aspect ratios near 1:1 (square) since perimeter is fixed
    # W + H = 2, so W in [1, 2] covers all cases by symmetry
    # Try a few key aspect ratios
    widths = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15]
    
    for W in widths:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        H = 2.0 - W
        if H <= 0:
            continue
            
        v, val = _optimize_for_aspect(W, H)
        
        if v is None:
            continue
            
        x = v[:N]
        y = v[N:]
        
        # Validate
        if _validate(x, y, W, H):
            if val > best_val:
                best_val = val
                best_res = np.column_stack((x, y, _compute_radii(x, y, W, H)))
                
    if best_res is None:
        # Fallback to seed: 1x1 square, grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_res
