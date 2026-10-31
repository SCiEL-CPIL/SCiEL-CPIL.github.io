import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def get_free_indices():
    # Index 0 and 1 are fixed
    return list(range(2, N))

def reconstruct_points(x_free):
    # x_free shape (28,)
    pts = np.zeros((N, D))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    idxs = get_free_indices()
    for i, idx in enumerate(idxs):
        pts[idx, 0] = x_free[i*2]
        pts[idx, 1] = x_free[i*2+1]
    return pts

def compute_objective_and_grad_full(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    mask = ~np.eye(N, dtype=bool)
    valid_dists = dists_sq[mask]
    
    if len(valid_dists) == 0:
        return np.inf, np.zeros_like(pts)
        
    m = np.min(valid_dists)
    M = np.max(valid_dists)
    
    if m == 0:
        return np.inf, np.zeros_like(pts)
        
    val = M / m
    
    tol = 1e-6
    max_mask = (dists_sq > M - tol) & mask
    min_mask = (dists_sq < m + tol) & mask
    
    grad_M = 2 * np.sum(max_mask[:, :, np.newaxis] * diff, axis=1)
    grad_m = 2 * np.sum(min_mask[:, :, np.newaxis] * diff, axis=1)
    
    grad_val = (m * grad_M - M * grad_m) / (m * m)
    return val, grad_val

def objective_and_gradient(x_free):
    pts = reconstruct_points(x_free)
    val, grad_full = compute_objective_and_grad_full(pts)
    
    # Extract gradient for free variables
    idxs = get_free_indices()
    grad_free = np.zeros_like(x_free)
    for i, idx in enumerate(idxs):
        grad_free[i*2] = grad_full[idx, 0]
        grad_free[i*2+1] = grad_full[idx, 1]
        
    return val, grad_free

def fun(x_free):
    val, _ = objective_and_gradient(x_free)
    return val

def jac(x_free):
    _, grad = objective_and_gradient(x_free)
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_val = np.inf
    best_x_free = None
    
    # Initializations for free variables
    # We need to generate full points, then extract free vars
    
    inits_full = []
    
    # 1. Grid
    side = int(np.ceil(N ** (1 / D)))
    grid = list(np.meshgrid(np.arange(side), np.arange(side)))
    grid_pts = np.array([p for p in zip(grid[0].flatten(), grid[1].flatten())])[:N]
    # Normalize: move center to 0, scale so min dist is 1?
    # Actually, we fix x0=(0,0), x1=(1,0).
    # We need to transform the grid to match this.
    # This is complex. 
    # Alternative: Just use the grid as is, and let the optimizer figure it out?
    # No, the fixed points break the symmetry.
    # Better: Generate inits in the "fixed frame".
    # For grid: 
    # Identify two points that are distance 1 apart.
    # Set them to (0,0) and (1,0).
    # Rotate/translate the rest.
    
    def normalize_to_fixed_frame(pts):
        # Find pair with min distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        p_i = pts[i]
        p_j = pts[j]
        
        # Vector from i to j
        v = p_j - p_i
        norm = np.linalg.norm(v)
        
        # New coordinates
        new_pts = np.zeros_like(pts)
        # i becomes (0,0)
        new_pts[i] = [0, 0]
        # j becomes (1,0)
        new_pts[j] = [1, 0]
        
        # Rotation matrix
        # v_norm = v / norm
        # u = [1, 0]
        # We want to map v to u.
        # In 2D, this is a rotation by -angle(v).
        angle = np.arctan2(v[1], v[0])
        c = np.cos(-angle)
        s = np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        
        # Translate and rotate all points
        # p' = R * (p - p_i) / norm
        for k in range(N):
            if k == i:
                continue
            if k == j:
                continue
            p_rel = pts[k] - p_i
            p_rot = R @ p_rel
            p_scaled = p_rot / norm
            new_pts[k] = p_scaled
            
        return new_pts

    # Generate inits
    # Grid
    side = 4
    grid_pts = np.array([(i, j) for i in range(side) for j in range(side)])[:N].astype(float)
    inits_full.append(normalize_to_fixed_frame(grid_pts))
    
    # Hexagonal
    a = np.array([1, 0])
    b = np.array([0.5, np.sqrt(3)/2])
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * a + j * b
            pts.append(p)
    pts = np.array(pts)
    d_to_origin = np.sum(pts**2, axis=1)
    idx = np.argsort(d_to_origin)[:N]
    hex_pts = pts[idx]
    inits_full.append(normalize_to_fixed_frame(hex_pts))
    
    # Random
    np.random.seed(42)
    rand_pts = np.random.rand(N, D) * 2 - 1
    inits_full.append(normalize_to_fixed_frame(rand_pts))
    
    # Run optimization
    for init_full in inits_full:
        if time.time() > deadline:
            break
        
        # Extract free vars
        idxs = get_free_indices()
        x0 = np.zeros(len(idxs)*2)
        for i, idx in enumerate(idxs):
            x0[i*2] = init_full[idx, 0]
            x0[i*2+1] = init_full[idx, 1]
            
        try:
            res = minimize(
                fun,
                x0,
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x_free = res.x
        except Exception:
            pass
        
        if time.time() > deadline:
            break

    # Random restarts
    for i in range(20):
        if time.time() > deadline:
            break
        np.random.seed(100 + i)
        rand_pts = np.random.rand(N, D) * 2 - 1
        init_full = normalize_to_fixed_frame(rand_pts)
        
        idxs = get_free_indices()
        x0 = np.zeros(len(idxs)*2)
        for i, idx in enumerate(idxs):
            x0[i*2] = init_full[idx, 0]
            x0[i*2+1] = init_full[idx, 1]
            
        try:
            res = minimize(
                fun,
                x0,
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x_free = res.x
        except Exception:
            pass

    if best_x_free is None:
        # Fallback
        return inits_full[0]
        
    return reconstruct_points(best_x_free)
