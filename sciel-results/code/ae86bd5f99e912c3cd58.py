import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0  # Safety margin for 120s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def coulomb_obj(bary_flat):
    """
    Smooth surrogate: Minimize sum of inverse squared distances.
    This promotes spreading points apart, related to energy minimization/OT.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    # Calculate pairwise squared distances in barycentric space 
    # (Metric is not Euclidean in barycentric, but distance ordering is preserved enough for initialization)
    # Better: Convert to Cartesian for distance calculation to be physically meaningful
    cart = bary_to_cart(bary)
    
    # Compute pairwise squared distances
    # cart shape (N, 2)
    dists_sq = np.sum((cart[:, np.newaxis, :] - cart[np.newaxis, :, :]) ** 2, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists_sq, 1.0)
    
    # Sum of inverse squared distances (Coulomb-like)
    # Add small epsilon to avoid division by zero
    energy = np.sum(1.0 / (dists_sq + 1e-12))
    
    return energy

def refine_smooth(bary_init, time_limit=2.0):
    """
    Stage 1: Minimize Coulomb energy using L-BFGS-B.
    This is fast and finds a well-spread configuration.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    try:
        result = minimize(
            coulomb_obj,
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        bary_smooth = clamp_bary(result.x.reshape(N, 2))
    except Exception:
        bary_smooth = bary_init
        
    return bary_smooth

def refine_hard(bary_init, time_limit=3.0):
    """
    Stage 2: Polish using Nelder-Mead on the hard max-min area objective.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    try:
        # Nelder-Mead is derivative-free and handles non-smooth objectives well
        # We use a tight time limit
        result = minimize(
            hard_min_obj,
            bary_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        bary_hard = clamp_bary(result.x.reshape(N, 2))
    except Exception:
        bary_hard = bary_init
        
    return bary_hard

def generate_hexagonal_init(scale=None, seed=42):
    """
    Generate a hexagonal lattice-like initial configuration at a given scale.
    """
    if scale is None:
        scale = 0.25
    
    pts = []
    max_k = int(1.0 / scale) + 1
    for k in range(max_k + 1):
        w_val = k * scale
        if w_val > 1.0:
            break
        num_pts_in_row = int((1 - w_val) / scale) + 1
        for j in range(num_pts_in_row):
            u_val = j * scale
            v_val = (1 - w_val) - u_val
            if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                pts.append([u_val, v_val])
    
    if len(pts) < N:
        # Fallback to random if not enough points
        rng = np.random.default_rng(seed)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        return np.column_stack([u, v])
    
    # Select N points from the lattice
    rng = np.random.default_rng(seed)
    if len(pts) == N:
        bary = np.array(pts)
    else:
        idx = rng.choice(len(pts), N, replace=False)
        bary = np.array(pts)[idx]
    
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate diverse structured initializations
    inits = []
    
    # Hexagonal at multiple scales
    for scale in [0.2, 0.25, 0.3, 0.35, 0.4]:
        inits.append(generate_hexagonal_init(scale=scale, seed=42 + int(scale * 100)))
    
    # Random
    rng = np.random.default_rng(300)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    inits.append(np.column_stack([u, v]))
    
    # Refine each initialization using two-stage approach
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        # Stage 1: Smooth spread (Coulomb)
        bary_smooth = refine_smooth(init, time_limit=1.0)
        
        # Stage 2: Hard polish (Max-Min Area)
        bary_hard = refine_hard(bary_smooth, time_limit=2.0)
        
        update_best(bary_hard)

    # Multi-start with perturbations of best
    restart_count = 0
    while time.time() < deadline - 4:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            rng = np.random.default_rng(5000 + restart_count)
            noise = rng.normal(0, 0.03, best_bary.shape)
            init = clamp_bary(best_bary + noise)
        else:
            # Random restart
            rng = np.random.default_rng(6000 + restart_count)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            init = np.column_stack([u, v])
        
        remaining = deadline - 4 - time.time()
        if remaining < 3:
            break
            
        # Two-stage refinement for restarts
        bary_smooth = refine_smooth(init, time_limit=1.0)
        bary_hard = refine_hard(bary_smooth, time_limit=1.5)
        update_best(bary_hard)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 2:
        bary_final = refine_hard(best_bary, time_limit=2.0)
        update_best(bary_final)

    # Convert to cartesian
    if best_bary is None:
        # Fallback
        best_bary = inits[0]
        
    cart = bary_to_cart(best_bary)
    
    return cart
