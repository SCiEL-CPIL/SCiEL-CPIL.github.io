import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 140.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
NUM_TRIS = len(tri_indices)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_TOTAL = np.sqrt(3) / 4.0

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary_vec(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords.
    Returns array of shape (NUM_TRIS,) with signed determinants (areas)."""
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
    
    return det

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

def nlp_objective(x):
    """Objective for NLP: minimize -t where x = [bary(N*2), t]."""
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    return -t

def nlp_constraint_areas(x):
    """Constraint: all triangle areas >= t."""
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    areas = calc_areas_bary_vec(bary)
    return areas - t

def nlp_constraint_bary(x):
    """Constraints: u >= 0, v >= 0, u+v <= 1 for each point."""
    bary = x[:N*2].reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    return np.concatenate([u, v, w])

def get_min_area(bary):
    """Return the minimum triangle area for a configuration."""
    areas = calc_areas_bary_vec(bary)
    return np.min(np.abs(areas))

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary_vec(bary)
    return -np.min(np.abs(areas))

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
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
        
        if len(pts) >= N:
            # Try different subsets
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def solve_nlp(bary_init, time_limit=10.0):
    """
    Solve the NLP formulation: maximize t subject to all triangle areas >= t.
    Uses SLSQP with explicit constraints.
    """
    start_time = time.time()
    
    bary = clamp_bary(bary_init)
    min_area = get_min_area(bary)
    
    # x = [bary.flatten(), t]
    x0 = np.concatenate([bary.flatten(), [min_area]])
    
    # Bounds: bary coords in [0,1], t in [0, 0.5]
    bounds = [(0, 1)] * (N * 2) + [(0, 0.5)]
    
    # Define constraints
    constraints = []
    
    # Area constraints: areas[i] - t >= 0 for all i
    # We'll use a single constraint function returning all 165 values
    def area_constraints(x):
        bary_x = x[:N*2].reshape(N, 2)
        t_x = x[N*2]
        areas = calc_areas_bary_vec(bary_x)
        return areas - t_x
    
    constraints.append({'type': 'ineq', 'fun': area_constraints})
    
    # Barycentric constraints: u >= 0, v >= 0, w = 1-u-v >= 0
    # These are handled by bounds on u, v and we need u+v <= 1
    # Actually, with bounds [0,1] on u and v, we still need u+v <= 1
    def bary_sum_constraint(x):
        bary_x = x[:N*2].reshape(N, 2)
        u = bary_x[:, 0]
        v = bary_x[:, 1]
        return 1.0 - u - v
    
    constraints.append({'type': 'ineq', 'fun': bary_sum_constraint})
    
    try:
        result = minimize(
            nlp_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        
        if time.time() - start_time > time_limit:
            return bary_init, min_area
        
        x_opt = result.x
        bary_opt = clamp_bary(x_opt[:N*2].reshape(N, 2))
        t_opt = x_opt[N*2]
        
        # Verify and get actual min area
        actual_min = get_min_area(bary_opt)
        
        return bary_opt, actual_min
        
    except Exception as e:
        return bary_init, min_area

def generate_structured_inits():
    """
    Generate diverse structured initial configurations inspired by known Heilbronn results.
    For n=11, use boundary-heavy arrangements with interior points.
    """
    candidates = []
    
    # Config 1: 3 corners, 4 edge quarters, 4 interior
    cfg1 = np.array([
        [0.0, 0.0],   # corner A
        [1.0, 0.0],   # corner B
        [0.0, 1.0],   # corner C
        [0.25, 0.0],  # edge AB quarter
        [0.75, 0.0],  # edge AB three-quarter
        [0.0, 0.25],  # edge AC quarter
        [0.0, 0.75],  # edge AC three-quarter
        [0.25, 0.75], # edge BC
        [0.75, 0.25], # edge BC
        [1/3, 1/3],   # centroid
        [0.3, 0.3],   # interior
    ])
    candidates.append(clamp_bary(cfg1))
    
    # Config 2: 3 corners, 3 edge midpoints, 5 interior (symmetric)
    cfg2 = np.array([
        [0.0, 0.0],   # corner
        [1.0, 0.0],   # corner
        [0.0, 1.0],   # corner
        [0.5, 0.0],   # edge midpoint AB
        [0.0, 0.5],   # edge midpoint AC
        [0.5, 0.5],   # edge midpoint BC
        [1/3, 1/3],   # centroid
        [0.2, 0.3],   # interior
        [0.5, 0.2],   # interior
        [0.3, 0.5],   # interior
        [0.4, 0.4],   # interior
    ])
    candidates.append(clamp_bary(cfg2))
    
    # Config 3: Hexagonal-inspired
    cfg3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.33, 0.0],
        [0.67, 0.0],
        [0.0, 0.33],
        [0.0, 0.67],
        [0.17, 0.5],
        [0.5, 0.17],
        [0.33, 0.33],
        [0.5, 0.33],
    ])
    candidates.append(clamp_bary(cfg3))
    
    # Config 4: Dense boundary
    cfg4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.0],
        [0.8, 0.0],
        [0.0, 0.2],
        [0.0, 0.8],
        [0.1, 0.9],
        [0.9, 0.1],
        [1/3, 1/3],
        [0.4, 0.35],
    ])
    candidates.append(clamp_bary(cfg4))
    
    # Config 5: Perturbed hexagonal
    rng = np.random.default_rng(42)
    base_hex = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.5],
        [0.5, 0.25],
        [1/3, 1/3],
        [0.4, 0.35],
    ])
    for _ in range(3):
        perturbed = base_hex + rng.normal(0, 0.03, size=base_hex.shape)
        candidates.append(clamp_bary(perturbed))
    
    # Config 6: Symmetric 3-fold with interior ring
    cfg6 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.4, 0.0],
        [0.6, 0.0],
        [0.0, 0.4],
        [0.0, 0.6],
        [0.3, 0.3],
        [0.45, 0.15],
        [0.15, 0.45],
        [0.33, 0.33],
    ])
    candidates.append(clamp_bary(cfg6))
    
    return candidates

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def solve_nlp_multi(bary_init, time_limit=15.0):
    """
    Try multiple NLP solves with different initializations and return best.
    """
    start_time = time.time()
    best_bary = bary_init.copy()
    best_val = get_min_area(bary_init)
    
    # Try the direct NLP
    if time.time() - start_time < time_limit * 0.4:
        bary1, val1 = solve_nlp(bary_init, time_limit=time_limit * 0.35)
        if val1 > best_val:
            best_bary = bary1
            best_val = val1
    
    # Try NLP from perturbed versions
    for sigma in [0.03, 0.06, 0.1]:
        if time.time() - start_time > time_limit * 0.8:
            break
        remaining = time_limit * 0.8 - (time.time() - start_time)
        if remaining < 1.0:
            break
        perturbed = generate_perturbed_init(bary_init, sigma=sigma, seed=int(sigma * 1000))
        bary_p, val_p = solve_nlp(perturbed, time_limit=remaining * 0.8)
        if val_p > best_val:
            best_bary = bary_p
            best_val = val_p
    
    return best_bary, best_val

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using NLP formulation.
    """
    start_time = time.time()
    
    if time_limit < 3.0:
        return bary_init
    
    bary_current, val_current = solve_nlp_multi(bary_init, time_limit=time_limit)
    
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Try Hexagonal Init with NLP
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=12.0)
    update_best(refined_hex)
    
    # 2. Multi-start: structured + random restarts with NLP refinement
    rng = np.random.default_rng(123)
    structured = generate_structured_inits()
    
    # Add random inits
    for i in range(5):
        structured.append(generate_random_init(seed=1000 + i))
    
    restart_count = 0
    for cand in structured:
        if time.time() > deadline - 20:
            break
        
        remaining_time = deadline - 20 - time.time()
        if remaining_time < 5:
            break
            
        try:
            refined = refine_local(cand, time_limit=min(10.0, remaining_time * 0.8))
            update_best(refined)
        except Exception:
            pass
        
        restart_count += 1
        
        # Try perturbed versions of current best
        if best_bary is not None and time.time() < deadline - 20:
            for sigma in [0.03, 0.07]:
                if time.time() > deadline - 20:
                    break
                perturbed = generate_perturbed_init(best_bary, sigma=sigma, seed=3000 + restart_count * 10 + int(sigma * 100))
                try:
                    refined_p = refine_local(perturbed, time_limit=min(8.0, deadline - 20 - time.time()))
                    update_best(refined_p)
                except Exception:
                    pass
    
    # Final NLP polish on the best candidate
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bary_final, val_final = solve_nlp_multi(best_bary, time_limit=min(10.0, deadline - 8 - time.time()))
            update_best(bary_final)
        except Exception:
            pass
    
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
