"""Heilbronn triangles: maximize the smallest triangle area among 11 points in an equilateral triangle."""
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
S3 = np.sqrt(3)
H = S3 / 2.0

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all triangle index combinations
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # shape (165, 3)

def bary_to_cart(bary):
    """Convert barycentric coordinates (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points given in barycentric coords."""
    # For points in barycentric coords, the area of triangle (i,j,k) is:
    # area = 0.5 * |det([u_i, v_i, 1; u_j, v_j, 1; u_k, v_k, 1])| * AREA_TRI
    # where AREA_TRI = sqrt(3)/4
    # Actually, let's compute directly from the 2D barycentric representation.
    # The area in barycentric space: for points (u,v), the triangle area is
    # 0.5 * | (u_j - u_i)(v_k - v_i) - (v_j - v_i)(u_k - u_i) |
    # This gives area in units of the reference triangle's area.
    # So actual area = this * AREA_TRI where AREA_TRI = sqrt(3)/4
    
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Extract coordinates for all triangle combinations
    ui = u[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]
    vi = v[tri_idx[:, 0]]
    vj = v[tri_idx[:, 1]]
    vk = v[tri_idx[:, 2]]
    
    # Compute cross products
    cross = (uj - ui) * (vk - vi) - (vj - vi) * (uk - ui)
    areas = 0.5 * np.abs(cross) * (S3 / 4.0)  # Multiply by area of reference triangle
    
    return areas

def clamp_bary(bary):
    """Clamp barycentric coordinates to be inside the triangle."""
    bary = bary.copy()
    # Ensure u >= 0, v >= 0, u + v <= 1
    bary[:, 0] = np.clip(bary[:, 0], 0, 1)
    bary[:, 1] = np.clip(bary[:, 1], 0, 1)
    # Fix sum constraint
    sum_uv = bary[:, 0] + bary[:, 1]
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        bary[mask, 0] *= scale
        bary[mask, 1] *= scale
    return bary

def hard_min_obj(x):
    """Objective: negative of minimum triangle area. x is (N, 2) barycentric coords."""
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(x, beta=50.0):
    """Soft-min objective using log-sum-exp for smoothness."""
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # This is smooth and approximates -min(areas) as beta -> inf
    max_neg = np.max(-beta * areas)
    log_sum = max_neg + np.log(np.sum(np.exp(-beta * areas - max_neg)))
    return log_sum / beta

def generate_hexagonal_init():
    """Generate a hexagonal packing-inspired initialization."""
    # Start with a symmetric pattern
    # Place points in a hexagonal grid pattern inside the triangle
    pts_bary = []
    
    # Center point
    pts_bary.append([1/3, 1/3])
    
    # Ring of 6 points around center
    r = 0.15
    for i in range(6):
        angle = i * np.pi / 3
        # In barycentric, we need to convert from a local coordinate system
        # This is approximate; we'll refine later
        u = 1/3 + r * np.cos(angle) * 0.8
        v = 1/3 + r * np.sin(angle) * 0.8
        pts_bary.append([u, v])
    
    # Add more points to reach 11
    random_seeds = [0.3, 0.7, 0.5, 0.2, 0.8]
    for i in range(5):
        u = random_seeds[i] * 0.6 + 0.2
        v = random_seeds[(i+2) % 5] * 0.6 + 0.2
        pts_bary.append([u, v])
    
    pts_bary = np.array(pts_bary[:N])
    return clamp_bary(pts_bary)

def generate_random_init(seed):
    """Generate random initialization inside the triangle."""
    rng = np.random.RandomState(seed)
    pts_bary = []
    while len(pts_bary) < N:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            pts_bary.append([u, v])
    return np.array(pts_bary)

def generate_perturbed_init(base, seed, scale=0.05):
    """Generate perturbed version of a base initialization."""
    rng = np.random.RandomState(seed)
    perturbed = base.copy()
    perturbed += rng.normal(0, scale, size=perturbed.shape)
    return clamp_bary(perturbed)

def refine_local(x0, beta_start=20.0, beta_end=200.0, n_stages=5, max_iter=500):
    """Multi-stage local refinement with increasing beta."""
    current = x0.copy()
    
    for stage in range(n_stages):
        # Interpolate beta
        t = stage / max(1, n_stages - 1)
        beta = beta_start + t * (beta_end - beta_start)
        
        def obj(x):
            return soft_min_obj(x, beta=beta)
        
        try:
            res = minimize(
                obj,
                current,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if res.success or res.fun < obj(current):
                current = res.x
        except Exception:
            pass
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_bary = None
    best_score = -float('inf')
    
    def update_best(bary):
        nonlocal best_bary, best_score
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        score = np.min(areas)
        if score > best_score:
            best_score = score
            best_bary = bary.copy()
    
    # Generate initializations
    inits = []
    inits.append(generate_hexagonal_init())
    for seed in range(10):
        inits.append(generate_random_init(seed))
    
    # Add perturbed versions of the hex init
    for i in range(5):
        inits.append(generate_perturbed_init(inits[0], seed=i, scale=0.03))
    
    # Evaluate all inits
    for init in inits:
        update_best(init)
    
    # Local refinement from top candidates
    # Sort inits by their current best score
    init_scores = []
    for init in inits:
        areas = calc_areas_bary(init)
        init_scores.append(np.min(areas))
    
    order = np.argsort(init_scores)[::-1]  # Best first
    
    # Refine the top 5 initializations
    for idx in order[:5]:
        if time.time() > deadline - 5.0:
            break
        
        refined = refine_local(inits[idx])
        update_best(refined)
    
    # Try a few more random starts with refinement if time permits
    for seed in range(10, 20):
        if time.time() > deadline - 10.0:
            break
        
        init = generate_random_init(seed)
        refined = refine_local(init, beta_start=20.0, beta_end=100.0, n_stages=3, max_iter=200)
        update_best(refined)
    
    # Final high-precision polish from the best solution
    if best_bary is not None and time.time() < deadline - 5.0:
        # Very tight polish with high beta
        def final_obj(x):
            return soft_min_obj(x, beta=500.0)
        
        try:
            res = minimize(
                final_obj,
                best_bary,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
            )
            if res.fun < final_obj(best_bary):
                update_best(res.x)
        except Exception:
            pass
    
    # Convert best barycentric to Cartesian
    if best_bary is None:
        best_bary = generate_random_init(0)
    
    best_cart = bary_to_cart(best_bary)
    
    return best_cart

# EVOLVE-BLOCK-END
