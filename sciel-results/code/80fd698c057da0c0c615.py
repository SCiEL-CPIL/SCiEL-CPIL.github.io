import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric coords (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    u = 1.0 - x * (2.0 / 3.0) - y * (2.0 / (3.0 * H))
    v = 1.0 - u - y / H
    return np.column_stack([u, v])

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

def get_min_area(bary):
    """Get the minimum area for a given barycentric configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def get_active_triangles(bary, threshold=1e-6):
    """Return indices of triangles whose area is within threshold of the minimum."""
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    active = np.where(areas <= min_area + threshold)[0]
    return active

def soft_min_obj(x, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    a_max = np.max(areas)
    val = a_max - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (areas - a_max))))
    return -val

def coordinated_ascent(bary, max_iter=200, tol=1e-10):
    """
    Coordinated ascent: iteratively identify active (minimum-area) triangles,
    then perform a small local optimization that maximizes the minimum area
    among only those active triangles. This avoids the gradient dilution
    problem of soft-min over all 165 triangles.
    """
    current = clamp_bary(bary.copy())
    current_min = get_min_area(current)
    
    for it in range(max_iter):
        if time.time() > TIME_LIMIT - 5:
            break
            
        areas = calc_areas_bary(current)
        min_area = np.min(areas)
        active = np.where(areas <= min_area + 1e-6)[0]
        
        if len(active) == 0:
            active = np.array([np.argmin(areas)])
        
        # Build a sub-objective: maximize the min area among active triangles
        # Use soft-min over just the active triangles
        active_areas = areas[active]
        n_active = len(active_areas)
        
        # Objective: -soft_min(active_areas)
        # We'll use a high beta for precision
        beta = 200.0
        
        def local_obj(x):
            b = clamp_bary(x.reshape(N, 2))
            a = calc_areas_bary(b)
            aa = a[active]
            amax = np.max(aa)
            sm = amax - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (aa - amax))))
            return -sm
        
        x0 = current.flatten()
        
        try:
            res = minimize(local_obj, x0, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.success or res.fun < local_obj(x0):
                new_bary = clamp_bary(res.x.reshape(N, 2))
                new_min = get_min_area(new_bary)
                if new_min > current_min + tol:
                    current = new_bary
                    current_min = new_min
                else:
                    break
            else:
                break
        except Exception:
            break
    
    return current

def generate_symmetric_init():
    """Generate a symmetric initialization based on hexagonal packing."""
    # Place points in a pattern that respects the 3-fold symmetry of the triangle
    # Center + 3 rings
    pts = []
    
    # Center
    pts.append([1/3, 1/3])
    
    # Inner ring: 3 points at 120 degrees
    r1 = 0.15
    for k in range(3):
        angle = k * 2 * np.pi / 3 - np.pi / 2
        u = 1/3 + r1 * np.cos(angle)
        v = 1/3 + r1 * np.sin(angle) * (2/np.sqrt(3))
        pts.append([u, v])
    
    # Outer ring: 6 points
    r2 = 0.28
    for k in range(6):
        angle = k * 2 * np.pi / 6 - np.pi / 2
        u = 1/3 + r2 * np.cos(angle)
        v = 1/3 + r2 * np.sin(angle) * (2/np.sqrt(3))
        pts.append([u, v])
    
    return np.array(pts[:N])

def generate_radial_init():
    """Generate a radial initialization with points spread from center to edges."""
    pts = []
    # Center
    pts.append([1/3, 1/3])
    
    # Ring 1: 3 points
    r1 = 0.12
    for k in range(3):
        angle = k * 2 * np.pi / 3
        u = 1/3 + r1 * np.cos(angle)
        v = 1/3 + r1 * np.sin(angle) * (2/np.sqrt(3))
        pts.append([u, v])
    
    # Ring 2: 7 points
    r2 = 0.25
    for k in range(7):
        angle = k * 2 * np.pi / 7
        u = 1/3 + r2 * np.cos(angle)
        v = 1/3 + r2 * np.sin(angle) * (2/np.sqrt(3))
        pts.append([u, v])
    
    return np.array(pts[:N])

def generate_boundary_init():
    """Generate initialization with some points near the boundary."""
    pts = []
    # Vertices (slightly inset)
    eps = 0.05
    pts.append([eps, eps])
    pts.append([1 - eps, eps])
    pts.append([1/2, H - eps])
    
    # Edge midpoints (slightly inset)
    pts.append([0.5, eps])
    pts.append([0.5, H/2])
    pts.append([0.25, H/4])
    
    # Interior points
    pts.append([1/3, 1/3])
    pts.append([0.4, 0.3])
    pts.append([0.6, 0.3])
    pts.append([0.5, 0.2])
    
    return np.array(pts[:N])

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initialization."""
    pts = []
    # Use a grid in barycentric space
    grid_size = 10
    us = np.linspace(0.1, 0.8, grid_size)
    vs = np.linspace(0.1, 0.8, grid_size)
    
    candidates = []
    for u in us:
        for v in vs:
            if u + v <= 0.9:
                candidates.append([u, v])
                
    if len(candidates) < N:
        return None
        
    # We want to select 11 points that are well spread.
    # Just picking a subset of the grid.
    # To be deterministic but diverse, we can use a fixed seed for selection.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(candidates), N, replace=False)
    return np.array(candidates)[idx]

def generate_random_init(rng):
    """Generate a random initialization inside the triangle."""
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(bary, rng, sigma=0.05):
    """Generate a perturbed version of an existing configuration."""
    pert = rng.normal(0, sigma, size=(N, 2))
    return clamp_bary(bary + pert)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Initialization - diverse starting points
    start_points = []
    
    sym_init = generate_symmetric_init()
    if sym_init is not None:
        start_points.append(sym_init)
        
    rad_init = generate_radial_init()
    if rad_init is not None:
        start_points.append(rad_init)
        
    bound_init = generate_boundary_init()
    if bound_init is not None:
        start_points.append(bound_init)
        
    hex_init = generate_hexagonal_init()
    if hex_init is not None:
        start_points.append(hex_init)
        
    rng_master = np.random.default_rng(123)
    for _ in range(3):
        start_points.append(generate_random_init(rng_master))
        
    # 2. Local Refinement using coordinated ascent
    for sp in start_points:
        if time.time() > deadline - 15:
            break
        refined_bary = coordinated_ascent(sp)
        update_best(refined_bary)
        
    # 3. Iterative Perturbation and Coordinated Ascent
    current_best = best_bary.copy() if best_bary is not None else start_points[0]
    
    for iteration in range(30):
        if time.time() > deadline - 10:
            break
            
        # Try several perturbations of the current best
        for _ in range(8):
            if time.time() > deadline - 5:
                break
                
            # Decrease perturbation size over time
            sigma = 0.08 / (1.0 + 0.05 * iteration)
            perturbed = generate_perturbed_init(current_best, rng_master, sigma)
            refined = coordinated_ascent(perturbed)
            
            if update_best(refined):
                current_best = best_bary.copy()
                break  # If we found a better point, restart perturbation loop
        
    # 4. Final Fallback
    if best_bary is None:
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    cart = bary_to_cart(best_bary)
    
    return cart
