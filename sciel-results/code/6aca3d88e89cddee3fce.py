import numpy as np
import scipy.optimize as opt
import time
import itertools

N = 11
# Vertices of the unit-area equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

# Pre-compute all triangle index combinations
tri_indices = list(itertools.combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * (np.sqrt(3) / 2.0)
    return np.column_stack([x, y])

def calc_areas_bary(ba):
    """
    Compute all triangle areas from barycentric coordinates.
    ba: (N, 2) array of (u, v)
    Returns: (165,) array of triangle areas
    """
    # Area of triangle with barycentric coords:
    # For points p_i, p_j, p_k with barycentric (u,v),
    # the area is proportional to |det of the barycentric matrix|
    # Actually, let's compute in Cartesian for clarity
    pts = bary_to_cart(ba)
    areas = np.zeros(len(tri_idx))
    for idx, (i, j, k) in enumerate(tri_idx):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        areas[idx] = 0.5 * abs(cross)
    return areas

def clamp_bary(ba):
    """Clamp barycentric coordinates to valid range [0,1] with u+v<=1."""
    u = np.clip(ba[:, 0], 0, 1)
    v = np.clip(ba[:, 1], 0, 1)
    s = u + v
    mask = s > 1
    if np.any(mask):
        scale = 1.0 / s[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])

def soft_min_obj(x, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for soft-min of triangle areas.
    We want to maximize the minimum area, so minimize -soft_min(areas).
    soft_min(areas) = -(1/beta) * log(sum(exp(-beta * area)))
    """
    ba = x.reshape(N, 2)
    ba = clamp_bary(ba)
    areas = calc_areas_bary(ba)
    
    # Log-sum-exp for numerical stability
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a)))) + (-beta * min_a)
    soft_min_val = -log_sum / beta
    
    # Penalty for being outside (should be 0 after clamp, but just in case)
    u = ba[:, 0]
    v = ba[:, 1]
    penalty = np.sum(np.maximum(0, -u)**2) + np.sum(np.maximum(0, -v)**2) + np.sum(np.maximum(0, u + v - 1)**2)
    
    return -soft_min_val + 1000.0 * penalty

def hard_min_obj(x):
    """Hard minimum area objective."""
    ba = x.reshape(N, 2)
    ba = clamp_bary(ba)
    areas = calc_areas_bary(ba)
    return -np.min(areas)

def generate_hexagonal_init(rng):
    """Generate a hexagonal/triangular lattice initialization."""
    ba = np.zeros((N, 2))
    # Triangular lattice pattern for 11 points
    # Row 0: 1 point
    # Row 1: 2 points  
    # Row 2: 3 points
    # Row 3: 3 points
    # Row 4: 2 points
    # Total: 11
    rows = [1, 2, 3, 3, 2]
    point_idx = 0
    for r, count in enumerate(rows):
        for c in range(count):
            # Barycentric coordinates for triangular lattice
            # u is "vertical" position, v is "horizontal" offset
            u = (r + 0.5) / 5.0  # distribute vertically
            # Distribute horizontally within each row
            if count == 1:
                v = 0.5
            else:
                v = (c + 0.5) / count * (1.0 - u)  # scale by available width
            ba[point_idx] = [u, v]
            point_idx += 1
    # Add small noise
    ba += rng.normal(0, 0.02, size=ba.shape)
    return ba.flatten()

def generate_random_init(rng):
    """Generate random initialization."""
    ba = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                ba[i] = [u, v]
                break
    return ba.flatten()

def generate_perturbed_init(best_ba, rng, sigma=0.05):
    """Generate perturbed version of best solution."""
    ba = best_ba + rng.normal(0, sigma, size=best_ba.shape)
    return ba.flatten()

def refine_local(x0, deadline, rng):
    """Refine a starting point using Nelder-Mead on hard objective."""
    start_time = time.time()
    res = opt.minimize(hard_min_obj, x0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    if time.time() > deadline:
        return x0
    return res.x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = -np.inf
    best_points = None
    best_ba = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Global search with soft surrogate
    n_global = 15
    for i in range(n_global):
        if time.time() > deadline:
            break
        
        if i == 0:
            x0 = generate_hexagonal_init(rng)
        elif i % 3 == 0:
            x0 = generate_hexagonal_init(rng)
        else:
            x0 = generate_random_init(rng)
        
        try:
            # Use soft surrogate for global exploration
            res = opt.minimize(soft_min_obj, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'beta': 50.0})
            
            if time.time() > deadline:
                break
            
            x_opt = res.x
            ba_opt = clamp_bary(x_opt.reshape(N, 2))
            points = bary_to_cart(ba_opt)
            score = -hard_min_obj(x_opt)
            
            if score > best_score:
                best_score = score
                best_points = points
                best_ba = ba_opt.flatten()
                
        except Exception as e:
            pass
    
    # Phase 2: Local refinement on hard objective
    if best_ba is not None:
        n_local = 8
        for i in range(n_local):
            if time.time() > deadline:
                break
            
            if i == 0:
                x0 = best_ba.copy()
            else:
                x0 = generate_perturbed_init(best_ba, rng, sigma=0.03)
            
            try:
                x_opt = refine_local(x0, deadline, rng)
                ba_opt = clamp_bary(x_opt.reshape(N, 2))
                points = bary_to_cart(ba_opt)
                score = -hard_min_obj(x_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = points
                    best_ba = ba_opt.flatten()
                    
            except Exception as e:
                pass
    
    # Phase 3: Final polish with tighter tolerances
    if best_ba is not None and time.time() < deadline - 5.0:
        try:
            res = opt.minimize(hard_min_obj, best_ba, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            x_opt = res.x
            ba_opt = clamp_bary(x_opt.reshape(N, 2))
            points = bary_to_cart(ba_opt)
            score = -hard_min_obj(x_opt)
            
            if score > best_score:
                best_score = score
                best_points = points
                best_ba = ba_opt.flatten()
        except Exception as e:
            pass
    
    if best_points is None:
        # Fallback: random points
        ba = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v <= 1:
                    ba[i] = [u, v]
                    break
        best_points = bary_to_cart(ba)
    
    return best_points
