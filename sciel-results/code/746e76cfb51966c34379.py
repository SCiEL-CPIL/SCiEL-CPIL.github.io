"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
N = 11
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from itertools import combinations

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])
TRI_VERTS = np.array([A, B, C])

# Barycentric helpers
def cart_to_bary(p):
    """Convert Cartesian point(s) to barycentric coordinates relative to triangle ABC."""
    p = np.atleast_2d(p)
    A2, B2, C2 = TRI_VERTS[0], TRI_VERTS[1], TRI_VERTS[2]
    v0 = B2 - A2
    v1 = C2 - A2
    v2 = p - A2
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    den = d00 * d11 - d01 * d01
    u = (d11 * d20 - d01 * d21) / den
    v = (d00 * d21 - d01 * d20) / den
    w = 1.0 - u - v
    return np.stack([u, v, w], axis=-1)

def bary_to_cart(b):
    """Convert barycentric coordinate(s) to Cartesian."""
    b = np.atleast_2d(b)
    return b[:, 0:1] * TRI_VERTS[0] + b[:, 1:2] * TRI_VERTS[1] + b[:, 2:3] * TRI_VERTS[2]

def clamp_bary(b, eps=1e-6):
    """Clamp barycentric coords to be valid (non-negative, sum to 1)."""
    b = np.atleast_2d(b).copy()
    b = np.clip(b, eps, 1.0 - 2 * eps)
    b /= b.sum(axis=-1, keepdims=True)
    return b

def min_triangle_area_cart(pts):
    """Compute minimum triangle area among all C(n,3) triangles."""
    n = len(pts)
    min_a = float('inf')
    for i, j, k in combinations(range(n), 3):
        area = 0.5 * abs(np.cross(pts[j] - pts[i], pts[k] - pts[i]))
        if area < min_a:
            min_a = area
    return min_a

def min_triangle_area_bary(bary_pts):
    """Compute minimum triangle area using barycentric coords (area in barycentric space * triangle area)."""
    # Area of triangle with barycentric coords is proportional to |det| of the 3x3 matrix
    # The actual area = |det| * (area of reference triangle)
    # Area of reference triangle = sqrt(3)/4
    n = len(bary_pts)
    ref_area = np.sqrt(3) / 4
    min_a = float('inf')
    for i, j, k in combinations(range(n), 3):
        M = np.array([bary_pts[i], bary_pts[j], bary_pts[k]])
        det = np.linalg.det(M)
        area = abs(det) * ref_area
        if area < min_a:
            min_a = area
    return min_a

def delaunay_triangles(pts):
    """Get Delaunay triangulation indices for points."""
    try:
        tri = Delaunay(pts)
        return tri.simplices
    except:
        return None

def smooth_min_obj(bary_flat, weight=100.0, var_weight=0.1):
    """
    Smooth surrogate objective: minimize -soft_min of triangle areas.
    Uses log-sum-exp approximation for the minimum.
    """
    b = bary_flat.reshape(-1, 3)
    b = clamp_bary(b)
    pts = bary_to_cart(b)
    
    n = len(pts)
    areas = []
    for i, j, k in combinations(range(n), 3):
        area = 0.5 * abs(np.cross(pts[j] - pts[i], pts[k] - pts[i]))
        areas.append(area)
    areas = np.array(areas)
    
    # Soft min using log-sum-exp
    m = np.min(areas)
    soft_min = m + np.log(np.sum(np.exp(-weight * (areas - m)))) / weight
    
    # Variance regularization on smallest triangles (top 20%)
    sorted_areas = np.sort(areas)
    k_small = max(1, len(sorted_areas) // 5)
    small_areas = sorted_areas[:k_small]
    var_penalty = var_weight * np.var(small_areas)
    
    return -soft_min + var_penalty

def hard_min_obj(bary_flat):
    """Hard minimum objective (for final evaluation)."""
    b = bary_flat.reshape(-1, 3)
    b = clamp_bary(b)
    pts = bary_to_cart(b)
    return -min_triangle_area_cart(pts)

def generate_hexagonal_init(n, seed=42):
    """Generate a hexagonal packing initialization in barycentric coords."""
    rng = np.random.RandomState(seed)
    pts = []
    # Place points in a grid pattern in barycentric space
    # For n=11, use a 3x4 grid clipped to triangle
    rows = 3
    cols = 4
    for r in range(rows):
        for c in range(cols):
            u = (c + 0.5) / cols
            v = (r + 0.5) / rows
            w = 1.0 - u - v
            if u > 0.02 and v > 0.02 and w > 0.02:
                pts.append([u, v, w])
    # If we have more than n, randomly select n
    if len(pts) > n:
        idx = rng.choice(len(pts), size=n, replace=False)
        pts = [pts[i] for i in idx]
    # If we have fewer, add random points
    while len(pts) < n:
        u = rng.random()
        v = rng.random() * (1 - u)
        w = 1 - u - v
        pts.append([u, v, w])
    return np.array(pts)

def generate_random_init(n, seed=0):
    """Generate random initialization in barycentric coords."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(n):
        u = rng.random()
        v = rng.random() * (1 - u)
        w = 1 - u - v
        # Add small margin
        u = 0.02 + 0.96 * u
        v = 0.02 + 0.96 * v
        w = 0.02 + 0.96 * w
        s = u + v + w
        pts.append([u/s, v/s, w/s])
    return np.array(pts)

def local_polish(bary_init, time_limit, deadline):
    """Run L-BFGS-B optimization starting from bary_init."""
    b0 = bary_init.flatten()
    
    result = minimize(
        smooth_min_obj,
        b0,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
    )
    
    # Check time
    if time.time() > deadline:
        return b0
    
    best_b = clamp_bary(result.x.reshape(-1, 3))
    best_score = -smooth_min_obj(best_b.flatten())
    
    # Try a few more restarts if time permits
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        # Perturb best solution
        perturbation = 0.05 * np.random.RandomState(100 + restart).randn(*best_b.shape)
        b_pert = clamp_bary(best_b + perturbation)
        
        res = minimize(
            smooth_min_obj,
            b_pert.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000}
        )
        
        if time.time() > deadline:
            break
        
        b_new = clamp_bary(res.x.reshape(-1, 3))
        score_new = -smooth_min_obj(b_new.flatten())
        
        if score_new > best_score:
            best_b = b_new
            best_score = score_new
    
    return best_b

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_bary = None
    best_score = -float('inf')
    
    # Try multiple initializations
    inits = []
    
    # Hexagonal init
    inits.append(('hex', generate_hexagonal_init(N, seed=42)))
    
    # Random inits with different seeds
    for seed in [0, 1, 2, 3, 7, 13, 21, 42]:
        inits.append((f'rand_{seed}', generate_random_init(N, seed=seed)))
    
    # Perturbed hexagonal
    hex_init = generate_hexagonal_init(N, seed=42)
    for seed in [10, 20, 30]:
        pert = 0.1 * np.random.RandomState(seed).randn(*hex_init.shape)
        inits.append((f'hex_pert_{seed}', clamp_bary(hex_init + pert)))
    
    for name, init_bary in inits:
        if time.time() > deadline - 15:
            break
        
        result_b = local_polish(init_bary, 10, deadline)
        
        if time.time() > deadline:
            break
        
        score = -smooth_min_obj(result_b.flatten())
        if score > best_score:
            best_score = score
            best_bary = result_b
    
    # Final evaluation with hard objective
    if best_bary is None:
        best_bary = generate_hexagonal_init(N, seed=42)
    
    # One more polish on the best solution
    if time.time() < deadline - 5:
        final_res = minimize(
            hard_min_obj,
            best_bary.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000}
        )
        if time.time() < deadline:
            candidate = clamp_bary(final_res.x.reshape(-1, 3))
            if -hard_min_obj(candidate.flatten()) > -hard_min_obj(best_bary.flatten()):
                best_bary = candidate
    
    # Convert to Cartesian and verify all points are inside
    pts = bary_to_cart(best_bary)
    
    # Verify points are inside the triangle
    bary_check = cart_to_bary(pts)
    for i in range(N):
        if np.any(bary_check[i] < -1e-6):
            # Clamp
            best_bary[i] = clamp_bary(best_bary[i])[0]
    
    return bary_to_cart(best_bary)
