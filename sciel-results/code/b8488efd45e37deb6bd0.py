"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

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
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-6):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
    """
    # Triangular lattice with spacing
    # Try different spacings
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        # Generate points on a triangular lattice
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        # Fallback: random
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def try_optimize(bary_init, method='L-BFGS-B', beta=100.0, maxiter=500):
        """Try optimizing from an initial configuration."""
        nonlocal best_bary, best_val
        
        if time.time() > deadline - 5:
            return
        
        try:
            def obj(x):
                return soft_min_obj(x, beta=beta)
            
            bounds = [(0, 1)] * (N * 2)
            
            result = minimize(
                obj,
                bary_init.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Final evaluation with hard min
            final_bary = clamp_bary(result.x.reshape(N, 2))
            final_val = -hard_min_obj(final_bary.flatten())
            
            if final_val > best_val:
                best_val = final_val
                best_bary = final_bary.copy()
                
        except Exception:
            pass
    
    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, method='L-BFGS-B', beta=50.0, maxiter=300)
    
    # 2. Try multiple random initializations with L-BFGS-B
    for i in range(5):
        if time.time() > deadline - 30:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        try_optimize(rand_init, method='L-BFGS-B', beta=50.0, maxiter=200)
    
    # 3. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(3):
            if time.time() > deadline - 15:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.03, seed=200 + i)
            try_optimize(perturbed, method='L-BFGS-B', beta=100.0, maxiter=300)
    
    # 4. Final polish with Nelder-Mead on hard min
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bary_nm = best_bary.copy()
            # Ensure strictly inside
            bary_nm[:, 0] = np.clip(bary_nm[:, 0], 1e-6, 1 - 1e-6)
            bary_nm[:, 1] = np.clip(bary_nm[:, 1], 1e-6, 1 - bary_nm[:, 0] - 1e-6)
            
            result = minimize(
                hard_min_obj,
                bary_nm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            final_val = -hard_min_obj(final_bary.flatten())
            
            if final_val > best_val:
                best_val = final_val
                best_bary = final_bary.copy()
        except Exception:
            pass
    
    # If all failed, generate a random valid set
    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    # Convert to Cartesian
    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    # Final safety check: ensure all points are strictly inside
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # For a point (x,y) to be inside:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  (below line from (0,0) to (0.5, H)? No, line from (0,0) to (0.5,H) is y = sqrt(3)x. Wait.
    #    Line 1: (0,0) to (1,0) => y=0. So y >= 0.
    #    Line 2: (0,0) to (0.5, H) => slope = H/0.5 = 2H = sqrt(3). y = sqrt(3)x.
    #            The triangle is to the right of this line? No, (1,0) has y=0, x=1. sqrt(3)*1 = 1.732. 0 < 1.732.
    #            So we need y <= sqrt(3)x ? Let's check (0.5, 0). y=0, sqrt(3)*0.5 = 0.866. 0 <= 0.866. Correct.
    #            Actually, the line equation is y = sqrt(3)x. The interior is y <= sqrt(3)x?
    #            Let's check centroid (0.5, H/3 = sqrt(3)/6).
    #            sqrt(3)*0.5 = sqrt(3)/2. sqrt(3)/6 < sqrt(3)/2. Yes.
    #    Line 3: (1,0) to (0.5, H) => slope = (H-0)/(0.5-1) = H/-0.5 = -2H = -sqrt(3).
    #            y - 0 = -sqrt(3)(x - 1) => y = -sqrt(3)x + sqrt(3).
    #            So y <= -sqrt(3)x + sqrt(3) => y + sqrt(3)x <= sqrt(3).
    
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            # Adjust x to satisfy y < sqrt(3)x
            # x > y / sqrt(3)
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            # Adjust x to satisfy y + sqrt(3)x < sqrt(3)
            # x < (sqrt(3) - y) / sqrt(3)
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)
            
    # Re-clamp y if x adjustment pushed it out?
    # It's safer to just re-clamp barycentric coords if we did manual cartesian fixes, 
    # but since we just fixed cartesian, let's ensure y is still valid.
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
             cart[i, 1] = 1e-9
             
    return cart
