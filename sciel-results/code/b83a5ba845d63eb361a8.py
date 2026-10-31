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

def clamp_bary(bary, eps=1e-8):
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
        # Scale u and v down to fit
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
    Generate a hexagonal grid of points inside the triangle.
    This provides a structured starting point close to the optimal solution.
    """
    # Create a grid of points in barycentric coordinates
    # For N=11, a 3x3 grid + 2 extra or similar structure works well.
    # Let's try to place points roughly in a hexagonal pattern.
    
    # Simple approach: divide the triangle into smaller triangles.
    # For 11 points, we can use a grid of 4x4 and select 11, or generate a hex lattice.
    
    # Hex lattice generation in barycentric coords is tricky.
    # Let's use a simple grid in Cartesian, convert to barycentric, and keep those inside.
    
    # Grid size
    n_grid = 4
    xs = np.linspace(0.1, 0.9, n_grid)
    ys = np.linspace(0.1, H - 0.1, n_grid)
    
    candidates = []
    for x in xs:
        for y in ys:
            # Check if point is inside triangle
            # y <= sqrt(3) * x
            # y <= -sqrt(3) * (x - 1)  => y + sqrt(3)*x <= sqrt(3)
            if y <= np.sqrt(3) * x and y + np.sqrt(3) * x <= np.sqrt(3):
                # Convert to barycentric
                # w = y / H
                # v = x - 0.5 * w
                # u = 1 - v - w
                w = y / H
                v = x - 0.5 * w
                u = 1 - v - w
                if u > 0 and v > 0 and w > 0:
                    candidates.append([u, v])
    
    candidates = np.array(candidates)
    
    # We need exactly N points. If we have more, we need to select N.
    # If we have less, we need to add random points.
    
    if len(candidates) >= N:
        # Select N points that are most spread out?
        # For simplicity, just take the first N or use a greedy selection.
        # Let's use a simple greedy selection to maximize min distance initially.
        selected = [candidates[0]]
        for i in range(1, N):
            # Calculate min distance to selected points for each candidate
            dists = np.min([np.linalg.norm(candidates - sel, axis=1) for sel in selected], axis=0)
            # Pick the candidate with max min distance
            idx = np.argmax(dists)
            selected.append(candidates[idx])
        return np.array(selected)
    else:
        # Add random points to fill up
        rng = np.random.default_rng(42)
        u = rng.random(N - len(candidates))
        v = rng.random(N - len(candidates)) * (1 - u)
        extra = np.column_stack([u, v])
        return np.vstack([candidates, extra])

def generate_perturbed_init(bary_init, sigma=0.05, seed=42):
    """
    Generate a perturbed version of an initial configuration.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, bary_init.shape)
    perturbed = bary_init + noise
    return clamp_bary(perturbed)

def refine_local(bary_init, beta=200.0, maxiter=500):
    """
    Refine a configuration using L-BFGS on the soft-min objective.
    """
    try:
        def obj(x):
            return soft_min_obj(x, beta=beta)
        
        bounds = [(0, 1)] * (N * 2)
        
        result = minimize(
            obj,
            bary_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        return result.x.reshape(N, 2)
    except Exception:
        return bary_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Strategy: Multi-start local search with structured initialization
    # 1. Hexagonal grid init
    # 2. Perturbations of hex init
    # 3. Random inits
    
    # 1. Hexagonal init
    hex_init = generate_hexagonal_init()
    refined = refine_local(hex_init, beta=100.0, maxiter=200)
    update_best(refined)
    
    # 2. Perturbations
    for i in range(10):
        if time.time() > deadline - 20:
            break
        perturbed = generate_perturbed_init(hex_init, sigma=0.05 + 0.01*i, seed=100+i)
        refined = refine_local(perturbed, beta=100.0, maxiter=200)
        update_best(refined)
        
    # 3. Random inits
    rng = np.random.default_rng(123)
    for i in range(5):
        if time.time() > deadline - 10:
            break
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        init_bary = np.column_stack([u, v])
        refined = refine_local(init_bary, beta=100.0, maxiter=200)
        update_best(refined)

    # Final polish with Nelder-Mead on hard min if time allows
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bary_nm = best_bary.copy()
            # Ensure strictly inside
            bary_nm = clamp_bary(bary_nm, eps=1e-6)
            
            result = minimize(
                hard_min_obj,
                bary_nm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16}
            )
            
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    # If all failed, generate a random valid set
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    # Convert to Cartesian
    bary_final = clamp_bary(best_bary, eps=1e-5)
    cart = bary_to_cart(bary_final)
    
    # Final safety check: ensure all points are strictly inside
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)
            
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
             cart[i, 1] = 1e-9
             
    return cart
