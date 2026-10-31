"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
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

def clamp_bary(bary, eps=1e-7):
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
    
    for s in np.linspace(0.08, 0.5, 30):
        # Generate points on a triangular lattice
        points = []
        for i in range(8):
            for j in range(8):
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

def refine_local(bary_init, deadline, max_beta=800.0, n_stages=3):
    """
    Multi-stage local refinement using increasing beta values.
    Returns the best bary found.
    """
    current = bary_init.copy()
    best_val = -hard_min_obj(current.flatten())
    best_bary = current.copy()
    
    betas = np.linspace(30, max_beta, n_stages)
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        def obj(x):
            return soft_min_obj(x, beta=beta)
        
        bounds = [(0, 1)] * (N * 2)
        
        try:
            result = minimize(
                obj,
                current.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            
            if new_val > best_val:
                best_val = new_val
                best_bary = new_bary.copy()
            current = new_bary
        except Exception:
            pass
    
    # Final SLSQP polish on hard min
    if time.time() > deadline - 10:
        return best_bary
    
    try:
        def slsqp_obj(x):
            bary = x.reshape(N, 2)
            bary = clamp_bary(bary)
            return -np.min(calc_areas_bary(bary))
        
        # Constraints: u >= 0, v >= 0, u+v <= 1 for each point
        cons = []
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - 1e-8})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - 1e-8})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - 1e-8 - x[2*i] - x[2*i+1]})
        
        bounds_slsqp = [(1e-8, 1 - 1e-8)] * (N * 2)
        
        result = minimize(
            slsqp_obj,
            best_bary.flatten(),
            method='SLSQP',
            bounds=bounds_slsqp,
            constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-15}
        )
        
        new_bary = clamp_bary(result.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        
        if new_val > best_val:
            best_val = new_val
            best_bary = new_bary.copy()
    except Exception:
        pass
    
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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
    
    # 1. Start with hexagonal lattice initialization and refine
    hex_init = generate_hexagonal_init()
    if time.time() < deadline - 20:
        refined = refine_local(hex_init, deadline, max_beta=500.0, n_stages=4)
        update_best(refined)
    
    # 2. Differential evolution for global search (if time permits)
    if time.time() < deadline - 40 and best_bary is not None:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            bounds_de = [(0, 1)] * (N * 2)
            
            result_de = differential_evolution(
                de_obj,
                bounds_de,
                seed=42,
                maxiter=20,
                popsize=15,
                tol=1e-10,
                polish=False
            )
            
            if result_de.success or result_de.fun < best_val:
                de_bary = clamp_bary(result_de.x.reshape(N, 2))
                de_val = -hard_min_obj(de_bary.flatten())
                if de_val > best_val:
                    best_val = de_val
                    best_bary = de_bary.copy()
                    # Refine the DE result
                    if time.time() < deadline - 15:
                        refined_de = refine_local(de_bary, deadline, max_beta=600.0, n_stages=3)
                        update_best(refined_de)
        except Exception:
            pass
    
    # 3. Multiple random initializations with refinement
    num_random_starts = 12
    for i in range(num_random_starts):
        if time.time() > deadline - 30:
            break
        rng_seed = 100 + i * 7
        rand_init = generate_random_init(seed=rng_seed)
        
        # Quick refine with moderate settings
        if time.time() < deadline - 25:
            refined = refine_local(rand_init, deadline, max_beta=400.0, n_stages=3)
            update_best(refined)
    
    # 4. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(8):
            if time.time() > deadline - 15:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.015, seed=300 + i * 3)
            refined = refine_local(perturbed, deadline, max_beta=700.0, n_stages=3)
            update_best(refined)
    
    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 10:
        try:
            def obj_final(x):
                return soft_min_obj(x, beta=1000.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_final = minimize(
                obj_final,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            update_best(result_final.x.reshape(N, 2))
        except Exception:
            pass
    
    # 6. Final SLSQP polish on hard min
    if best_bary is not None and time.time() < deadline - 5:
        try:
            def slsqp_obj_final(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            cons = []
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - 1e-8})
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - 1e-8})
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - 1e-8 - x[2*i] - x[2*i+1]})
            
            bounds_slsqp = [(1e-8, 1 - 1e-8)] * (N * 2)
            
            result = minimize(
                slsqp_obj_final,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds_slsqp,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-16}
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
    bary_final = clamp_bary(best_bary, eps=1e-6)
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
