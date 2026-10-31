"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
import warnings
from scipy.optimize import differential_evolution, minimize
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

def de_obj(bary_flat):
    """
    Objective for differential evolution. Uses soft-min with moderate beta
    for a smooth landscape that DE can navigate.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Use log-sum-exp soft min for smoothness
    beta = 80.0
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def slsqp_obj(bary_flat):
    """
    Objective for SLSQP: minimize negative min area.
    """
    bary = bary_flat.reshape(N, 2)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def slsqp_constraints():
    """
    Define constraints for SLSQP in barycentric (u, v) space.
    For each point i: u_i >= 0, v_i >= 0, u_i + v_i <= 1
    """
    constraints = []
    for i in range(N):
        # u_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        # v_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        # u_i + v_i <= 1  =>  1 - u_i - v_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    return constraints

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
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
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
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
    
    # Phase 1: Differential Evolution global search
    # DE explores the full 22-dim space with population-based search
    bounds = [(0.001, 0.999)] * (N * 2)  # Box bounds on u, v
    
    def check_time():
        return time.time() < deadline - 20  # Reserve 20s for refinement
    
    if check_time():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                de_result = differential_evolution(
                    de_obj,
                    bounds=bounds,
                    maxiter=60,
                    popsize=15,
                    tol=1e-8,
                    mutation=(0.3, 1.0),
                    recombination=0.7,
                    seed=42,
                    polish=False,  # We'll do our own refinement
                    init='latinhypercube',
                    update_interval=5,
                    callback=None,
                    x0=None,
                    workers=1
                )
            
            if de_result.success or de_result.x is not None:
                de_bary = de_result.x.reshape(N, 2)
                update_best(de_bary)
        except Exception:
            pass
    
    # Phase 2: SLSQP refinement from DE result and hex init
    constraints = slsqp_constraints()
    
    def try_slsqp(bary_init, maxiter=500):
        """Run SLSQP refinement from a starting point."""
        if time.time() > deadline - 15:
            return
        try:
            x0 = bary_init.flatten()
            # Ensure feasible start
            x0 = np.clip(x0, 1e-5, 1 - 1e-5)
            for i in range(N):
                if x0[2*i] + x0[2*i+1] > 1 - 1e-5:
                    s = (1 - 1e-5) / max(x0[2*i] + x0[2*i+1], 1e-10)
                    x0[2*i] *= s
                    x0[2*i+1] *= s
            
            result = minimize(
                slsqp_obj,
                x0,
                method='SLSQP',
                bounds=[(1e-6, 1 - 1e-6)] * (N * 2),
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-14}
            )
            
            if result.x is not None:
                update_best(result.x.reshape(N, 2))
        except Exception:
            pass
    
    # SLSQP from DE result
    if best_bary is not None:
        try_slsqp(best_bary, maxiter=400)
    
    # SLSQP from hexagonal init
    hex_init = generate_hexagonal_init()
    try_slsqp(hex_init, maxiter=300)
    
    # Phase 3: Multiple SLSQP refinements from perturbed best
    if best_bary is not None:
        for i in range(8):
            if time.time() > deadline - 10:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.03, seed=500 + i)
            try_slsqp(perturbed, maxiter=200)
    
    # Phase 4: Final polish with L-BFGS-B on soft-min (smooth)
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds_lbfgs = [(1e-6, 1 - 1e-6)] * (N * 2)
            result = minimize(
                soft_min_obj,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds_lbfgs,
                args=(200.0,),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead final polish on hard min
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bary_nm = best_bary.copy()
            bary_nm[:, 0] = np.clip(bary_nm[:, 0], 1e-5, 1 - 1e-5)
            bary_nm[:, 1] = np.clip(bary_nm[:, 1], 1e-5, 1 - bary_nm[:, 0] - 1e-5)
            
            result = minimize(
                hard_min_obj,
                bary_nm.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            final_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(final_bary)
        except Exception:
            pass
    
    # If all failed, generate a random valid set
    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
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
