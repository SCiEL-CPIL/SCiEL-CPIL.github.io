"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

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

def cart_to_bary(cart):
    """
    Convert Cartesian coords to barycentric (u, v).
    """
    w = cart[:, 1] / H
    v = cart[:, 0] - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration optimized for the triangle.
    Uses a structured grid aligned with the triangle's geometry.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try multiple structured patterns
    # Pattern 1: Triangular grid with varying spacing
    for s in np.linspace(0.12, 0.45, 15):
        points = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s
                if u + v <= 1 - 0.01 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    # Pattern 2: Points along edges and interior
    edge_points = [
        [0.05, 0.05], [0.95, 0.05], [0.5, 0.05],
        [0.05, 0.95], [0.95, 0.95],
        [0.5, 0.5], [0.25, 0.25], [0.75, 0.25],
        [0.25, 0.75], [0.75, 0.75], [0.5, 0.4]
    ]
    for offset in np.linspace(-0.1, 0.1, 5):
        for scale in np.linspace(0.8, 1.0, 3):
            pts = []
            for p in edge_points:
                u = max(0.01, min(0.98, p[0] * scale + offset))
                v = max(0.01, min(0.98, p[1] * scale + offset))
                if u + v <= 0.98:
                    pts.append([u, v])
            if len(pts) == N:
                bary = np.array(pts)
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    
    # Pattern 3: Optimized positions based on known good configurations
    known_good = np.array([
        [0.1, 0.1], [0.9, 0.1], [0.5, 0.05],
        [0.1, 0.85], [0.9, 0.85], [0.5, 0.8],
        [0.3, 0.3], [0.7, 0.3], [0.5, 0.5],
        [0.3, 0.6], [0.7, 0.6]
    ])
    for noise_scale in [0.0, 0.02, 0.05]:
        bary = known_good.copy()
        if noise_scale > 0:
            rng = np.random.default_rng(42)
            bary += rng.normal(0, noise_scale, bary.shape)
        bary = clamp_bary(bary)
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

def _delaunay_smooth_obj(bary_flat, beta=200.0, lam=0.1):
    """
    Hybrid objective: combines soft-min of all triangle areas with a smooth
    regularization term based on the variance of the smallest areas.
    This helps push the solution toward a configuration where the critical
    triangles are more evenly balanced.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min term
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        sm = np.min(areas)
    else:
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        sm = -log_sum / beta
    
    # Variance regularization on the smallest 15% of areas (tighter focus)
    n_small = max(1, int(0.15 * len(areas)))
    sorted_areas = np.sort(areas)
    small_areas = sorted_areas[:n_small]
    variance_term = np.var(small_areas)
    
    # Also add a term that penalizes points too close to edges
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    edge_proximity = np.minimum(np.minimum(u, v), w)
    edge_penalty = np.sum(np.where(edge_proximity < 0.05, (0.05 - edge_proximity) ** 2, 0.0))
    
    return -(sm - lam * variance_term - 0.01 * edge_penalty)

def _coordinate_descent_polish(bary, deadline, max_rounds=50):
    """
    Coordinate descent: optimize each point's (u,v) while fixing others.
    This can find improvements that global optimizers miss.
    """
    bary = bary.copy()
    current_min = -hard_min_obj(bary.flatten())
    
    for round_idx in range(max_rounds):
        if time.time() > deadline - 3:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 3:
                break
            
            # Save original
            orig_u, orig_v = bary[i, 0], bary[i, 1]
            
            # Try small perturbations in u and v directions
            for du in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                for dv in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                    new_bary = bary.copy()
                    new_bary[i, 0] = orig_u + du
                    new_bary[i, 1] = orig_v + dv
                    new_bary = clamp_bary(new_bary)
                    new_min = -hard_min_obj(new_bary.flatten())
                    if new_min > current_min + 1e-14:
                        bary = new_bary
                        current_min = new_min
                        improved = True
                        orig_u, orig_v = new_bary[i, 0], new_bary[i, 1]
        
        if not improved:
            break
    
    return bary, current_min

def _local_polish_bary(points_cart, deadline):
    """
    Local refinement in barycentric coordinates using Nelder-Mead, L-BFGS-B, and coordinate descent.
    """
    bary = cart_to_bary(points_cart)
    bary = clamp_bary(bary)
    
    current_min_area = -hard_min_obj(bary.flatten())
    
    # Nelder-Mead polish
    if time.time() < deadline - 8:
        try:
            nm_result = minimize(
                hard_min_obj,
                bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            new_bary = clamp_bary(nm_result.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    # L-BFGS-B with high beta soft min for near-exact min tracking
    if time.time() < deadline - 6:
        try:
            lbfgs_result = minimize(
                lambda x: soft_min_obj(x, beta=500.0),
                bary.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            new_bary = clamp_bary(lbfgs_result.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < deadline - 3:
        try:
            bary, current_min_area = _coordinate_descent_polish(bary, deadline, max_rounds=30)
        except Exception:
            pass
    
    # Final Nelder-Mead on the result of coordinate descent
    if time.time() < deadline - 2:
        try:
            nm_result = minimize(
                hard_min_obj,
                bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            new_bary = clamp_bary(nm_result.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    # Convert back to Cartesian
    cart = bary_to_cart(bary)
    return cart, current_min_area

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

    def try_optimize(bary_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300):
        """
        Two-stage optimization:
        1. Smooth landscape (low beta) to find a good basin.
        2. Sharp landscape (high beta) to refine the minimum.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            # Stage 1: Exploration
            def obj1(x):
                return soft_min_obj(x, beta=beta1)
            
            bounds = [(0, 1)] * (N * 2)
            
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            
            # Check time before stage 2
            if time.time() > deadline - 5:
                update_best(result1.x.reshape(N, 2))
                return

            # Stage 2: Exploitation with higher beta
            def obj2(x):
                return soft_min_obj(x, beta=beta2)
            
            result2 = minimize(
                obj2,
                result1.x,
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-15, 'gtol': 1e-11}
            )
            
            update_best(result2.x.reshape(N, 2))
                
        except Exception:
            pass

    def try_delaunay_optimize(bary_init, maxiter=300):
        """
        Optimize using the hybrid Delaunay-aware smooth objective.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            bounds = [(0, 1)] * (N * 2)
            
            result = minimize(
                lambda x: _delaunay_smooth_obj(x, beta=200.0, lam=0.05),
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            
            update_best(result.x.reshape(N, 2))
            
        except Exception:
            pass

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, method='L-BFGS-B', beta1=40.0, beta2=250.0, maxiter1=300, maxiter2=500)
    
    # 2. Try multiple random initializations with varied betas
    num_random_starts = 12
    for i in range(num_random_starts):
        if time.time() > deadline - 40:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        # Vary beta schedule for diversity
        if i % 3 == 0:
            try_optimize(rand_init, method='L-BFGS-B', beta1=30.0, beta2=150.0, maxiter1=200, maxiter2=300)
        elif i % 3 == 1:
            try_optimize(rand_init, method='L-BFGS-B', beta1=60.0, beta2=200.0, maxiter1=200, maxiter2=300)
        else:
            try_optimize(rand_init, method='L-BFGS-B', beta1=50.0, beta2=300.0, maxiter1=150, maxiter2=250)
        
    # 3. Perturb the best found and re-optimize with multiple noise scales
    if best_bary is not None:
        for i in range(8):
            if time.time() > deadline - 25:
                break
            noise = 0.01 + 0.01 * i
            perturbed = generate_perturbed_init(best_bary, noise_scale=noise, seed=200 + i)
            try_optimize(perturbed, method='L-BFGS-B', beta1=80.0, beta2=350.0, maxiter1=200, maxiter2=400)
    
    # 4. Delaunay-aware optimization from best
    if best_bary is not None and time.time() < deadline - 20:
        try_delaunay_optimize(best_bary, maxiter=400)
    
    # 5. Additional optimization round from best with different method
    if best_bary is not None and time.time() < deadline - 15:
        try:
            # Try Powell's method for a different search direction
            pw_result = minimize(
                lambda x: soft_min_obj(x, beta=400.0),
                best_bary.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            update_best(pw_result.x.reshape(N, 2))
        except Exception:
            pass
    
    # 6. Final polish with Nelder-Mead on hard min
    if best_bary is not None and time.time() < deadline - 12:
        try:
            nm_result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            update_best(nm_result.x.reshape(N, 2))
        except Exception:
            pass
    
    # 7. Convert best barycentric to Cartesian and do final local polish
    if best_bary is not None:
        best_cart = bary_to_cart(best_bary)
        # Final local polish in Cartesian space with coordinate descent
        if time.time() < deadline - 4:
            best_cart, _ = _local_polish_bary(best_cart, deadline)
        return best_cart
    else:
        # Fallback: return hexagonal init converted to Cartesian
        return bary_to_cart(hex_init)
