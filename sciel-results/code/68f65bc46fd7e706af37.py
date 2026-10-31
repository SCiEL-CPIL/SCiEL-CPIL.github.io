import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2

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

def clamp_bary(bary, eps=1e-7):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
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

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
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

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def delaunay_variance_obj(bary_flat, lam=1.0):
    """
    Structural change: Use Delaunay triangulation to identify critical triangles.
    Objective: -min_area + lam * variance of Delaunay triangle areas.
    This encourages a more uniform distribution of small triangle areas,
    helping to escape local minima where one triangle is much smaller than others.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    # Get all triangle areas
    all_areas = calc_areas_bary(bary)
    min_area = np.min(all_areas)
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(cart)
        simplices = tri.simplices
        if len(simplices) == 0:
            return -min_area
        
        # Compute areas of Delaunay triangles
        d_areas = np.zeros(len(simplices))
        for i, (a, b, c) in enumerate(simplices):
            ab = cart[b] - cart[a]
            ac = cart[c] - cart[a]
            d_areas[i] = 0.5 * abs(ab[0]*ac[1] - ab[1]*ac[0])
        
        # Variance of Delaunay areas
        var = np.var(d_areas)
        
        # Objective: maximize min_area, minimize variance
        return -min_area + lam * var
    except Exception:
        return -min_area

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
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

def _local_polish_bary(points_cart, deadline):
    """
    Local refinement in barycentric coordinates using Nelder-Mead and L-BFGS-B.
    """
    bary = cart_to_bary(points_cart)
    bary = clamp_bary(bary)
    
    current_min_area = -hard_min_obj(bary.flatten())
    
    # Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            nm_result = minimize(
                hard_min_obj,
                bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            new_bary = clamp_bary(nm_result.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    # L-BFGS-B with soft min for finer tuning
    if time.time() < deadline - 3:
        try:
            lbfgs_result = minimize(
                lambda x: soft_min_obj(x, beta=300.0),
                bary.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            new_bary = clamp_bary(lbfgs_result.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    cart = bary_to_cart(bary)
    return cart, current_min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
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

    def try_optimize(bary_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300, use_delaunay=False):
        """
        Two-stage optimization with optional Delaunay variance regularization.
        Stage 1: Smooth landscape (low beta) to find a good basin.
        Stage 2: Sharper landscape (high beta) to refine.
        If use_delaunay, use the Delaunay variance objective for stage 1.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            # Stage 1
            if use_delaunay:
                def obj1(x):
                    return delaunay_variance_obj(x, lam=0.5)
            else:
                def obj1(x):
                    return soft_min_obj(x, beta=beta1)
            
            res1 = minimize(
                obj1,
                bary_init.flatten(),
                method=method,
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': maxiter1, 'ftol': 1e-10}
            )
            
            if res1.success:
                bary_mid = clamp_bary(res1.x.reshape(N, 2))
            else:
                bary_mid = clamp_bary(bary_init)
            
            # Stage 2
            if time.time() < deadline - 5:
                res2 = minimize(
                    lambda x: soft_min_obj(x, beta=beta2),
                    bary_mid.flatten(),
                    method=method,
                    bounds=[(0, 1)] * (N * 2),
                    options={'maxiter': maxiter2, 'ftol': 1e-12}
                )
                if res2.success:
                    bary_final = clamp_bary(res2.x.reshape(N, 2))
                else:
                    bary_final = bary_mid
            else:
                bary_final = bary_mid
                
            update_best(bary_final)
            
        except Exception:
            pass

    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal-like init
    hex_init = generate_hexagonal_init()
    inits.append(hex_init)
    
    # 2. Random inits
    rng = np.random.default_rng(42)
    for i in range(10):
        inits.append(generate_random_init(seed=42 + i))
        
    # 3. Perturbed hex init
    for scale in [0.02, 0.05, 0.1]:
        inits.append(generate_perturbed_init(hex_init, noise_scale=scale, seed=100 + int(scale*100)))
        
    # Optimize each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Use different methods/parameters for diversity
        # Every 3rd init uses Delaunay variance objective
        use_delaunay = (i % 3 == 0)
        
        if i % 3 == 0:
            try_optimize(init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=150, maxiter2=200, use_delaunay=use_delaunay)
        elif i % 3 == 1:
            try_optimize(init, method='SLSQP', beta1=100.0, beta2=400.0, maxiter1=100, maxiter2=150, use_delaunay=use_delaunay)
        else:
            try_optimize(init, method='L-BFGS-B', beta1=20.0, beta2=100.0, maxiter1=200, maxiter2=300, use_delaunay=use_delaunay)
            
    # Final polish with Nelder-Mead on the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        best_cart = bary_to_cart(best_bary)
        polished_cart, polished_val = _local_polish_bary(best_cart, deadline)
        if polished_val > best_val:
            best_bary = cart_to_bary(polished_cart)
            best_val = polished_val

    # Convert to Cartesian
    if best_bary is not None:
        cart_points = bary_to_cart(best_bary)
        return cart_points
    else:
        # Fallback
        return bary_to_cart(generate_random_init(0))

# EVOLVE-BLOCK-END
