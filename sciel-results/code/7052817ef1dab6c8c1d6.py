import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from itertools import combinations
import time
import warnings
warnings.filterwarnings('ignore')

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

def _delaunay_smooth_obj(bary_flat, beta=50.0):
    """
    Delaunay-based smooth surrogate: uses only Delaunay triangles for a smoother landscape.
    This reduces the number of constraints and provides better gradient information.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    try:
        tri = Delaunay(cart)
        # Get areas of Delaunay triangles
        idx = tri.simplices
        # Compute areas in barycentric space
        u = bary[:, 0]
        v = bary[:, 1]
        w = 1.0 - u - v
        
        ui = u[idx[:, 0]]
        vi = v[idx[:, 0]]
        wi = w[idx[:, 0]]
        
        uj = u[idx[:, 1]]
        vj = v[idx[:, 1]]
        wj = w[idx[:, 1]]
        
        uk = u[idx[:, 2]]
        vk = v[idx[:, 2]]
        wk = w[idx[:, 2]]
        
        det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
        areas = np.abs(det)
        
        # Soft min over Delaunay triangles
        scaled = -beta * areas
        max_scaled = np.max(scaled)
        if max_scaled > 700:
            return -np.min(areas)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta
        
        # Add a small penalty for non-Delaunay triangles to encourage global spread
        all_areas = calc_areas_bary(bary)
        min_all = np.min(all_areas)
        penalty = 0.1 * max(0, min_all - soft_min)
        
        return -(soft_min - penalty)
    except Exception:
        # Fallback to full soft-min
        return soft_min_obj(bary_flat, beta=beta)

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
    deadline = start_time + 142.0  # Safety margin before 150s kill
    
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
        Two-stage optimization:
        1. Smooth landscape (low beta) to find a good basin.
        2. Sharp landscape (high beta) to refine the minimum.
        Optionally uses Delaunay surrogate for stage 1.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            # Stage 1: Exploration
            if use_delaunay:
                def obj1(x):
                    return _delaunay_smooth_obj(x, beta=beta1)
            else:
                def obj1(x):
                    return soft_min_obj(x, beta=beta1)
            
            bounds = [(0, 1)] * (N * 2)
            
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Check time before stage 2
            if time.time() > deadline - 5:
                update_best(result1.x.reshape(N, 2))
                return

            # Stage 2: Exploitation
            def obj2(x):
                return soft_min_obj(x, beta=beta2)
            
            result2 = minimize(
                obj2,
                result1.x,
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            update_best(result2.x.reshape(N, 2))
                
        except Exception:
            pass

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=300, maxiter2=400)
    
    # 2. Try Delaunay-based exploration from hex init
    if time.time() < deadline - 60:
        try_optimize(hex_init, method='L-BFGS-B', beta1=40.0, beta2=250.0, maxiter1=200, maxiter2=300, use_delaunay=True)
    
    # 3. Try multiple random initializations with varied parameters
    num_random_starts = 14
    for i in range(num_random_starts):
        if time.time() > deadline - 45:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        # Vary beta and iteration counts for diversity
        beta1 = 35.0 + 8.0 * (i % 4)
        beta2 = 150.0 + 30.0 * (i % 3)
        maxiter1 = 120 + 40 * (i % 3)
        maxiter2 = 180 + 40 * (i % 2)
        use_del = (i % 3 == 0)  # Use Delaunay for every 3rd random start
        try_optimize(rand_init, method='L-BFGS-B', beta1=beta1, beta2=beta2, 
                    maxiter1=maxiter1, maxiter2=maxiter2, use_delaunay=use_del)
    
    # 4. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(6):
            if time.time() > deadline - 20:
                break
            noise_scale = 0.015 + 0.01 * (i % 3)
            perturbed = generate_perturbed_init(best_bary, noise_scale=noise_scale, seed=200 + i)
            try_optimize(perturbed, method='L-BFGS-B', beta1=80.0 + 20.0 * (i % 2), 
                        beta2=280.0 + 40.0 * (i % 2), maxiter1=200, maxiter2=300)
    
    # 5. Final polish: high-precision L-BFGS-B with very high beta
    if best_bary is not None and time.time() < deadline - 15:
        try:
            def obj_final(x):
                return soft_min_obj(x, beta=500.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_final = minimize(
                obj_final,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            update_best(result_final.x.reshape(N, 2))
            
        except Exception:
            pass
    
    # 6. Second polish pass with even tighter tolerance
    if best_bary is not None and time.time() < deadline - 10:
        try:
            def obj_polish2(x):
                return soft_min_obj(x, beta=800.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_polish2 = minimize(
                obj_polish2,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-13}
            )
            
            update_best(result_polish2.x.reshape(N, 2))
            
        except Exception:
            pass
    
    # 7. Additional local refinement with Nelder-Mead
    if best_bary is not None and time.time() < deadline - 8:
        try:
            nm_result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
            )
            update_best(nm_result.x.reshape(N, 2))
        except Exception:
            pass
    
    # 8. Final SLSQP refinement for precise constraint handling
    if best_bary is not None and time.time() < deadline - 5:
        try:
            def obj_slsqp(x):
                return soft_min_obj(x, beta=1000.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            # Inequality constraints: u >= 0, v >= 0, u+v <= 1 for each point
            def make_constraint(i):
                def constraint(x):
                    u = x[2*i]
                    v = x[2*i+1]
                    return np.array([u, v, 1.0 - u - v])
                return constraint
            
            constraints = []
            for i in range(N):
                constraints.append({'type': 'ineq', 'fun': make_constraint(i)})
            
            result_slsqp = minimize(
                obj_slsqp,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            
            update_best(result_slsqp.x.reshape(N, 2))
            
        except Exception:
            pass
    
    if best_bary is None:
        # Fallback
        best_bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
            [1/3, 1/3], [2/3, 0.0], [0.0, 2/3],
            [2/3, 1/3], [1/3, 2/3]
        ])
    
    # Final clamp to ensure validity
    best_bary = clamp_bary(best_bary, eps=1e-8)
    
    # Convert barycentric to Cartesian
    return bary_to_cart(best_bary)
