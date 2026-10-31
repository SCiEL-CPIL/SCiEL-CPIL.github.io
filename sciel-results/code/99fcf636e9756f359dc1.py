"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
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

def coordinate_descent(bary, max_iters=100, tol=1e-10):
    """
    Perform coordinate descent to maximize the minimum triangle area.
    For each point, we adjust its barycentric coordinates to maximize the
    minimum area of triangles involving that point.
    """
    bary = clamp_bary(bary)
    current_min_area = -hard_min_obj(bary.flatten())
    
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            # Save current point
            old_pt = bary[i].copy()
            
            # We want to maximize min(areas involving i).
            # Areas not involving i are constant.
            # Let A_const be min area not involving i.
            # Let A_i be areas involving i.
            # We want to maximize min(A_const, min(A_i)).
            
            # Compute areas not involving i
            # This is expensive if done naively, but N is small.
            # Alternatively, just evaluate the full objective for candidate moves.
            
            # Simple hill climbing in barycentric space for point i
            # Try moving in small steps in u and v directions.
            best_local_val = current_min_area
            best_local_pt = old_pt.copy()
            
            step = 0.01
            for _ in range(50): # Inner loop for this point
                found_improvement = False
                for du in [-step, 0, step]:
                    for dv in [-step, 0, step]:
                        if du == 0 and dv == 0: continue
                        new_u = old_pt[0] + du
                        new_v = old_pt[1] + dv
                        # Check bounds roughly
                        if new_u < 0 or new_v < 0 or new_u + new_v > 1:
                            continue
                        
                        new_bary = bary.copy()
                        new_bary[i] = [new_u, new_v]
                        new_bary = clamp_bary(new_bary)
                        
                        val = -hard_min_obj(new_bary.flatten())
                        if val > best_local_val + tol:
                            best_local_val = val
                            best_local_pt = new_bary[i].copy()
                            found_improvement = True
                            old_pt = new_bary[i].copy() # Update reference for next step
                            
                if not found_improvement:
                    break
                step *= 0.7 # Shrink step
                
            if np.linalg.norm(best_local_pt - bary[i]) > 1e-8:
                bary[i] = best_local_pt
                bary = clamp_bary(bary)
                new_global_val = -hard_min_obj(bary.flatten())
                if new_global_val > current_min_area:
                    current_min_area = new_global_val
                    improved = True
                    
        if not improved:
            break
            
    return bary

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
    
    # 2. Try multiple random initializations
    num_random_starts = 8
    for i in range(num_random_starts):
        if time.time() > deadline - 30:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        # Use slightly fewer iterations for random starts to allow more of them
        try_optimize(rand_init, method='L-BFGS-B', beta1=50.0, beta2=150.0, maxiter1=150, maxiter2=200)
        
    # 3. Differential Evolution for global search
    if time.time() < deadline - 60:
        try:
            # DE bounds: u in [0,1], v in [0,1] (clamped later)
            de_bounds = [(0, 1)] * (N * 2)
            
            def de_obj(x):
                b = x.reshape(N, 2)
                b = clamp_bary(b)
                # Use soft min for DE to provide gradient-like info? 
                # DE doesn't use gradients, but soft min is smoother.
                # However, hard min is what we want to maximize.
                # Soft min with high beta is a good proxy.
                return soft_min_obj(x, beta=100.0)
            
            # Run DE with limited budget
            de_result = differential_evolution(
                de_obj, 
                de_bounds, 
                seed=42, 
                maxiter=50, 
                popsize=10,
                polish=False, # We will polish manually
                tol=1e-6
            )
            
            if de_result.success or de_result.fun < 0: # Check if it found something
                de_bary = de_result.x.reshape(N, 2)
                # Refine DE result with L-BFGS
                try_optimize(de_bary, method='L-BFGS-B', beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300)
        except Exception:
            pass

    # 4. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(5):
            if time.time() > deadline - 20:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            try_optimize(perturbed, method='L-BFGS-B', beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300)
    
    # 5. Final polish with Coordinate Descent (Direct Search)
    if best_bary is not None and time.time() < deadline - 10:
        try:
            # Coordinate descent is very good for max-min problems
            cd_bary = coordinate_descent(best_bary, max_iters=50)
            update_best(cd_bary)
        except Exception:
            pass
            
    # 6. Final polish with Nelder-Mead on hard min
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
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16}
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
