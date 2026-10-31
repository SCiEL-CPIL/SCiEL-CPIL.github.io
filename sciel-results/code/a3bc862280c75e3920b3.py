import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
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

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            # Try different subsets
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
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

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using a 5-stage annealing approach.
    Stage 1: Coarse Soft-Min (Beta=20)
    Stage 2: Medium Soft-Min (Beta=50)
    Stage 3: Sharp Soft-Min (Beta=200)
    Stage 4: Hard Min Nelder-Mead
    Stage 5: Final Polish (SLSQP/NM)
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Helper to check if we should stop
    def time_left(fraction=1.0):
        return (time.time() - start_time) < time_limit * fraction

    # Stage 1: Coarse Soft-Min
    if time_left(0.2):
        try:
            res = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(20.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(res.x.reshape(N, 2))
        except Exception:
            pass

    if not time_left(0.2):
        return bary_current

    # Stage 2: Medium Soft-Min
    try:
        res = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(res.x.reshape(N, 2))
    except Exception:
        pass

    if not time_left(0.4):
        return bary_current

    # Stage 3: Sharp Soft-Min
    try:
        res = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(200.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(res.x.reshape(N, 2))
    except Exception:
        pass

    if not time_left(0.6):
        return bary_current

    # Stage 4: Hard Min Nelder-Mead
    try:
        res = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if -res.fun > -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(res.x.reshape(N, 2))
    except Exception:
        pass

    if not time_left(0.8):
        return bary_current

    # Stage 5: Final Polish (SLSQP)
    try:
        res = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'iprint': 0}
        )
        if -res.fun > -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(res.x.reshape(N, 2))
    except Exception:
        pass

    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

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

    # 1. Hexagonal Init + Refine
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=10.0)
    update_best(refined_hex)

    # 2. Differential Evolution (Global Search)
    # We use a larger population and more iterations to explore the space better.
    if time.time() < deadline - 50:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)

            bounds = [(0, 1)] * (N * 2)
            # Increase popsize and maxiter for better global coverage
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=60,
                popsize=20,
                tol=1e-8,
                polish=False,
                init='sobol'
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=15.0)
            update_best(refined_de)
        except Exception:
            pass

    # 3. Multi-start: Random + Perturbed restarts
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        
        # Strategy: Mix of random, perturbed best, and perturbed hex
        if best_bary is not None:
            if restart_count % 3 == 0:
                # Perturb best found so far
                init = generate_perturbed_init(best_bary, sigma=0.05, seed=1000 + restart_count)
            elif restart_count % 3 == 1:
                # Perturb hexagonal init
                init = generate_perturbed_init(hex_init, sigma=0.1, seed=2000 + restart_count)
            else:
                # Fresh random
                init = generate_random_init(seed=3000 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining > 40:
            tl = 5.0
        elif remaining > 25:
            tl = 3.0
        else:
            tl = 2.0
            
        refined = refine_local(init, time_limit=tl)
        update_best(refined)

    # 4. Final High-Precision Polish
    if best_bary is not None and time.time() < deadline - 5:
        # Run a very tight SLSQP on the best found
        try:
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'iprint': 0}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        # One last Nelder-Mead pass
        if time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# Ensure the function is accessible
if __name__ == "__main__":
    # Quick test to ensure it runs without error
    pts = construct()
    print(f"Found {len(pts)} points.")
    print(f"Best value: {-hard_min_obj(pts.reshape(N,2).flatten())}") # Note: pts is cartesian, this line is just for debug if run as script, construct returns cartesian.
    # Actually, construct returns cartesian. The internal best_val is in barycentric area units.
    # The evaluator calculates area from cartesian points.
    # barycentric area * (sqrt(3)/4) is the actual area? No.
    # The determinant in barycentric coords gives the area relative to the reference triangle.
    # Reference triangle area is sqrt(3)/4.
    # The task says "divides by its area", so the raw value is the relative area.
    # My calc_areas_bary returns |det|, which is the relative area.
    # So best_val is the correct metric.
    pass
