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
    Generate a structured initial configuration based on triangular lattice points.
    This provides a strong starting point for local refinement.
    """
    # Triangular lattice points in barycentric coordinates
    # For 11 points, a 3x3 grid plus corners or similar doesn't fit perfectly.
    # Let's use a set of points that are well-distributed.
    # A known good configuration often involves points near the centroid and vertices.
    
    # Manually curated good start based on symmetry
    # (u, v) where w = 1-u-v
    bary = np.array([
        [1/6, 1/6],      # Center
        [1/3, 1/3],      # 
        [2/3, 1/6],      #
        [1/6, 2/3],      #
        [1/2, 1/6],      #
        [1/6, 1/2],      #
        [2/3, 1/3],      #
        [1/3, 2/3],      #
        [0.05, 0.05],    # Near vertex
        [0.95, 0.05],    # Near vertex
        [0.05, 0.95]     # Near vertex
    ])
    
    return clamp_bary(bary)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    1. L-BFGS-B with soft-min (graduated beta)
    2. Nelder-Mead with hard-min
    3. Coordinate Descent with hard-min (polish)
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())

    def try_update(result):
        nonlocal bary_current, current_val
        if result is not None:
            new_bary = clamp_bary(result.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
                return True
        return False

    # Stage 1: Coarse soft-min
    if time.time() < start_time + time_limit * 0.2:
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(20.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            try_update(result1)
        except Exception:
            pass

    # Stage 2: Medium soft-min
    if time.time() < start_time + time_limit * 0.4:
        try:
            result2 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(50.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            try_update(result2)
        except Exception:
            pass

    # Stage 3: Sharp soft-min
    if time.time() < start_time + time_limit * 0.6:
        try:
            result3 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(200.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            try_update(result3)
        except Exception:
            pass

    # Stage 4: Hard min Nelder-Mead
    if time.time() < start_time + time_limit * 0.8:
        try:
            result4 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            try_update(result4)
        except Exception:
            pass

    # Stage 5: Coordinate Descent Polish
    # Optimize each point individually. This is often the most effective final step
    # for packing problems as it resolves local conflicts between specific points.
    if time.time() < start_time + time_limit:
        bary_cd = bary_current.copy()
        val_cd = -hard_min_obj(bary_cd.flatten())
        
        # Perform a few sweeps
        max_sweeps = 3
        for sweep in range(max_sweeps):
            if time.time() > start_time + time_limit:
                break
            for i in range(N):
                if time.time() > start_time + time_limit:
                    break
                # Optimize point i
                def point_obj(xi):
                    bary_temp = bary_cd.copy()
                    bary_temp[i] = xi
                    bary_temp = clamp_bary(bary_temp)
                    areas = calc_areas_bary(bary_temp)
                    return -np.min(areas)
                
                try:
                    res_i = minimize(
                        point_obj,
                        bary_cd[i],
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                    )
                    if res_i.fun < val_cd:
                        bary_cd[i] = res_i.x
                        bary_cd = clamp_bary(bary_cd)
                        val_cd = -hard_min_obj(bary_cd.flatten())
                except Exception:
                    pass
        
        if val_cd > current_val:
            bary_current = bary_cd
            current_val = val_cd

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

    # 1. Global Search using Differential Evolution
    # DE is robust to non-smoothness. We use a moderate soft-min objective.
    bounds = [(0, 1)] * (N * 2)
    
    def de_callback(xk, convergence):
        return time.time() > deadline - 60  # Stop DE to allow time for refinement

    de_result = None
    try:
        de_result = differential_evolution(
            soft_min_obj,
            bounds,
            args=(50.0,),
            maxiter=150,
            popsize=25,
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol',
            callback=de_callback
        )
    except Exception as e:
        pass

    if de_result is not None:
        de_bary = clamp_bary(de_result.x.reshape(N, 2))
        update_best(de_bary)

    # 2. Refine the DE result
    if best_bary is not None:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            refined_de = refine_local(best_bary, time_limit=min(50.0, remaining_time))
            update_best(refined_de)

    # 3. Try Structured Initialization (Hexagonal/Lattice)
    if time.time() < deadline - 30:
        hex_init = generate_hexagonal_init()
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            refined_hex = refine_local(hex_init, time_limit=min(30.0, remaining_time))
            update_best(refined_hex)

    # 4. Random Restarts with Local Refinement
    if time.time() < deadline - 20:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 15:
                break
            rng = np.random.default_rng(seed)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            rand_bary = np.column_stack([u, v])
            
            remaining_time = deadline - time.time() - 5
            if remaining_time > 5:
                refined_rand = refine_local(rand_bary, time_limit=min(15.0, remaining_time))
                update_best(refined_rand)

    # 5. Final Polish if time remains
    if best_bary is not None and time.time() < deadline - 5:
        final_polish = refine_local(best_bary, time_limit=min(10.0, deadline - time.time() - 2))
        update_best(final_polish)

    if best_bary is None:
        # Absolute fallback
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    cart = bary_to_cart(best_bary)
    return cart

if __name__ == "__main__":
    pts = construct()
    print(pts)
