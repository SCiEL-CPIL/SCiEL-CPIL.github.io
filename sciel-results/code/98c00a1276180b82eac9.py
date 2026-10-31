import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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
    """Objective using hard min. Returns negative min area to be minimized."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try a few scales to find a good subset of 11 points
    for scale in [0.2, 0.25, 0.3, 0.35, 0.4]:
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
            # Use the first N points or a random subset? 
            # Hexagonal lattice points are symmetric. Let's try to pick a "central" subset if possible, 
            # or just random subsets to break symmetry.
            rng = np.random.default_rng(42)
            for _ in range(10):
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

def local_minimize(bary_init, maxiter=1000):
    """
    Perform local minimization using Nelder-Mead, which is robust to non-smoothness.
    """
    result = minimize(
        hard_min_obj,
        bary_init.flatten(),
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
    )
    return clamp_bary(result.x.reshape(N, 2))

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

    # 1. Hexagonal Init and Local Refine
    hex_init = generate_hexagonal_init()
    refined_hex = local_minimize(hex_init, maxiter=2000)
    update_best(refined_hex)

    # 2. Differential Evolution for Global Search
    # DE is good for finding diverse basins.
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=False,
                mutation=(0.5, 1.0)
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = local_minimize(de_bary, maxiter=2000)
            update_best(refined_de)
            
            # Also try the best individual from DE population if available? 
            # DE returns the best. Let's assume result_de.x is the best.
            
        except Exception:
            pass

    # 3. Basin Hopping
    # Basin hopping is powerful for escaping local minima.
    # We run it from the current best solution.
    if best_bary is not None and time.time() < deadline - 15:
        try:
            # Define the step function for basin hopping
            def step(x):
                # x is flat array of size N*2
                bary = x.reshape(N, 2)
                # Perturb with Gaussian noise
                noise = np.random.normal(0, 0.05, bary.shape)
                new_bary = bary + noise
                new_bary = clamp_bary(new_bary)
                return new_bary.flatten()
            
            # Run basin hopping
            # We limit the number of iterations to respect time budget
            # min_steps and max_steps control the number of local minimizations
            result_bh = basinhopping(
                hard_min_obj,
                best_bary.flatten(),
                stepfunc=step,
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}},
                niter=20,  # Number of basin hopping steps
                T=1e-4,
                stepsize=0.05,
                seed=42
            )
            
            bh_bary = clamp_bary(result_bh.x.reshape(N, 2))
            update_best(bh_bary)
            
        except Exception:
            pass

    # 4. Final High-Precision Polish
    # Use SLSQP or Nelder-Mead with very tight tolerances on the best solution found.
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # SLSQP can be sensitive, but with a good start it's precise.
            # Let's try Nelder-Mead with very high iterations first, it's more robust.
            result_nm = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            update_best(clamp_bary(result_nm.x.reshape(N, 2)))
        except Exception:
            pass

        # Try SLSQP as a final touch
        if time.time() < deadline - 2:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
