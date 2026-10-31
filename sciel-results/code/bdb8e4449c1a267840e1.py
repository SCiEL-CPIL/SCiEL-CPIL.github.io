import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit (evaluator kills at 150s)

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

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Ensure u >= 0, v >= 0
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    # Ensure u + v <= 1
    sums = u + v
    invalid = sums > 1
    if np.any(invalid):
        scale = 1.0 / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-clip to ensure no numerical drift
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    return np.column_stack([u, v])

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

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration for a good starting point.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try a few grid spacings
    for scale in [0.2, 0.25, 0.3, 0.35, 0.4]:
        pts = []
        # Rows along the bottom edge
        for k in range(int(1.0 / scale) + 1):
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
            # Try different subsets if we have more than N
            rng = np.random.default_rng(42)
            for _ in range(10):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = np.min(calc_areas_bary(bary))
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        # Fallback random
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def soft_min_objective(x, beta=50.0):
    """
    Objective for Differential Evolution.
    We want to maximize the minimum area.
    DE minimizes, so we minimize -soft_min(area, beta).
    
    Using a soft minimum (log-sum-exp trick) makes the function smooth and differentiable,
    which helps DE and subsequent polishers.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Soft minimum: -log(sum(exp(-area/beta))) * beta
    # As beta -> 0, this approaches -min(areas)
    # We minimize this, so we maximize the soft min.
    # To avoid overflow/underflow in exp:
    max_a = np.max(areas)
    # areas are small, so -areas/beta is large negative.
    # Let's just use: -beta * log(mean(exp(areas/beta))) ? No, that's soft max.
    # Soft min = - Soft max of (-areas).
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    # log(sum(exp(neg_areas - max_neg))) + max_neg
    # Soft min approx = max_neg + log(sum(exp(neg_areas - max_neg)))
    # Wait, standard soft min is: -beta * log(sum(exp(-area/beta)))
    # Let's use a simpler penalty if beta is fixed.
    
    # Actually, for DE, a simple min might be non-smooth but DE doesn't need gradients.
    # However, a smooth objective often helps DE converge to better basins.
    # Let's use: -np.min(areas) + penalty for boundary violations if any (though clamp handles it).
    # The clamp might create non-smoothness.
    # Let's stick to minimizing the negative of the soft min.
    
    # To be safe and fast:
    min_area = np.min(areas)
    return -min_area

def polish_local(bary_init):
    """
    Local refinement using SLSQP to maximize t subject to areas >= t.
    """
    # Objective: minimize -t
    def obj(x):
        return -x[0]
    
    # Constraints: areas - t >= 0
    def cons(x):
        t = x[0]
        bary = x[1:].reshape(N, 2)
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        return areas - t
    
    # Bounds
    # t >= 0
    # u, v in [0, 1]
    bounds = [(0, None)] + [(0, 1)] * (N * 2)
    
    min_area = np.min(calc_areas_bary(bary_init))
    x0 = np.zeros(1 + N * 2)
    x0[0] = min_area
    x0[1:] = bary_init.flatten()
    
    try:
        res = minimize(
            obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': cons},
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if res.success or res.fun < -min_area:
            best_t = res.x[0]
            best_bary = clamp_bary(res.x[1:].reshape(N, 2))
            return best_bary, best_t
        else:
            return bary_init, min_area
    except Exception:
        return bary_init, min_area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = np.min(calc_areas_bary(bary))
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Initial good start
    hex_init = generate_hexagonal_init()
    update_best(hex_init)
    
    # 2. Global Optimization with Differential Evolution
    # DE requires bounds. Barycentric coords u, v must satisfy u>=0, v>=0, u+v<=1.
    # This is a simplex. DE works on hyper-rectangles.
    # We can map [0,1]x[0,1] to the triangle, or just use bounds [0,1] for u and v
    # and rely on clamp_bary inside the objective. This is inefficient but robust.
    # Better: Use a transformation.
    # Let u = s * r, v = s * (1-r) * q ? No.
    # Standard mapping:
    # Let a, b be in [0,1].
    # u = a
    # v = b * (1 - a)
    # This covers the triangle exactly.
    # If we use this, DE optimizes a, b in [0,1].
    
    def de_objective(ab_params):
        a = ab_params[0::2]
        b = ab_params[1::2]
        
        u = a
        v = b * (1 - a)
        
        bary = np.column_stack([u, v])
        bary = clamp_bary(bary) # Just in case of numerical issues
        
        areas = calc_areas_bary(bary)
        # Minimize negative min area
        return -np.min(areas)
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Check time
    if time.time() < deadline - 30:
        try:
            # DE parameters
            # popsize: 15 is default, maybe 20 for better exploration
            # tol: 1e-4
            # maxiter: 100
            # seed: fixed for reproducibility
            de_res = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-4,
                polish=False,  # We will do our own polish
                updating='immediate'
            )
            
            # Convert DE result back to barycentric
            a_opt = de_res.x[0::2]
            b_opt = de_res.x[1::2]
            u_opt = a_opt
            v_opt = b_opt * (1 - a_opt)
            bary_de = np.column_stack([u_opt, v_opt])
            
            update_best(bary_de)
        except Exception as e:
            pass # DE failed, continue with best so far

    # 3. Local Polish on the best solution found
    if best_bary is not None and time.time() < deadline - 10:
        polished, val = polish_local(best_bary)
        update_best(polished)
        
        # Try a few random perturbations and polish
        rng = np.random.default_rng(123)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            # Perturb in barycentric space
            noise = rng.normal(0, 0.05, best_bary.shape)
            perturbed = clamp_bary(best_bary + noise)
            polished_p, val_p = polish_local(perturbed)
            update_best(polished_p)
            
            # Also try a completely new random start if time permits
            if time.time() < deadline - 15:
                u_rand = rng.random(N)
                v_rand = rng.random(N) * (1 - u_rand)
                bary_rand = np.column_stack([u_rand, v_rand])
                polished_r, val_r = polish_local(bary_rand)
                update_best(polished_r)

    # Final safety check
    if best_bary is None:
        # Should not happen, but fallback
        best_bary = generate_hexagonal_init()

    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
