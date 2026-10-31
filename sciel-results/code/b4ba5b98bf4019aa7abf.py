import numpy as np
from scipy.optimize import differential_evolution, minimize
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
AREA_TOTAL = np.sqrt(3) / 4.0

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
    """Clamp barycentric coordinates to the valid triangle region."""
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

def get_min_area(bary):
    """Return the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def soft_min_obj(bary_flat, temp=0.01):
    """
    Objective function for optimization using a soft minimum (log-sum-exp)
    to make the function smooth and differentiable.
    Minimizes -log(sum(exp(-area/temp))) which approximates -min(area).
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    # Check validity (should be handled by clamp, but good for safety)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(w < -1e-6):
        return 1e10

    areas = calc_areas_bary(bary)
    
    # Soft min approximation
    # We want to maximize min(area), so minimize -min(area).
    # -min(a) = log(sum(exp(-a))) is not quite right. 
    # min(a) approx -log(sum(exp(-a/temp))) * temp?
    # Actually, -min(a) = - ( -1/temp * log(sum(exp(-a/temp))) ) = 1/temp * log(sum(exp(-a/temp)))
    # So we minimize 1/temp * log(sum(exp(-areas/temp)))
    
    # To avoid overflow, subtract max
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    
    return (1.0 / temp) * log_sum_exp

def hard_min_obj(bary_flat):
    """Objective function for final evaluation (non-smooth)."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def refine_local(bary_init, time_limit=5.0):
    """
    Local refinement using L-BFGS-B on the soft-min objective, 
    then a final polish on the hard-min objective.
    """
    start_time = time.time()
    best_bary = bary_init.copy()
    best_val = get_min_area(best_bary)
    
    # Phase 1: L-BFGS-B on soft min (smooth)
    try:
        res = minimize(
            soft_min_obj,
            bary_init.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
            args=(0.01,)  # Temperature for soft min
        )
        if res.fun < 1e9: # Check if it failed
            candidate = clamp_bary(res.x.reshape(N, 2))
            val = get_min_area(candidate)
            if val > best_val:
                best_bary = candidate
                best_val = val
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead on hard min (non-smooth) for final polish
    if time.time() < start_time + time_limit:
        try:
            res2 = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            candidate = clamp_bary(res2.x.reshape(N, 2))
            val = get_min_area(candidate)
            if val > best_val:
                best_bary = candidate
                best_val = val
        except Exception:
            pass
            
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Define bounds for barycentric coordinates
    # u in [0, 1], v in [0, 1], but u+v <= 1.
    # L-BFGS-B and DE work best with box constraints.
    # We can map the triangle to a square [0,1]x[0,1] and reject/clamp invalid points.
    # Or, we can define bounds for u and v such that the mapping stays valid?
    # Simplest: Use bounds [0, 1] for u and v, and rely on clamp_bary to project back.
    # However, DE/L-BFGS-B might waste time outside.
    # Better: Parameterize with 2 variables per point, but constrain the domain.
    # Since we are using barycentric (u,v), the valid region is a triangle.
    # We can use a transformation to a box:
    # Let u' in [0,1], v' in [0,1].
    # u = u' * (1 - v')
    # v = v'
    # This maps the unit square to the triangle.
    # Jacobian is non-singular.
    # But our optimizers work on the variables directly.
    # If we optimize u' and v', the objective is non-linear in u', v'.
    # Let's stick to optimizing u, v directly with bounds [0, 1] and let the
    # soft-min objective penalize invalid points (or clamp them).
    # Clamping makes the function non-differentiable at the boundary.
    # To make it smooth, we can use a barrier function or just rely on the fact
    # that the optimum is likely interior or on the boundary, and L-BFGS-B handles
    # bounds well if we define them correctly.
    
    # Actually, for L-BFGS-B, we can define bounds:
    # u_i >= 0, v_i >= 0, u_i + v_i <= 1.
    # L-BFGS-B only supports box bounds (lower_i <= x_i <= upper_i).
    # It does NOT support coupled constraints like u_i + v_i <= 1 directly.
    # So we MUST use the transformation or penalize.
    
    # Transformation approach for the optimizer variables:
    # Optimize z = [z_0, ..., z_21] where z_{2i} = u'_i, z_{2i+1} = v'_i
    # u_i = z_{2i} * (1 - z_{2i+1})
    # v_i = z_{2i+1}
    # Bounds: z in [0, 1].
    
    def z_to_bary(z_flat):
        z = z_flat.reshape(N, 2)
        u_prime = z[:, 0]
        v_prime = z[:, 1]
        u = u_prime * (1.0 - v_prime)
        v = v_prime
        return np.column_stack([u, v])

    def obj_z(z_flat):
        bary = z_to_bary(z_flat)
        # No need to clamp if transformation is perfect, but add small eps for safety
        u = bary[:, 0]
        v = bary[:, 1]
        # Ensure numerical stability
        u = np.clip(u, 1e-9, 1-1e-9)
        v = np.clip(v, 1e-9, 1-1e-9)
        # Re-normalize if needed? The transformation guarantees u+v <= 1.
        # u + v = u'(1-v') + v' = u' - u'v' + v'.
        # If u'=1, v'=0 -> u=1, v=0, sum=1.
        # If u'=0, v'=1 -> u=0, v=1, sum=1.
        # If u'=1, v'=1 -> u=0, v=1, sum=1.
        # It stays in the triangle.
        bary_clean = np.column_stack([u, v])
        return soft_min_obj(bary_clean.flatten(), temp=0.01)

    bounds_z = [(0.0, 1.0)] * (2 * N)

    # 2. Global Search with Differential Evolution
    # DE is robust and handles the non-convexity well.
    # We use a moderate population size and generation count to balance exploration and time.
    
    try:
        # Check time
        if time.time() > deadline - 30:
            raise TimeoutError()
            
        de_result = differential_evolution(
            obj_z,
            bounds_z,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            polish=False, # We will polish manually
            init='sobol' # Latin hypercube sampling for better initial spread
        )
        
        if de_result.success or de_result.fun < 1e9:
            best_z = de_result.x
            best_bary = z_to_bary(best_z)
            update_best(best_bary)
    except Exception:
        pass

    # 3. Local Refinement of the best candidate
    if best_bary is not None:
        # We need to refine in the z-space or bary-space?
        # Refining in bary-space with L-BFGS-B is tricky due to the u+v<=1 constraint.
        # Let's refine in z-space using L-BFGS-B on the soft-min objective.
        
        if time.time() < deadline - 10:
            try:
                # Initial z for L-BFGS-B
                # Invert z_to_bary to get z from best_bary?
                # v = v_prime -> v_prime = v
                # u = u_prime * (1 - v_prime) -> u_prime = u / (1 - v_prime)
                # If v_prime is close to 1, division by zero.
                # Let's just use the de_result.x as the start for L-BFGS-B.
                
                start_z = de_result.x if 'de_result' in locals() and de_result.x is not None else np.random.random(2*N)
                
                res_local = minimize(
                    obj_z,
                    start_z,
                    method='L-BFGS-B',
                    bounds=bounds_z,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res_local.fun < 1e9:
                    candidate_bary = z_to_bary(res_local.x)
                    update_best(candidate_bary)
                    
                    # Final Nelder-Mead polish in bary space (handling the shape directly)
                    if time.time() < deadline - 5:
                        refined = refine_local(candidate_bary, time_limit=min(5.0, deadline - time.time() - 2))
                        update_best(refined)
                        
            except Exception:
                pass

    # 4. Fallback: If DE failed or didn't find anything good, try a few random inits with local polish
    if best_val < 0.01: # Very low score, likely failure
        rng = np.random.default_rng(42)
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            # Random barycentric points
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            bary_init = np.column_stack([u, v])
            refined = refine_local(bary_init, time_limit=min(2.0, deadline - time.time() - 5))
            update_best(refined)

    if best_bary is None:
        # Absolute fallback
        # Centroid
        best_bary = np.full((N, 2), 1/3)
        
    cart = bary_to_cart(best_bary)
    
    return cart
