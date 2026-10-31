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

# Pre-compute area coefficients for fast evaluation
# For barycentric coords, area of triangle (i,j,k) = |det| where
# det = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
# We'll use the precomputed indices for vectorized computation

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

def calc_areas_bary_signed(bary):
    """Calculate signed determinants (without abs) for gradient computation."""
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
    
    return det

def clamp_bary(bary, eps=1e-10):
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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def smooth_min_obj(bary_flat, beta):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this is equivalent to maximizing the soft-min area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Handle zero areas to avoid log(0)
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 1e10
        
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        # Fallback to hard min if beta is too large causing overflow
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def epigraph_obj(x, beta=200.0):
    """
    Epigraph formulation: minimize -t subject to areas >= t.
    x = [bary_flat (N*2), t]
    Uses smooth constraint violation penalty.
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Smooth constraint: penalize areas that are below t
    # Use log-sum-exp for smooth max of (t - areas)
    violations = t - areas
    max_viol = np.max(violations)
    
    if max_viol > 700 / beta:
        # Overflow protection
        return 1e10
    
    # Soft max of violations
    shifted = beta * violations
    max_shifted = np.max(shifted)
    if max_shifted > 700:
        return 1e10
    
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_max_viol = log_sum / beta
    
    # Objective: -t + penalty * soft_max_violation
    # If all areas >= t, soft_max_viol should be small
    penalty = 10.0
    return -t + penalty * soft_max_viol

def epigraph_obj_with_grad(x, beta=200.0):
    """
    Epigraph objective with analytical gradient.
    x = [bary_flat (N*2), t]
    """
    bary_flat = x[:N*2].copy()
    t = x[N*2]
    
    bary = bary_flat.reshape(N, 2)
    bary_clamped = clamp_bary(bary)
    
    # Compute areas
    u = bary_clamped[:, 0]
    v = bary_clamped[:, 1]
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
    
    # Signed determinants
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    areas = np.abs(det)
    
    # Violations
    violations = t - areas
    
    # Soft max via log-sum-exp
    shifted = beta * violations
    max_shifted = np.max(shifted)
    if max_shifted > 700:
        return 1e10, None
    
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum = max_shifted + np.log(sum_exp)
    soft_max_viol = log_sum / beta
    
    penalty = 10.0
    obj = -t + penalty * soft_max_viol
    
    # Gradient computation
    # d(soft_max_viol)/d(violations) = exp(shifted) / sum_exp
    weights = exp_shifted / sum_exp  # (165,)
    
    # d(soft_max_viol)/d(t) = sum(weights) = 1
    # d(soft_max_viol)/d(areas) = -weights
    # d(areas)/d(bary) = sign(det) * d(det)/d(bary)
    
    # d(det)/d(u_i) for triangle i
    # det = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
    # where w = 1-u-v
    
    # Let's compute gradients numerically for reliability
    grad = np.zeros_like(x)
    grad[N*2] = -1.0 + penalty * 1.0  # d(-t + penalty*soft_max)/dt = -1 + penalty*sum(weights) = -1 + penalty
    
    # For barycentric coordinates, use finite differences for gradient
    # This is more robust than analytical
    eps_fd = 1e-7
    grad_bary = np.zeros(N*2)
    
    for idx in range(N*2):
        x_plus = x.copy()
        x_plus[idx] += eps_fd
        obj_plus = epigraph_obj(x_plus, beta)
        
        x_minus = x.copy()
        x_minus[idx] -= eps_fd
        obj_minus = epigraph_obj(x_minus, beta)
        
        grad_bary[idx] = (obj_plus - obj_minus) / (2 * eps_fd)
    
    grad[:N*2] = grad_bary
    grad[N*2] = -1.0 + penalty * np.sum(weights)
    
    return obj, grad

def generate_hexagonal_init():
    """
    Generate a hexagonal-like lattice initialization.
    """
    pts = []
    
    # Row 0: 4 points along the bottom
    for i in range(4):
        pts.append([i / 3.0, 0.0])
    
    # Row 1: 3 points offset
    for i in range(3):
        pts.append([(i + 0.5) / 3.0, H / 3.0])
    
    # Row 2: 2 points
    for i in range(2):
        pts.append([(i + 1.0) / 3.0, 2 * H / 3.0])
    
    # Row 3: 1 point at top
    pts.append([0.5, H])
    
    pts = np.array(pts)
    # Convert to barycentric
    # x = v + 0.5*w, y = w*H
    # w = y/H
    # v = x - 0.5*w
    # u = 1 - v - w
    bary_pts = []
    for p in pts:
        x, y = p
        w = y / H
        v = x - 0.5 * w
        u = 1 - v - w
        bary_pts.append([u, v])
        
    return np.array(bary_pts)

def generate_random_init(rng):
    """Generate random initial configuration in barycentric coords."""
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_symmetric_init():
    """
    Generate a symmetric configuration exploiting the triangle's 3-fold symmetry.
    Place points on orbits of the dihedral group D3.
    """
    pts = []
    
    # Center point
    pts.append([1/3, 1/3])
    
    # 3 points at 1/3 height, one on each edge midpoint direction
    # In barycentric: (1/3, 1/3, 1/3) is center
    # Points at distance r from center along the 3 symmetry axes
    
    # Axis 1: towards vertex 0 (u-axis)
    r = 0.25
    pts.append([1/3 + r, 1/3 - r/2])  # towards (1,0,0)
    pts.append([1/3 - r/2, 1/3 + r, 1/3 - r/2])  # towards (0,1,0) - need to convert
    pts.append([1/3 - r/2, 1/3 - r/2, 1/3 + r])  # towards (0,0,1)
    
    # Actually, let's use Cartesian and convert
    pts_cart = [[0.5, H/3]]  # center-ish
    
    # 3 points forming an equilateral triangle around center
    for angle in [0, 2*np.pi/3, 4*np.pi/3]:
        x = 0.5 + 0.2 * np.cos(angle)
        y = H/3 + 0.2 * np.sin(angle) * (H/0.5)  # scale y
        pts_cart.append([x, y])
    
    # 3 more points at different radius
    for angle in [np.pi/3, np.pi, 5*np.pi/3]:
        x = 0.5 + 0.35 * np.cos(angle)
        y = H/3 + 0.35 * np.sin(angle) * (H/0.5)
        pts_cart.append([x, y])
    
    # Convert to barycentric
    bary_pts = []
    for x, y in pts_cart:
        w = y / H
        v = x - 0.5 * w
        u = 1 - v - w
        bary_pts.append([u, v])
    
    return np.array(bary_pts)

def refine_local(bary_init, time_limit=10.0):
    """
    Refine a configuration using smooth annealing, epigraph optimization, and hard polish.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth annealing
    # Start with low beta (smooth) and increase to high beta (sharp)
    betas = [10, 50, 100, 200, 400]
    
    for beta in betas:
        if time.time() - start_time > time_limit * 0.5:
            break
            
        try:
            # Use L-BFGS-B for the smooth objective
            result = minimize(
                smooth_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            
            # Check if it improved
            curr_val = -hard_min_obj(bary_current.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            
            if new_val > curr_val:
                bary_current = new_bary
            else:
                # If no improvement, try SLSQP with current beta
                result2 = minimize(
                    smooth_min_obj,
                    bary_current.flatten(),
                    args=(beta,),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-13}
                )
                new_bary2 = clamp_bary(result2.x.reshape(N, 2))
                new_val2 = -hard_min_obj(new_bary2.flatten())
                if new_val2 > curr_val:
                    bary_current = new_bary2
                    
        except Exception:
            pass
        
    # Stage 2: Epigraph optimization
    # This uses the auxiliary variable t to create a smoother landscape
    if time.time() - start_time < time_limit * 0.75:
        try:
            # Initialize t with current min area
            areas = calc_areas_bary(bary_current)
            t_init = np.min(areas) * 0.95  # Slightly below to be feasible
            
            x_init = np.concatenate([bary_current.flatten(), [t_init]])
            
            # Bounds for epigraph problem
            epigraph_bounds = [(0, 1)] * (N * 2) + [(0, 1)]  # t in [0, 1]
            
            # Try multiple betas for epigraph
            for beta in [100, 200, 400]:
                if time.time() - start_time > time_limit * 0.85:
                    break
                    
                try:
                    result_ep = minimize(
                        epigraph_obj,
                        x_init,
                        args=(beta,),
                        method='L-BFGS-B',
                        bounds=epigraph_bounds,
                        options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    
                    new_bary = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
                    curr_val = -hard_min_obj(bary_current.flatten())
                    new_val = -hard_min_obj(new_bary.flatten())
                    
                    if new_val > curr_val:
                        bary_current = new_bary
                        x_init = np.concatenate([bary_current.flatten(), [t_init]])
                        
                except Exception:
                    pass
                    
        except Exception:
            pass
        
    # Stage 3: Hard min polish with Nelder-Mead
    if time.time() - start_time < time_limit * 0.9:
        try:
            result_nm = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-17, 'adaptive': True}
            )
            bary_current = clamp_bary(result_nm.x.reshape(N, 2))
        except Exception:
            pass
            
    # Stage 4: Final SLSQP polish on hard min
    if time.time() - start_time < time_limit:
        try:
            result_slsqp = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15}
            )
            bary_current = clamp_bary(result_slsqp.x.reshape(N, 2))
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

    # 1. Generate Initial Candidates
    candidates = []
    
    # Hexagonal
    hex_init = generate_hexagonal_init()
    candidates.append(hex_init)
    
    # Symmetric
    sym_init = generate_symmetric_init()
    candidates.append(sym_init)
    
    # Perturbed Hexagonal
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = hex_init + rng.normal(0, 0.05, size=hex_init.shape)
        candidates.append(clamp_bary(perturbed))
        
    # Perturbed Symmetric
    for _ in range(2):
        perturbed_sym = sym_init + rng.normal(0, 0.04, size=sym_init.shape)
        candidates.append(clamp_bary(perturbed_sym))
        
    # Random
    for _ in range(4):
        candidates.append(generate_random_init(rng))
        
    # 2. Global Search with Differential Evolution
    # Use smooth objective for DE to navigate better
    if time.time() < deadline - 50:
        try:
            def de_obj(x):
                return smooth_min_obj(x, beta=50.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=40,
                popsize=15,
                tol=1e-5,
                polish=False,
                init='sobol'
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            candidates.append(de_bary)
        except Exception:
            pass
            
    # 3. Local Refinement of Candidates
    n_cands = len(candidates)
    # Allocate time based on remaining time
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 25:
            break
            
        remaining = deadline - 25 - time.time()
        if remaining < 5:
            break
            
        # Prioritize hexagonal, symmetric, and DE results
        if i < 5:
            t_limit = min(12.0, remaining)
        elif i < 8:
            t_limit = min(8.0, remaining)
        else:
            t_limit = min(5.0, remaining)
            
        try:
            refined = refine_local(cand, time_limit=t_limit)
            update_best(refined)
        except Exception:
            continue

    # 4. Second DE run if time permits
    if time.time() < deadline - 35:
        try:
            def de_obj2(x):
                return smooth_min_obj(x, beta=100.0)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_de2 = differential_evolution(
                de_obj2,
                bounds,
                seed=123,
                maxiter=25,
                popsize=12,
                tol=1e-5,
                polish=False
            )
            
            de_bary2 = clamp_bary(result_de2.x.reshape(N, 2))
            remaining = deadline - 25 - time.time()
            if remaining > 10:
                refined_de2 = refine_local(de_bary2, time_limit=min(15.0, remaining))
                update_best(refined_de2)
        except Exception:
            pass

    # 5. Final Polish on Best Solution
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds = [(0, 1)] * (N * 2)
            # High precision polish with Nelder-Mead
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        # Additional SLSQP polish if time remains
        if best_bary is not None and time.time() < deadline - 5:
            try:
                result_slsqp_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_slsqp_final.x.reshape(N, 2)))
            except Exception:
                pass
            
    if best_bary is None:
        # Fallback
        best_bary = generate_hexagonal_init()

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
