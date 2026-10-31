import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 140.0  # Safety margin for 150s limit

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

def cart_to_bary(x, y):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v."""
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - w - v
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

def epigraph_obj_and_grad(z):
    """
    Epigraph formulation: variables are (u, v, t) where t is the target min area.
    Objective: minimize -t (maximize t)
    Constraints: all triangle areas >= t (i.e., area - t >= 0)
    
    We use the smooth approximation:
    f(z) = -t + penalty for violations
    
    Actually, let's use a proper constrained approach with SLSQP.
    Variables: z = [bary.flatten(), t]  (22 + 1 = 23 vars)
    Objective: -t
    Constraints: for each triangle i, area_i - t >= 0
    Bounds: bary in [0,1]^2, t >= 0
    """
    bary = z[:N*2].reshape(N, 2)
    t = z[N*2]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Objective: -t
    obj = -t
    
    # Gradient of objective w.r.t. bary is 0, w.r.t. t is -1
    grad = np.zeros_like(z)
    grad[N*2] = -1.0
    
    # Constraints: area_i - t >= 0 for all i
    # For SLSQP, we return constraint values (should be >= 0)
    # and their gradients
    con_vals = areas - t
    con_grads = []
    for i in range(len(areas)):
        g = np.zeros_like(z)
        # d(area_i)/dt = -1
        g[N*2] = -1.0
        # d(area_i)/d(bary) - we need the gradient of the determinant
        # This is complex, so let's use finite differences or a simpler approach
        con_grads.append(g)
    
    return obj, grad, con_vals

def soft_min_obj_grad(z, beta=100.0):
    """
    Smooth surrogate using log-sum-exp.
    z = [bary.flatten(), t]
    We want to maximize t such that soft_min(areas) >= t.
    Objective: -t
    But we can also just minimize the soft-min of (t - areas) to push t up.
    
    Simpler: minimize -t + lambda * sum(exp(beta*(t - area_i)))
    This is differentiable.
    """
    bary = z[:N*2].reshape(N, 2)
    t = z[N*2]
    
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas below t
    diff = t - areas
    # Soft penalty: sum(exp(beta * diff)) for diff > 0
    # To avoid overflow, clip
    diff_clipped = np.clip(diff, -50.0/beta, 50.0/beta)
    penalty = np.sum(np.exp(beta * diff_clipped))
    
    obj = -t + 0.1 * penalty
    
    # Gradient
    grad = np.zeros_like(z)
    grad[N*2] = -1.0 + 0.1 * beta * np.sum(np.exp(beta * diff_clipped))
    
    # Gradient w.r.t. bary: need d(area_i)/d(bary)
    # This requires the Jacobian of the determinant
    # For simplicity, we'll use a numerical approximation in the optimizer
    # or implement it properly
    
    return obj, grad

def simulated_annealing(rng, time_limit=10.0, beta_start=50.0, beta_end=500.0):
    """
    Simulated annealing with soft-min objective.
    """
    start_time = time.time()
    
    # Initialize with a structured configuration
    bary = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.5],
        [0.5, 0.25],
        [1/3, 1/3],
        [0.4, 0.35],
    ])
    bary = clamp_bary(bary)
    
    current_bary = bary.copy()
    current_min = get_min_area(current_bary)
    best_bary = current_bary.copy()
    best_min = current_min
    
    n_iter = 0
    total_iter_target = 5000
    temp = 0.05
    
    while time.time() - start_time < time_limit:
        # Anneal temperature
        progress = min(1.0, (time.time() - start_time) / time_limit)
        temp = 0.05 * (0.99 ** progress) + 0.001
        
        # Generate a perturbation
        n_iter += 1
        scale = temp
        perturbed = current_bary + rng.normal(0, scale, size=current_bary.shape)
        perturbed = clamp_bary(perturbed)
        
        new_min = get_min_area(perturbed)
        
        # Accept if better or with probability exp(-(new_min - current_min)/temp)
        if new_min > current_min:
            current_bary = perturbed
            current_min = new_min
            if new_min > best_min:
                best_bary = perturbed.copy()
                best_min = new_min
        else:
            delta = current_min - new_min
            if delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / max(temp, 1e-10))
            if rng.random() < accept_prob:
                current_bary = perturbed
                current_min = new_min
        
        # Batch check time
        if n_iter % 100 == 0:
            pass  # Just a checkpoint
    
    return best_bary, best_min

def slsqp_refine(bary_init, time_limit=5.0):
    """
    Refine using SLSQP with epigraph formulation.
    We'll use a simpler objective: minimize -min(areas) with constraints.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_val = get_min_area(bary)
    
    # Use a multi-start approach with SLSQP on the soft objective
    # Variables: bary flattened (22 vars)
    # Objective: negative soft-min of areas
    
    def soft_obj(x):
        b = x.reshape(N, 2)
        b = clamp_bary(b)
        areas = calc_areas_bary(b)
        if np.any(areas < 1e-15):
            return 1e10
        beta = 100.0
        scaled = -beta * areas
        max_scaled = np.max(scaled)
        if max_scaled > 700:
            return -np.min(areas)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta
        return -soft_min
    
    bounds = [(0, 1)] * (N * 2)
    
    try:
        result = minimize(
            soft_obj,
            bary.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_bary = clamp_bary(result.x.reshape(N, 2))
        new_val = get_min_area(new_bary)
        if new_val > current_val:
            bary = new_bary
            current_val = new_val
    except Exception:
        pass
    
    # Now try SLSQP with explicit constraints
    if time.time() - start_time < time_limit * 0.7:
        try:
            # Define constraints: area_i >= t, but we'll fix t and just maximize min
            # Actually, let's use the direct approach: minimize -min(areas)
            # with SLSQP, which can handle the non-smoothness via finite differences
            
            def hard_obj(x):
                b = x.reshape(N, 2)
                b = clamp_bary(b)
                areas = calc_areas_bary(b)
                return -np.min(areas)
            
            # Use SLSQP with bounds
            result2 = minimize(
                hard_obj,
                bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            new_bary2 = clamp_bary(result2.x.reshape(N, 2))
            new_val2 = get_min_area(new_bary2)
            if new_val2 > current_val:
                bary = new_bary2
                current_val = new_val2
        except Exception:
            pass
    
    return bary, current_val

def generate_structured_inits():
    """Generate structured initial configurations."""
    candidates = []
    
    # Config 1: 3 corners, 4 edge quarters, 4 interior
    cfg1 = np.array([
        [0.0, 0.0],   # corner A
        [1.0, 0.0],   # corner B
        [0.0, 1.0],   # corner C
        [0.25, 0.0],  # edge AB quarter
        [0.75, 0.0],  # edge AB three-quarter
        [0.0, 0.25],  # edge AC quarter
        [0.0, 0.75],  # edge AC three-quarter
        [0.25, 0.5],  # edge BC
        [0.5, 0.25],  # edge BC
        [1/3, 1/3],   # centroid
        [0.3, 0.3],   # interior
    ])
    candidates.append(clamp_bary(cfg1[:11]))
    
    # Config 2: Hexagonal-inspired
    cfg2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.33, 0.0],
        [0.67, 0.0],
        [0.0, 0.33],
        [0.0, 0.67],
        [0.17, 0.5],
        [0.5, 0.17],
        [0.33, 0.33],
        [0.5, 0.33],
    ])
    candidates.append(clamp_bary(cfg2))
    
    # Config 3: Dense boundary
    cfg3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.0],
        [0.8, 0.0],
        [0.0, 0.2],
        [0.0, 0.8],
        [0.1, 0.9],
        [0.9, 0.1],
        [1/3, 1/3],
        [0.4, 0.35],
    ])
    candidates.append(clamp_bary(cfg3))
    
    # Config 4: Symmetric interior
    cfg4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.25, 0.75],
        [1/3, 1/3],
        [0.2, 0.3],
        [0.5, 0.2],
        [0.3, 0.5],
        [0.4, 0.4],
    ])
    candidates.append(clamp_bary(cfg4))
    
    return candidates

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

    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated annealing global search
    if time.time() < deadline - 30:
        sa_bary, sa_val = simulated_annealing(rng, time_limit=20.0)
        update_best(sa_bary)
    
    # Phase 2: Refine best with SLSQP/L-BFGS-B
    if best_bary is not None and time.time() < deadline - 20:
        remaining = deadline - 20 - time.time()
        refined, val = slsqp_refine(best_bary, time_limit=min(10.0, remaining))
        update_best(refined)
    
    # Phase 3: Multi-start refinement from structured inits
    if time.time() < deadline - 10:
        inits = generate_structured_inits()
        for i, init in enumerate(inits):
            if time.time() > deadline - 10:
                break
            remaining = deadline - 10 - time.time()
            if remaining < 3:
                break
            try:
                refined, val = slsqp_refine(init, time_limit=min(5.0, remaining))
                update_best(refined)
            except Exception:
                continue
    
    # Phase 4: Perturbation and refinement of best
    if best_bary is not None and time.time() < deadline - 5:
        rng2 = np.random.default_rng(999)
        for iter_num in range(5):
            if time.time() > deadline - 5:
                break
            remaining = deadline - 5 - time.time()
            if remaining < 3:
                break
            scale = 0.1 * (0.7 ** iter_num)
            perturbed = best_bary + rng2.normal(0, scale, size=best_bary.shape)
            perturbed = clamp_bary(perturbed)
            try:
                refined, val = slsqp_refine(perturbed, time_limit=min(4.0, remaining))
                update_best(refined)
            except Exception:
                continue
    
    # Final polish
    if best_bary is not None and time.time() < deadline - 2:
        try:
            remaining = deadline - 2 - time.time()
            if remaining > 1:
                refined, val = slsqp_refine(best_bary, time_limit=remaining)
                update_best(refined)
        except Exception:
            pass
    
    if best_bary is None:
        best_bary = generate_structured_inits()[0]

    cart = bary_to_cart(best_bary)
    
    return cart
