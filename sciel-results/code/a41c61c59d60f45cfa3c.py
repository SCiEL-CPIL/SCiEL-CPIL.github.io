"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

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

def calc_signed_dets(bary):
    """
    Calculate signed determinants (areas) of all triangles.
    Returns array of shape (165,) with signed values.
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
    
    return det

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

def _delaunay_smooth_obj(bary_flat, beta=50.0, variance_weight=0.1):
    """
    Delaunay-aware smooth objective that penalizes variance in small triangle areas.
    This encourages more balanced critical triangles.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        soft_min = -np.min(areas)
    else:
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta
    
    # Identify small triangles (below median)
    median_area = np.median(areas)
    small_mask = areas < median_area
    if np.sum(small_mask) > 1:
        small_areas = areas[small_mask]
        variance_penalty = variance_weight * np.var(small_areas)
    else:
        variance_penalty = 0.0
    
    return -soft_min + variance_penalty

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

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    For 11 points, we can use a triangular arrangement.
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

def anneal_obj(bary_flat, T):
    """
    Simulated annealing objective: minimize -min_area + T * (some penalty or just the min area).
    Actually, for SA we want to accept worse moves. So we just return the value to minimize.
    Value = -min_area.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def simulated_annealing(bary_init, T0=0.1, T_end=0.001, n_iter=2000, deadline=None):
    """
    Simulated Annealing to escape local minima.
    """
    if deadline is not None and time.time() > deadline - 2:
        return bary_init
        
    current_bary = clamp_bary(bary_init)
    current_val = anneal_obj(current_bary.flatten(), T0)
    
    best_bary = current_bary.copy()
    best_val = current_val
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
            
        T = T0 * (T_end / T0) ** (i / n_iter)
        
        # Generate a random perturbation
        # Scale noise by T to allow larger jumps early on
        noise_scale = 0.05 * np.sqrt(T / T0) + 0.01
        perturbed_bary = current_bary + rng.normal(0, noise_scale, current_bary.shape)
        perturbed_bary = clamp_bary(perturbed_bary)
        
        new_val = anneal_obj(perturbed_bary.flatten(), T)
        
        delta = new_val - current_val
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_bary = perturbed_bary
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_bary = current_bary.copy()
                
    return best_bary

def _penalty_gradient_descent(bary_init, n_iter=5000, lr=0.001, deadline=None):
    """
    Penalty-based gradient descent that directly enforces the minimum area constraint.
    Uses a penalty function that heavily weights triangles below the current min area.
    """
    if deadline is not None and time.time() > deadline - 2:
        return bary_init
    
    current_bary = clamp_bary(bary_init.copy())
    current_min_area = np.min(calc_areas_bary(current_bary))
    
    best_bary = current_bary.copy()
    best_min_area = current_min_area
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        bary = current_bary.copy()
        areas = calc_areas_bary(bary)
        signed_dets = calc_signed_dets(bary)
        
        # Find the minimum area triangle
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Penalty: heavily weight triangles below current best min area
        # This creates a smooth gradient that pushes all small triangles up
        threshold = best_min_area * 0.95
        below_threshold = areas < threshold
        if np.any(below_threshold):
            # Gradient contribution from penalty
            penalty_weights = np.zeros(len(areas))
            penalty_weights[below_threshold] = (threshold - areas[below_threshold]) ** 2
        else:
            penalty_weights = np.zeros(len(areas))
        
        # Compute gradient of the penalty with respect to barycentric coords
        # This is a numerical gradient approximation
        grad = np.zeros((N, 2))
        eps = 1e-6
        
        for i_pt in range(N):
            for coord in range(2):
                bary_plus = bary.copy()
                bary_plus[i_pt, coord] += eps
                bary_plus = clamp_bary(bary_plus)
                
                bary_minus = bary.copy()
                bary_minus[i_pt, coord] -= eps
                bary_minus = clamp_bary(bary_minus)
                
                areas_plus = calc_areas_bary(bary_plus)
                areas_minus = calc_areas_bary(bary_minus)
                
                # Compute penalty values
                pen_plus = np.sum((threshold - np.minimum(areas_plus, threshold)) ** 2)
                pen_minus = np.sum((threshold - np.minimum(areas_minus, threshold)) ** 2)
                
                grad[i_pt, coord] = (pen_plus - pen_minus) / (2 * eps)
        
        # Add a small regularization to keep points in the interior
        u = bary[:, 0]
        v = bary[:, 1]
        w = 1.0 - u - v
        
        # Boundary penalty gradient
        boundary_eps = 1e-4
        boundary_grad_u = np.zeros(N)
        boundary_grad_v = np.zeros(N)
        
        boundary_grad_u[u < boundary_eps] = -(boundary_eps - u[u < boundary_eps])
        boundary_grad_v[v < boundary_eps] = -(boundary_eps - v[v < boundary_eps])
        boundary_grad_u[w < boundary_eps] = (boundary_eps - w[w < boundary_eps])
        boundary_grad_v[w < boundary_eps] = (boundary_eps - w[w < boundary_eps])
        
        grad[:, 0] += 10.0 * boundary_grad_u
        grad[:, 1] += 10.0 * boundary_grad_v
        
        # Adaptive learning rate
        adaptive_lr = lr * (0.5 + 0.5 * (1.0 / (1.0 + i / 100.0)))
        
        # Update
        new_bary = bary - adaptive_lr * grad
        new_bary = clamp_bary(new_bary)
        
        new_min_area = np.min(calc_areas_bary(new_bary))
        
        # Accept if improved or with small probability (to escape local minima)
        if new_min_area > current_min_area or (i < 1000 and rng.random() < 0.1):
            current_bary = new_bary
            current_min_area = new_min_area
            
            if current_min_area > best_min_area:
                best_min_area = current_min_area
                best_bary = current_bary.copy()
    
    return best_bary

def _local_polish_bary(bary, max_iter=200, deadline=None):
    """
    Local polish using coordinate descent with small steps.
    For each point, try small perturbations in u and v directions.
    """
    if deadline is not None and time.time() > deadline - 2:
        return bary
    
    current_bary = clamp_bary(bary.copy())
    current_val = -hard_min_obj(current_bary.flatten())
    
    rng = np.random.default_rng(123)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        improved = False
        for i in range(N):
            # Try small steps in u direction
            for du in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01, 0.002, -0.002]:
                trial = current_bary.copy()
                trial[i, 0] += du
                trial = clamp_bary(trial)
                trial_val = -hard_min_obj(trial.flatten())
                if trial_val > current_val:
                    current_bary = trial
                    current_val = trial_val
                    improved = True
                    break
            
            # Try small steps in v direction
            for dv in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01, 0.002, -0.002]:
                trial = current_bary.copy()
                trial[i, 1] += dv
                trial = clamp_bary(trial)
                trial_val = -hard_min_obj(trial.flatten())
                if trial_val > current_val:
                    current_bary = trial
                    current_val = trial_val
                    improved = True
                    break
        
        if not improved:
            break
    
    return current_bary

def construct():
    start_time = time.time()
    # Evaluator kills after 150s. We use 140s deadline to be safe.
    deadline = start_time + 140.0
    
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

    def try_optimize_l_bfgs(bary_init, beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300, use_delaunay=False):
        """
        Two-stage L-BFGS optimization.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            # Stage 1: Exploration with smoother landscape
            if use_delaunay:
                def obj1(x):
                    return _delaunay_smooth_obj(x, beta=beta1, variance_weight=0.1)
            else:
                def obj1(x):
                    return soft_min_obj(x, beta=beta1)
            
            bounds = [(0, 1)] * (N * 2)
            
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 5:
                update_best(result1.x.reshape(N, 2))
                return

            # Stage 2: Exploitation with sharper landscape
            def obj2(x):
                return soft_min_obj(x, beta=beta2)
            
            result2 = minimize(
                obj2,
                result1.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            update_best(result2.x.reshape(N, 2))
                
        except Exception:
            pass

    def try_optimize_nm(bary_init, maxiter=2000):
        """
        Nelder-Mead optimization on hard min.
        """
        if time.time() > deadline - 10:
            return
        
        try:
            result = minimize(
                hard_min_obj,
                bary_init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    def try_penalty_gd(bary_init, n_iter=5000, lr=0.001):
        """
        Penalty-based gradient descent.
        """
        if time.time() > deadline - 15:
            return
        
        try:
            result = _penalty_gradient_descent(bary_init, n_iter=n_iter, lr=lr, deadline=deadline)
            update_best(result)
        except Exception:
            pass

    # --- Phase 1: Multi-start L-BFGS ---
    
    # 1. Structured Init
    hex_init = generate_hexagonal_init()
    try_optimize_l_bfgs(hex_init, beta1=50.0, beta2=200.0, maxiter1=300, maxiter2=400)
    
    # 2. Random Inits with Delaunay-aware objective
    num_random_starts = 8
    for i in range(num_random_starts):
        if time.time() > deadline - 50:
            break
        rand_init = generate_random_init(seed=100 + i * 7)
        use_del = (i % 2 == 0)  # Alternate between soft-min and Delaunay-aware
        try_optimize_l_bfgs(rand_init, beta1=50.0, beta2=150.0, maxiter1=150, maxiter2=200, use_delaunay=use_del)
        
    # --- Phase 2: Penalty-based gradient descent ---
    if best_bary is not None:
        try_penalty_gd(best_bary, n_iter=5000, lr=0.001)
        
        # Try from a few random starts
        for i in range(3):
            if time.time() > deadline - 30:
                break
            rand_start = generate_random_init(seed=500 + i * 13)
            # Quick optimize first
            try_optimize_l_bfgs(rand_start, beta1=50.0, beta2=150.0, maxiter1=100, maxiter2=100)
            if time.time() > deadline - 25:
                break
            # Then penalty GD
            try_penalty_gd(best_bary, n_iter=3000, lr=0.0008)

    # --- Phase 3: Simulated Annealing to escape local minima ---
    if best_bary is not None and time.time() < deadline - 20:
        sa_result = simulated_annealing(best_bary, T0=0.2, T_end=0.01, n_iter=2000, deadline=deadline)
        update_best(sa_result)

    # --- Phase 4: Perturb Best and Re-optimize (Local Refinement) ---
    if best_bary is not None:
        for i in range(4):
            if time.time() > deadline - 20:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=800 + i)
            use_del = (i % 2 == 1)
            try_optimize_l_bfgs(perturbed, beta1=100.0, beta2=300.0, maxiter1=200, maxiter2=300, use_delaunay=use_del)
    
    # --- Phase 5: Nelder-Mead Polish ---
    if best_bary is not None and time.time() < deadline - 15:
        try_optimize_nm(best_bary, maxiter=3000)
        
        # Re-perturb and NM if time allows
        for i in range(2):
            if time.time() > deadline - 12:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.01, seed=900 + i)
            try_optimize_nm(perturbed, maxiter=2000)

    # --- Phase 6: Penalty GD Polish ---
    if best_bary is not None and time.time() < deadline - 10:
        try_penalty_gd(best_bary, n_iter=3000, lr=0.0005)

    # --- Phase 7: Local Polish with Coordinate Descent ---
    if best_bary is not None and time.time() < deadline - 8:
        polished = _local_polish_bary(best_bary, max_iter=150, deadline=deadline)
        update_best(polished)
        
        # Additional polish rounds
        for i in range(3):
            if time.time() > deadline - 5:
                break
            polished = _local_polish_bary(best_bary, max_iter=50, deadline=deadline)
            update_best(polished)

    # --- Phase 8: Final NM Polish ---
    if best_bary is not None and time.time() < deadline - 4:
        try:
            result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            update_best(result.x.reshape(N, 2))
        except Exception:
            pass

    # Ensure we have a valid result
    if best_bary is None:
        best_bary = generate_random_init(seed=1)
        
    # Final clamp
    best_bary = clamp_bary(best_bary)
    
    # Convert to Cartesian
    cart_pts = bary_to_cart(best_bary)
    
    return cart_pts

# Test run if executed directly
if __name__ == "__main__":
    pts = construct()
    print("Points:")
    print(pts)
    # Check validity
    for p in pts:
        if p[0] < -1e-9 or p[1] < -1e-9 or p[0] > 1 + 1e-9:
            print("Invalid point:", p)
        if p[1] > np.sqrt(3) * p[0] + 1e-9 or p[1] > np.sqrt(3) * (1 - p[0]) + 1e-9:
            print("Invalid point (triangle bound):", p)
    # Calculate score
    bary = np.zeros((N, 2))
    # Inverse of bary_to_cart for debugging score
    # x = v + 0.5*w, y = w*H => w = y/H, v = x - 0.5*w, u = 1 - v - w
    w = pts[:, 1] / H
    v = pts[:, 0] - 0.5 * w
    u = 1 - v - w
    bary = np.column_stack([u, v])
    areas = calc_areas_bary(bary)
    print("Min Area (normalized):", np.min(areas))
    print("Max Area (normalized):", np.max(areas))
