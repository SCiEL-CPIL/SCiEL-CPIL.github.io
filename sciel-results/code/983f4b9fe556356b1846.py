import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin

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

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric coords (u, v)."""
    # Inverse of bary_to_cart
    # x = v + 0.5 * (1 - u - v) = 0.5 - 0.5u + 0.5v
    # y = H * (1 - u - v)
    # y/H = 1 - u - v  => u + v = 1 - y/H
    # x = 0.5 - 0.5u + 0.5v
    # Let s = u + v. Then v = s - u.
    # x = 0.5 - 0.5u + 0.5(s - u) = 0.5 + 0.5s - u
    # u = 0.5 + 0.5s - x
    # v = s - u = s - (0.5 + 0.5s - x) = 0.5s + x - 0.5
    # s = 1 - y/H
    s = 1.0 - cart[:, 1] / H
    u = 0.5 + 0.5 * s - cart[:, 0]
    v = 0.5 * s + cart[:, 0] - 0.5
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

def get_min_area(bary):
    """Return the minimum area for a given configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def optimize_subset(bary, subset_indices, fixed_indices, max_iter=50):
    """
    Optimize the positions of points in subset_indices to maximize the minimum area
    involving any point in subset_indices, while keeping fixed_indices constant.
    We use a local L-BFGS-B on a smooth surrogate for the subset.
    """
    n_sub = len(subset_indices)
    if n_sub == 0:
        return bary
    
    # Extract current positions of subset
    sub_bary = bary[subset_indices].copy()
    fixed_bary = bary[fixed_indices]
    
    # We need to maximize the min area of any triangle that includes at least one point from the subset.
    # This is still a max-min problem, but on a smaller variable space (2 * n_sub).
    # We can use a simple grid search or a few steps of gradient descent.
    # Given time constraints, we will do a small number of L-BFGS-B iterations on the smooth min of the relevant triangles.
    
    # Identify relevant triangles: those where at least one index is in subset_indices
    mask = np.zeros(len(tri_indices), dtype=bool)
    for idx in subset_indices:
        mask |= (tri_idx[:, 0] == idx) | (tri_idx[:, 1] == idx) | (tri_idx[:, 2] == idx)
    relevant_tris = tri_idx[mask]
    
    if len(relevant_tris) == 0:
        return bary

    def sub_obj(x_flat):
        sub_bary_new = x_flat.reshape(n_sub, 2)
        sub_bary_new = clamp_bary(sub_bary_new)
        
        # Construct full bary array
        full_bary = bary.copy()
        full_bary[subset_indices] = sub_bary_new
        
        # Calculate areas for relevant triangles only
        u = full_bary[:, 0]
        v = full_bary[:, 1]
        w = 1.0 - u - v
        
        ui = u[relevant_tris[:, 0]]
        vi = v[relevant_tris[:, 0]]
        wi = w[relevant_tris[:, 0]]
        
        uj = u[relevant_tris[:, 1]]
        vj = v[relevant_tris[:, 1]]
        wj = w[relevant_tris[:, 1]]
        
        uk = u[relevant_tris[:, 2]]
        vk = v[relevant_tris[:, 2]]
        wk = w[relevant_tris[:, 2]]
        
        det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
        areas = np.abs(det)
        
        # Smooth min
        beta = 100.0
        if np.any(areas < 1e-15):
            return 1e10
        scaled = -beta * areas
        max_scaled = np.max(scaled)
        if max_scaled > 700:
            return -np.min(areas)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / beta
        return -soft_min

    bounds = [(0, 1)] * (n_sub * 2)
    try:
        res = minimize(
            sub_obj,
            sub_bary.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        if res.fun < sub_obj(sub_bary.flatten()): # Only accept if improved in surrogate
            new_bary = res.x.reshape(n_sub, 2)
            new_bary = clamp_bary(new_bary)
            bary[subset_indices] = new_bary
    except:
        pass
        
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Initial configuration: Hexagonal grid
    def generate_hexagonal_init(scale=0.25):
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
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pts), N, replace=False)
            return np.array(pts)[idx]
        return None

    init_bary = generate_hexagonal_init()
    if init_bary is None:
        rng = np.random.default_rng(123)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        init_bary = np.column_stack([u, v])
        
    current_bary = init_bary.copy()
    update_best(current_bary)
    
    # Simulated Annealing with Subset Replacement
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    rng = np.random.default_rng(999)
    
    # Precompute all possible subsets of size 3 and 4 for moves
    # Size 3 is fast, size 4 is better for escaping deep minima
    subsets_3 = list(combinations(range(N), 3))
    subsets_4 = list(combinations(range(N), 4))
    
    current_val = best_val
    
    while time.time() < deadline - 5:
        # Choose a subset to optimize
        # Alternate between size 3 and 4
        if rng.random() < 0.5:
            subset = rng.choice(len(subsets_3))
            subset_indices = list(subsets_3[subset])
        else:
            subset = rng.choice(len(subsets_4))
            subset_indices = list(subsets_4[subset])
            
        fixed_indices = [i for i in range(N) if i not in subset_indices]
        
        # Optimize the subset
        new_bary = optimize_subset(current_bary, subset_indices, fixed_indices, max_iter=30)
        
        new_val = get_min_area(new_bary)
        delta = new_val - current_val
        
        # Acceptance criterion
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_bary = new_bary
            current_val = new_val
            update_best(current_bary)
            
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Occasionally restart from best to avoid getting stuck in a bad local opt
        if rng.random() < 0.01 and best_bary is not None:
            # Perturb the best solution
            noise = rng.normal(0, 0.05, (N, 2))
            current_bary = best_bary + noise
            current_bary = clamp_bary(current_bary)
            current_val = get_min_area(current_bary)
            T = 0.05 # Reheat
            
    # Final local refinement on the best solution found
    if best_bary is not None and time.time() < deadline - 2:
        # Do a few passes of optimizing random subsets on the best solution
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            subset_indices = list(rng.choice(N, 4, replace=False))
            fixed_indices = [i for i in range(N) if i not in subset_indices]
            new_bary = optimize_subset(best_bary, subset_indices, fixed_indices, max_iter=50)
            update_best(new_bary)
            
    if best_bary is None:
        best_bary = init_bary
        
    cart = bary_to_cart(best_bary)
    return cart
