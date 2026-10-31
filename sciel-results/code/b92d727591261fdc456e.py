import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 142.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_TOTAL = np.sqrt(3) / 4.0

# Barycentric coordinates of the 3 fixed corner points
CORNERS_BARY = np.array([
    [0.0, 0.0],  # V0
    [1.0, 0.0],  # V1
    [0.0, 1.0],  # V2
])

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

def min_area_from_interior(interior_bary):
    """
    Given 8 interior points in barycentric coords, compute the minimum triangle area
    when combined with the 3 fixed corner points.
    """
    # Build full 11-point barycentric array: 3 corners + 8 interior
    full_bary = np.vstack([CORNERS_BARY, interior_bary])
    areas = calc_areas_bary(full_bary)
    return np.min(areas)

def objective_interior(interior_flat):
    """Objective function for the 8 interior points (16 variables)."""
    interior_bary = interior_flat.reshape(8, 2)
    interior_bary = clamp_bary(interior_bary)
    return -min_area_from_interior(interior_bary)

def simulated_annealing_init(rng, n_iters=5000, t0=0.01, t_end=1e-5, alpha=0.9997):
    """
    Simulated annealing on the 8 interior points.
    Returns the best configuration found.
    """
    # Initialize with hexagonal-like pattern
    interior = np.array([
        [0.25, 0.0],   # bottom row
        [0.5, 0.0],
        [0.75, 0.0],
        [0.375, 0.25], # middle row
        [0.625, 0.25],
        [0.25, 0.5],   # upper row
        [0.5, 0.5],
        [0.375, 0.75], # top
    ])
    interior = clamp_bary(interior)
    
    current = interior.copy()
    current_val = -objective_interior(current.flatten())
    
    best = current.copy()
    best_val = current_val
    
    t = t0
    
    for i in range(n_iters):
        # Perturb a random point
        idx = rng.integers(0, 8)
        step = rng.normal(0, 0.05, size=2)
        new_interior = current.copy()
        new_interior[idx] += step
        new_interior = clamp_bary(new_interior)
        
        new_val = -objective_interior(new_interior.flatten())
        
        delta = new_val - current_val
        
        if delta > 0 or rng.random() < np.exp(delta / t):
            current = new_interior
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best = current.copy()
        
        t *= alpha
        if t < t_end:
            t = t_end
    
    return best

def polish_nelder_mead(interior, max_iter=500):
    """Polish with Nelder-Mead."""
    try:
        result = minimize(
            objective_interior,
            interior.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        polished = clamp_bary(result.x.reshape(8, 2))
        val = -objective_interior(polished.flatten())
        orig_val = -objective_interior(interior.flatten())
        if val >= orig_val:
            return polished
        return interior
    except Exception:
        return interior

def generate_hex_interior():
    """Generate hexagonal-like interior configurations."""
    candidates = []
    
    # Pattern 1: Hexagonal grid
    p1 = np.array([
        [0.25, 0.0],
        [0.5, 0.0],
        [0.75, 0.0],
        [0.375, 0.25],
        [0.625, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
        [0.375, 0.75],
    ])
    candidates.append(p1)
    
    # Pattern 2: Shifted hexagonal
    p2 = np.array([
        [0.2, 0.05],
        [0.5, 0.05],
        [0.8, 0.05],
        [0.35, 0.3],
        [0.65, 0.3],
        [0.25, 0.55],
        [0.5, 0.55],
        [0.4, 0.75],
    ])
    candidates.append(p2)
    
    # Pattern 3: Symmetric about center
    p3 = np.array([
        [0.25, 0.1],
        [0.5, 0.1],
        [0.75, 0.1],
        [0.35, 0.35],
        [0.65, 0.35],
        [0.25, 0.6],
        [0.5, 0.6],
        [0.4, 0.8],
    ])
    candidates.append(p3)
    
    # Pattern 4: More spread out
    p4 = np.array([
        [0.2, 0.0],
        [0.5, 0.0],
        [0.8, 0.0],
        [0.33, 0.33],
        [0.67, 0.33],
        [0.2, 0.66],
        [0.5, 0.66],
        [0.33, 0.9],
    ])
    candidates.append(p4)
    
    # Add perturbed versions
    rng = np.random.default_rng(123)
    for base in [p1, p2, p3, p4]:
        for _ in range(2):
            perturbed = base + rng.normal(0, 0.03, size=base.shape)
            perturbed = clamp_bary(perturbed)
            candidates.append(perturbed)
    
    return candidates

def generate_random_interior(rng, n=5):
    """Generate random interior configurations."""
    candidates = []
    for _ in range(n):
        u = rng.random(8)
        v = rng.random(8) * (1 - u) * 0.8  # Keep away from boundaries
        candidates.append(np.column_stack([u, v]))
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_interior = None
    best_val = -np.inf
    
    def update_best(interior):
        nonlocal best_interior, best_val
        interior = clamp_bary(interior)
        val = -objective_interior(interior.flatten())
        if val > best_val:
            best_val = val
            best_interior = interior.copy()
            return True
        return False

    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated annealing from multiple starts
    candidates = generate_hex_interior()
    candidates.extend(generate_random_interior(rng, n=4))
    
    n_sa = 4  # Number of SA runs
    sa_time_budget = 25.0  # seconds per SA run
    
    for i in range(min(n_sa, len(candidates))):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - 30 - time.time()
        if remaining < 5:
            break
        
        # Allocate time based on remaining
        t_budget = min(sa_time_budget, remaining / max(1, n_sa - i))
        
        # Estimate iterations based on time budget
        # Each iteration takes ~0.1ms, so ~10000 iters per second
        n_iters = int(min(5000, t_budget * 8000))
        
        try:
            sa_result = simulated_annealing_init(rng, n_iters=n_iters)
            update_best(sa_result)
        except Exception:
            continue
    
    # Phase 2: Polish the best result
    if best_interior is not None and time.time() < deadline - 20:
        polished = polish_nelder_mead(best_interior, max_iter=500)
        update_best(polished)
    
    # Phase 3: More SA runs with different parameters if time permits
    if time.time() < deadline - 25:
        for i in range(3):
            if time.time() > deadline - 15:
                break
            
            # Different initializations
            u = rng.random(8)
            v = rng.random(8) * (1 - u) * 0.9
            init = np.column_stack([u, v])
            init = clamp_bary(init)
            
            n_iters = int(min(3000, (deadline - 15 - time.time()) * 8000))
            
            try:
                sa_result = simulated_annealing_init(rng, n_iters=n_iters, t0=0.02, t_end=1e-6, alpha=0.9995)
                update_best(sa_result)
            except Exception:
                continue
    
    # Phase 4: Final polish
    if best_interior is not None and time.time() < deadline - 5:
        polished = polish_nelder_mead(best_interior, max_iter=1000)
        update_best(polished)
        
        if time.time() < deadline - 2:
            # Another round of SA from the best
            try:
                n_iters = int(min(2000, (deadline - 2 - time.time()) * 8000))
                sa_result = simulated_annealing_init(rng, n_iters=n_iters, t0=0.005, t_end=1e-7, alpha=0.9998)
                update_best(sa_result)
            except Exception:
                pass
    
    if best_interior is None:
        # Fallback: use the first hexagonal candidate
        best_interior = generate_hex_interior()[0]
    
    # Build full 11-point configuration
    full_bary = np.vstack([CORNERS_BARY, best_interior])
    cart = bary_to_cart(full_bary)
    
    return cart
