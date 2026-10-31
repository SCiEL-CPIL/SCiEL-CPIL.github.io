import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

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
AREA_EQUI = 0.5 * H  # Area of the unit equilateral triangle

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

def hard_min_obj(x):
    """
    Objective function for minimization.
    We want to maximize the minimum area, so we minimize the negative of the minimum area.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """
    Generate a hexagonal grid initialization.
    This provides a structured, spread-out starting point.
    """
    pts = []
    # A 3x3 grid in barycentric coordinates (10 points) plus one more
    # Let's try a 4x4 grid subset or a specific hexagon
    # Standard 10-point grid: i+j <= 3
    for i in range(4):
        for j in range(4 - i):
            pts.append([i/3.0, j/3.0])
    
    # Add 11th point at centroid
    pts.append([1/3.0, 1/3.0])
    
    bary = np.array(pts)
    return clamp_bary(bary)

def generate_random_init(rng):
    """Generate a random valid configuration."""
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return clamp_bary(np.column_stack([u, v]))

def generate_perturbed_init(bary, rng, scale=0.05):
    """Generate a perturbed version of a given configuration."""
    perturbation = rng.normal(0, scale, bary.shape)
    new_bary = bary + perturbation
    return clamp_bary(new_bary)

def refine_local(bary, time_limit=5.0):
    """
    Perform a local refinement using Nelder-Mead to polish the solution.
    """
    start_time = time.time()
    
    try:
        res = minimize(
            hard_min_obj,
            bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if time.time() > start_time + time_limit:
            return bary
            
        candidate_bary = clamp_bary(res.x.reshape(N, 2))
        return candidate_bary
    except Exception:
        return bary

def simulated_annealing(initial_bary, start_temp=0.01, end_temp=1e-6, time_limit=30.0):
    """
    Perform Simulated Annealing to escape local minima.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    current_bary = initial_bary.copy()
    current_val = -hard_min_obj(current_bary.flatten())
    
    best_bary = current_bary.copy()
    best_val = current_val
    
    n_iter = 0
    total_time = time_limit
    
    while time.time() - start_time < total_time:
        n_iter += 1
        
        # Cooling schedule
        progress = min(1.0, (time.time() - start_time) / total_time)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Generate neighbor
        # Perturb a random subset of points
        n_points_to_perturb = rng.integers(1, 4)
        idx = rng.choice(N, size=n_points_to_perturb, replace=False)
        
        new_bary = current_bary.copy()
        perturb_scale = 0.05 * (1.0 + temp) # Scale perturbation with temperature
        new_bary[idx] += rng.normal(0, perturb_scale, size=(n_points_to_perturb, 2))
        new_bary = clamp_bary(new_bary)
        
        new_val = -hard_min_obj(new_bary.flatten())
        
        # Acceptance probability
        delta = new_val - current_val
        if delta > 0 or rng.random() < np.exp(delta / temp):
            current_bary = new_bary
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
                
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = -np.inf
    best_bary = None
    
    rng = np.random.default_rng(42)
    
    # 1. Initial diverse starts
    inits = [generate_hexagonal_init()]
    for _ in range(5):
        inits.append(generate_random_init(rng))
    
    # Sort by initial quality
    init_scores = []
    for bary in inits:
        val = -hard_min_obj(bary.flatten())
        init_scores.append((val, bary))
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Differential Evolution for global search
    # DE works on the flattened vector of 22 parameters (11 points * 2 coords)
    # Bounds: u in [0,1], v in [0,1] (clamping handles the triangle constraint)
    bounds = [(0, 1)] * (2 * N)
    
    # We will run DE for a specific time budget, then refine the best result.
    # DE is slow, so we use a small population and limited iterations.
    
    # We'll interleave DE with local refinement of its best individuals.
    
    # First, let's do a quick local refinement of the best initial points
    for val, bary in init_scores:
        if time.time() > deadline - 20.0:
            break
        polished = refine_local(bary, time_limit=5.0)
        p_val = -hard_min_obj(polished.flatten())
        if p_val > best_val:
            best_val = p_val
            best_bary = polished
            
    # Now, use DE to explore the landscape
    # We define a wrapper for DE that respects the time limit
    de_start_time = time.time()
    de_deadline = start_time + TIME_LIMIT - 15.0 # Leave 15s for final refinement
    
    # DE objective
    def de_objective(x):
        return hard_min_obj(x)
    
    # Run DE
    try:
        de_res = differential_evolution(
            de_objective,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-4,
            seed=42,
            init='sobol',
            updating='deferred',
            polish=True, # Let DE do its own local polish at the end
            x0=[bary.flatten() for bary, _ in init_scores[:3]] # Use best inits as seeds
        )
        
        # DE returns the best solution found
        de_best_bary = clamp_bary(de_res.x.reshape(N, 2))
        de_best_val = -hard_min_obj(de_best_bary.flatten())
        
        if de_best_val > best_val:
            best_val = de_best_val
            best_bary = de_best_bary
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # 3. Simulated Annealing from the best solution found so far
    if best_bary is not None:
        if time.time() < de_deadline - 5.0:
            sa_time = min(20.0, de_deadline - time.time())
            sa_bary = simulated_annealing(best_bary, start_temp=0.05, end_temp=1e-5, time_limit=sa_time)
            sa_val = -hard_min_obj(sa_bary.flatten())
            if sa_val > best_val:
                best_val = sa_val
                best_bary = sa_bary
                
    # 4. Final local refinement
    if best_bary is not None:
        if time.time() < deadline - 2.0:
            final_bary = refine_local(best_bary, time_limit=5.0)
            final_val = -hard_min_obj(final_bary.flatten())
            if final_val > best_val:
                best_val = final_val
                best_bary = final_bary
                
    if best_bary is None:
        best_bary = inits[0]
        
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
