import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
h = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, h])

# Precompute all triplets for vectorized area computation
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int32)  # shape (165, 3)

def compute_areas(points):
    """
    points: (N, 2)
    Returns array of areas for all triplets.
    """
    p1 = points[_TRIPLETS[:, 0]]
    p2 = points[_TRIPLETS[:, 1]]
    p3 = points[_TRIPLETS[:, 2]]
    
    v12 = p2 - p1
    v13 = p3 - p1
    
    cross = v12[:, 0] * v13[:, 1] - v12[:, 1] * v13[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def min_triangle_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    return np.min(compute_areas(points))

def soft_min_triangle_area(points, beta=100.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    This makes the objective differentiable (mostly) and helps gradient-based 
    or annealing searches navigate the landscape more smoothly.
    """
    areas = compute_areas(points)
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # To avoid overflow, we subtract the max of -beta*areas (which is -beta*min(areas))
    # Let z = -beta * areas
    # soft_min = -1/beta * log(sum(exp(z)))
    # log_sum_exp(z) = max(z) + log(sum(exp(z - max(z))))
    
    z = -beta * areas
    z_max = np.max(z)
    if z_max > 700: # Prevent overflow
        z_max = 700
    log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)))
    return -log_sum_exp / beta

def project_to_triangle(p):
    """
    p: array of shape (N, 2)
    Returns projected points inside the triangle (0,0), (1,0), (0.5, h)
    """
    p = p.copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, h)
    
    for _ in range(5):
        # Edge AB: y >= 0
        mask_ab = p[:, 1] < 0.0
        if np.any(mask_ab):
            p[mask_ab, 1] = 0.0
            
        # Edge AC: y <= sqrt(3) x
        mask_ac = p[:, 1] > np.sqrt(3) * p[:, 0]
        if np.any(mask_ac):
            px = p[mask_ac, 0]
            py = p[mask_ac, 1]
            dot = px + py * np.sqrt(3)
            px_proj = dot / 4.0
            py_proj = dot * np.sqrt(3) / 4.0
            px_proj = np.clip(px_proj, 0.0, 0.5)
            py_proj = px_proj * np.sqrt(3)
            p[mask_ac, 0] = px_proj
            p[mask_ac, 1] = py_proj
            
        # Edge BC: y <= sqrt(3)(1-x)
        mask_bc = p[:, 1] > np.sqrt(3) * (1.0 - p[:, 0])
        if np.any(mask_bc):
            px = p[mask_bc, 0]
            py = p[mask_bc, 1]
            d = (np.sqrt(3) * px + py - np.sqrt(3)) / 4.0
            px_proj = px - d * np.sqrt(3)
            py_proj = py - d * 1.0
            px_proj = np.clip(px_proj, 0.5, 1.0)
            py_proj = np.sqrt(3) * (1.0 - px_proj)
            p[mask_bc, 0] = px_proj
            p[mask_bc, 1] = py_proj
    return p

def generate_structured_initial():
    """
    Generate a high-quality initial guess based on symmetry.
    We assume points are arranged in horizontal rows.
    For N=11, a common optimal structure is rows of 5, 3, 2, 1 or similar.
    Let's try 5, 3, 2, 1.
    """
    rows = [5, 3, 2, 1]
    y_levels = []
    # Distribute y levels somewhat evenly, but adjusted for triangle width
    # Width at y is 1 - 2*y/sqrt(3)
    # To maximize min area, we want points spaced out.
    # Let's just pick some reasonable y levels.
    y_levels = [0.1*h, 0.35*h, 0.6*h, 0.85*h]
    
    pts = []
    for y, count in zip(y_levels, rows):
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        width = x_max - x_min
        if count > 1:
            # Add margin to keep points strictly inside
            margin = 0.05 * width
            xs = np.linspace(x_min + margin, x_max - margin, count)
        else:
            xs = np.array([(x_min + x_max) / 2])
        for x in xs:
            pts.append([x, y])
            
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = -1
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # --- Phase 1: Simulated Annealing with Soft Min Objective ---
    # We use a structured initial point.
    current_points = generate_structured_initial()
    current_points = project_to_triangle(current_points)
    current_score = min_triangle_area(current_points)
    best_score = current_score
    best_points = current_points.copy()
    
    # Annealing parameters
    T_start = 0.05
    T_end = 1e-6
    n_iterations = 20000
    
    # We will perturb the points slightly.
    # To respect the structure, we can perturb x and y independently with small steps.
    # However, to allow breaking symmetry if beneficial, we allow any perturbation.
    # We use the soft min objective for the acceptance criterion to smooth the landscape.
    
    beta_schedule = [10.0, 50.0, 200.0] # Increase beta over time to sharpen the objective
    
    # Split iterations into 3 blocks with increasing beta
    iters_per_block = n_iterations // 3
    remaining_iters = n_iterations
    
    for beta_idx, beta in enumerate(beta_schedule):
        if time.time() > deadline - 10.0:
            break
            
        T = T_start
        # Cooling schedule: exponential
        cool_factor = (T_end / T_start) ** (1.0 / iters_per_block)
        
        for i in range(iters_per_block):
            if time.time() > deadline - 10.0:
                break
                
            # Generate a neighbor
            # Perturb a random subset of points
            n_perturb = rng.integers(1, 4) # 1 to 3 points
            indices = rng.choice(N, size=n_perturb, replace=False)
            
            new_points = current_points.copy()
            # Add Gaussian noise
            noise_scale = 0.01 * T # Scale noise with temperature
            noise = rng.normal(0, noise_scale, size=(n_perturb, 2))
            new_points[indices] += noise
            
            # Project to triangle
            new_points = project_to_triangle(new_points)
            
            # Evaluate
            # Use soft min for acceptance to allow smoother navigation
            old_soft_score = soft_min_triangle_area(current_points, beta=beta)
            new_soft_score = soft_min_triangle_area(new_points, beta=beta)
            
            # We want to MAXIMIZE the area.
            # So we want to accept if new_soft_score > old_soft_score.
            # Delta = new - old
            delta = new_soft_score - old_soft_score
            
            if delta > 0:
                # Accept
                current_points = new_points
                curr_real_score = min_triangle_area(current_points)
                if curr_real_score > best_score:
                    best_score = curr_real_score
                    best_points = current_points.copy()
            else:
                # Accept with probability exp(delta / T)
                # Note: delta is negative here.
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    curr_real_score = min_triangle_area(current_points)
                    if curr_real_score > best_score:
                        best_score = curr_real_score
                        best_points = current_points.copy()
            
            # Cool down
            T *= cool_factor
            
    # --- Phase 2: Local Refinement with Nelder-Mead ---
    # Use the best point found so far
    if best_points is not None and time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        
        def objective_flat(x):
            points = x.reshape(N, 2)
            points = project_to_triangle(points)
            return -min_triangle_area(points)
            
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 
                                   'adaptive': True})
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass

    # --- Phase 3: Final Polish with tighter tolerance ---
    if best_points is not None and time.time() < deadline - 2.0:
        x0 = best_points.flatten()
        
        def objective_flat(x):
            points = x.reshape(N, 2)
            points = project_to_triangle(points)
            return -min_triangle_area(points)
            
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 
                                   'adaptive': True})
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = generate_structured_initial()

    return best_points

# EVOLVE-BLOCK-END
