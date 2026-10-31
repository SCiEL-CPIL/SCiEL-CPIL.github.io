import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0
H = np.sqrt(3) / 2

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle index combinations
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def point_in_triangle(pt, v0, v1, v2):
    """Check if point is inside triangle using barycentric coordinates."""
    x, y = pt
    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2
    
    # Barycentric coordinates
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-15:
        return False
    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    c = 1.0 - a - b
    
    return a >= -1e-10 and b >= -1e-10 and c >= -1e-10

def project_to_triangle(pt):
    """Project point onto the triangle if it's outside."""
    if point_in_triangle(pt, V0, V1, V2):
        return pt
    
    # Find closest point on triangle boundary
    # Check edges
    best_pt = pt
    best_dist = np.inf
    
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    for p1, p2 in edges:
        # Project pt onto line through p1, p2
        v = p2 - p1
        t = np.dot(pt - p1, v) / np.dot(v, v)
        t = np.clip(t, 0, 1)
        proj = p1 + t * v
        dist = np.linalg.norm(pt - proj)
        if dist < best_dist:
            best_dist = dist
            best_pt = proj
    
    # Check vertices
    for v in [V0, V1, V2]:
        dist = np.linalg.norm(pt - v)
        if dist < best_dist:
            best_dist = dist
            best_pt = v
    
    return best_pt

def clamp_points_cart(pts):
    """Clamp all points to be inside the triangle."""
    clamped = np.array([project_to_triangle(p) for p in pts])
    return clamped

def compute_min_area_cart(pts):
    """Compute minimum triangle area for points in Cartesian coordinates."""
    # pts shape: (N, 2)
    # For each triangle (i, j, k), area = 0.5 * |cross(Pj-Pi, Pk-Pi)|
    pi = pts[tri_idx[:, 0]]  # (165, 2)
    pj = pts[tri_idx[:, 1]]  # (165, 2)
    pk = pts[tri_idx[:, 2]]  # (165, 2)
    
    v1 = pj - pi  # (165, 2)
    v2 = pk - pi  # (165, 2)
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective_cart(x):
    """Objective function for Cartesian optimization: minimize negative min area."""
    pts = x.reshape(N, 2)
    pts = clamp_points_cart(pts)
    min_area = compute_min_area_cart(pts)
    return -min_area

def simulated_annealing(pts_init, time_limit=10.0, T_init=0.05, cooling=0.995, min_T=1e-8):
    """Simulated annealing for local refinement in Cartesian space."""
    start_time = time.time()
    current_pts = pts_init.copy()
    current_val = -objective_cart(current_pts.flatten())
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = T_init
    
    while time.time() - start_time < time_limit and T > min_T:
        # Generate neighbor by perturbing one point
        p_idx = np.random.randint(N)
        delta = np.random.normal(0, T, 2)
        new_pts = current_pts.copy()
        new_pts[p_idx] += delta
        new_pts = clamp_points_cart(new_pts)
        
        new_val = -objective_cart(new_pts.flatten())
        delta_obj = new_val - current_val
        
        # Accept if better or with probability exp(delta_obj / T)
        if delta_obj > 0 or np.random.random() < np.exp(delta_obj / T):
            current_pts = new_pts
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_pts = current_pts.copy()
        
        T *= cooling
    
    return best_pts

def fine_tune(pts_init, time_limit=5.0):
    """Very fine-grained tuning with tiny perturbations."""
    start_time = time.time()
    current_pts = pts_init.copy()
    current_val = -objective_cart(current_pts.flatten())
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = 0.005
    cooling = 0.998
    min_T = 1e-10
    
    while time.time() - start_time < time_limit and T > min_T:
        p_idx = np.random.randint(N)
        delta = np.random.normal(0, T, 2)
        new_pts = current_pts.copy()
        new_pts[p_idx] += delta
        new_pts = clamp_points_cart(new_pts)
        
        new_val = -objective_cart(new_pts.flatten())
        delta_obj = new_val - current_val
        
        if delta_obj > 0 or np.random.random() < np.exp(delta_obj / T):
            current_pts = new_pts
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_pts = current_pts.copy()
        
        T *= cooling
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Define bounds for Cartesian coordinates
    # Triangle spans x in [0, 1], y in [0, H]
    bounds = [(0, 1), (0, H)] * N
    
    # Phase 1: Differential Evolution for global optimization
    try:
        de_result = differential_evolution(
            objective_cart,
            bounds,
            seed=42,
            maxiter=150,
            popsize=20,
            tol=1e-12,
            polish=False,
            init='sobol'
        )
        best_pts = clamp_points_cart(de_result.x.reshape(N, 2))
    except Exception as e:
        # Fallback: use a structured initial guess
        # Known good configuration: vertices + centroid + midpoints + symmetric points
        rng = np.random.default_rng(42)
        # Start with vertices and centroid
        pts = np.array([V0, V1, V2, [0.5, H/3]])
        # Add points along edges and interior
        pts = np.vstack([pts, 
                         [0.25, H/6], [0.75, H/6],
                         [0.5, H/6], [0.25, H/3], [0.75, H/3],
                         [0.5, 2*H/3], [0.25, 2*H/3], [0.75, 2*H/3]])
        best_pts = clamp_points_cart(pts)
    
    best_val = -objective_cart(best_pts.flatten())
    
    # Phase 2: Simulated Annealing refinement with multiple passes
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        sa_time1 = max(5.0, remaining_time * 0.3)
        refined_pts = simulated_annealing(best_pts, time_limit=sa_time1, T_init=0.05, cooling=0.995)
        refined_val = -objective_cart(refined_pts.flatten())
        if refined_val > best_val:
            best_pts = refined_pts
            best_val = refined_val
        
        # Second SA pass with different parameters
        remaining_time = deadline - time.time()
        if remaining_time > 8:
            sa_time2 = max(3.0, remaining_time * 0.25)
            refined_pts2 = simulated_annealing(best_pts, time_limit=sa_time2, T_init=0.03, cooling=0.997)
            refined_val2 = -objective_cart(refined_pts2.flatten())
            if refined_val2 > best_val:
                best_pts = refined_pts2
                best_val = refined_val2
    
    # Phase 3: Multi-start with perturbed versions
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        rng = np.random.default_rng(123)
        num_restarts = min(8, int(remaining_time / 8))
        
        for i in range(num_restarts):
            if time.time() > deadline - 12:
                break
            
            # Perturb best solution with varying sigma
            sigma = 0.03 + 0.015 * i
            init_pts = best_pts + rng.normal(0, sigma, best_pts.shape)
            init_pts = clamp_points_cart(init_pts)
            
            # Quick SA from this start
            sa_time = max(2.0, (deadline - time.time() - 12) / max(1, num_restarts - i))
            candidate = simulated_annealing(init_pts, time_limit=sa_time, T_init=0.04, cooling=0.996)
            candidate_val = -objective_cart(candidate.flatten())
            
            if candidate_val > best_val:
                best_pts = candidate
                best_val = candidate_val
    
    # Phase 4: L-BFGS-B polish (better for smooth objectives)
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        try:
            lbfgs_result = minimize(
                objective_cart,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            lbfgs_pts = clamp_points_cart(lbfgs_result.x.reshape(N, 2))
            lbfgs_val = -objective_cart(lbfgs_pts.flatten())
            
            if lbfgs_val > best_val:
                best_pts = lbfgs_pts
                best_val = lbfgs_val
        except Exception:
            pass
    
    # Phase 5: Fine-tuning
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        fine_pts = fine_tune(best_pts, time_limit=min(3.0, remaining_time - 1))
        fine_val = -objective_cart(fine_pts.flatten())
        if fine_val > best_val:
            best_pts = fine_pts
            best_val = fine_val
    
    return best_pts

# EVOLVE-BLOCK-END
