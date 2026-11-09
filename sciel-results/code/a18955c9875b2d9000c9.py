# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the minimum
    triangle area to the convex hull area.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Safety margin from 630s

    # Helper to compute score
    def calculate_score(points):
        """
        Returns the score: min_triangle_area / hull_area.
        Returns -inf if invalid.
        """
        if not np.all(np.isfinite(points)):
            return -np.inf
        
        # Calculate all triangle areas
        # indices of all combinations of 3 points
        idx = np.arange(n)
        # Use combinations for clarity, but vectorized is faster
        # For n=13, 286 triangles.
        # We can vectorize this.
        
        # p1, p2, p3
        # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        
        # Generate indices
        i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
        i = i.flatten()
        j = j.flatten()
        k = k.flatten()
        
        # Filter i < j < k
        mask = (i < j) & (j < k)
        i, j, k = i[mask], j[mask], k[mask]
        
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # Vectorized area calculation
        # Cross product of (p2-p1) and (p3-p1)
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        # Hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except:
            return -np.inf
            
        if hull_area < 1e-10:
            return -np.inf
            
        return min_area / hull_area

    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(init_reg)
    
    # 2. Hexagonal packing (1 center, 6 inner, 6 outer)
    pts = []
    pts.append([0, 0])
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    for i in range(6):
        angle = (i + 0.5) * np.pi / 3
        pts.append([2 * np.cos(angle), 2 * np.sin(angle)])
    init_hex = np.array(pts)
    inits.append(init_hex)
    
    # 3. Slightly perturbed regular 13-gon
    rng = np.random.default_rng(42)
    init_pert = init_reg + rng.normal(0, 0.1, (n, 2))
    inits.append(init_pert)

    best_points = None
    best_score = -np.inf
    
    # Evaluate inits
    for init in inits:
        score = calculate_score(init)
        if score > best_score:
            best_score = score
            best_points = init.copy()

    if time.time() - start_time > time_limit:
        return best_points

    # Optimization loop with annealing
    # We minimize -score
    def objective(flat_pts, beta=1000):
        pts = flat_pts.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
            
        # Calculate areas vectorized
        idx = np.arange(n)
        i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
        i = i.flatten()
        j = j.flatten()
        k = k.flatten()
        mask = (i < j) & (j < k)
        i, j, k = i[mask], j[mask], k[mask]
        
        p1 = pts[i]
        p2 = pts[j]
        p3 = pts[k]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft min
        # To avoid overflow in exp, shift areas
        # softmin = -1/beta * log(sum(exp(-beta * areas)))
        # This is numerically unstable if areas are large or beta is large.
        # Better: min_area approx
        # If beta is large, just use min.
        # Let's use a hybrid: if beta < 50, use soft min, else use min.
        # Or just use min for simplicity and robustness, relying on multiple restarts/annealing of beta in the objective value itself isn't straightforward with minimize.
        # Instead, we will run multiple minimization steps with different betas.
        
        if beta < 50:
            # Soft min
            # max area for stability
            a_max = np.max(areas)
            exp_terms = np.exp(-beta * (areas - a_max))
            log_sum = np.log(np.sum(exp_terms))
            soft_min = a_max - log_sum / beta
        else:
            soft_min = np.min(areas)
            
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except:
            return 1e10
            
        if hull_area < 1e-10:
            return 1e10
            
        return -soft_min / hull_area

    bounds = [(-10, 10)] * (2 * n)
    
    # Annealing schedule
    betas = [10, 50, 200, 1000]
    
    current_points = best_points
    current_flat = current_points.flatten()
    
    for beta in betas:
        if time.time() - start_time > time_limit - 10:
            break
            
        res = minimize(
            objective, 
            current_flat, 
            args=(beta,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < 1e9:
            new_points = res.x.reshape(n, 2)
            new_score = calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
                current_flat = new_points.flatten()
            else:
                # If it didn't improve, keep current_flat as start for next beta?
                # Or revert?
                current_flat = new_points.flatten() # Even if score dropped slightly due to soft-min mismatch, geometry might be better for higher beta
                
    # Final polish with hard min (beta=inf effectively)
    if time.time() - start_time < time_limit - 5:
        def objective_hard(flat_pts):
            return objective(flat_pts, beta=1000)
            
        res = minimize(
            objective_hard, 
            current_flat, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if res.success or res.fun < 1e9:
            new_points = res.x.reshape(n, 2)
            new_score = calculate_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points

    # Ensure validity
    if not np.all(np.isfinite(best_points)):
        return inits[0] # Fallback
        
    return best_points

# EVOLVE-BLOCK-END
