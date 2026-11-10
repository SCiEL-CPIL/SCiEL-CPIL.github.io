# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
import math

N = 13
# Precompute triangle indices for efficiency
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull
    Returns 0.0 if invalid
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    # Vectorized area calculation
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def soft_min_score(points, temperature=0.01):
    """
    Smooth surrogate for the objective function.
    We want to maximize A_min / A_hull.
    So we minimize -A_min / A_hull.
    A_min = min(areas).
    Soft min: -T * log(sum(exp(-area/T)))
    """
    if not np.all(np.isfinite(points)):
        return 1e10
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
    
    if A_hull <= 0:
        return 1e10
        
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    # Soft min of areas
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_area = -temperature * log_sum
    
    # Objective to minimize: -soft_min_area / A_hull
    return -soft_min_area / A_hull

def nelder_mead(f, x0, max_iter=5000, tol=1e-8, time_limit=None, start_time=None):
    """
    Nelder-Mead optimization algorithm.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 60.0
        
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    # Standard Nelder-Mead initialization: axis-aligned simplex
    step = 0.1
    for i in range(n):
        x_new = x0.copy()
        if abs(x0[i]) > 1e-6:
            x_new[i] *= 1.05
        else:
            x_new[i] += step
        simplex.append(x_new)
        
    # Parameters
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    # Evaluate initial simplex
    values = [f(x) for x in simplex]
    
    # Sort by value
    order = np.argsort(values)
    simplex = [simplex[i] for i in order]
    values = [values[i] for i in order]
    
    best_x = simplex[0].copy()
    best_f = values[0]
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
            
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_reflected = centroid + alpha * (centroid - simplex[-1])
        f_reflected = f(x_reflected)
        
        if f_reflected < values[0]:
            # Expansion
            x_expanded = centroid + gamma * (centroid - simplex[-1])
            f_expanded = f(x_expanded)
            if f_expanded < f_reflected:
                simplex[-1] = x_expanded
                values[-1] = f_expanded
            else:
                simplex[-1] = x_reflected
                values[-1] = f_reflected
        elif f_reflected < values[-2]:
            # Accept reflection
            simplex[-1] = x_reflected
            values[-1] = f_reflected
        else:
            # Contraction
            x_contracted = centroid + rho * (simplex[-1] - centroid)
            f_contracted = f(x_contracted)
            if f_contracted < values[-1]:
                simplex[-1] = x_contracted
                values[-1] = f_contracted
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = f(simplex[i])
        
        # Re-sort
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if values[0] < best_f:
            best_f = values[0]
            best_x = simplex[0].copy()
            
    return best_x, best_f

def generate_initial_candidates(rng):
    """
    Generate diverse initial candidates based on geometric templates.
    """
    candidates = []
    
    # Template 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts)
    
    # Template 2: 1 center + 12 outer (regular 12-gon)
    center = np.array([0.0, 0.0])
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    pts = np.vstack([center, outer])
    candidates.append(pts)
    
    # Template 3: 1 center + 6 inner + 6 outer
    center = np.array([0.0, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([center, inner, outer])
    candidates.append(pts)
    
    # Template 4: 2 center + 5 inner + 6 outer
    c1 = np.array([-0.1, 0.0])
    c2 = np.array([0.1, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 5, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([c1, c2, inner, outer])
    candidates.append(pts)
    
    # Template 5: Grid-like (4x4 grid, remove 3 points to get 13, centered)
    xs = np.array([-1, -0.33, 0.33, 1.0])
    ys = np.array([-1, -0.33, 0.33, 1.0])
    grid = np.array([[x, y] for x in xs for y in ys])
    pts = grid[:13]
    candidates.append(pts)
    
    # Template 6: Random
    pts = rng.uniform(-1, 1, size=(13, 2))
    candidates.append(pts)
    
    # Template 7: Fibonacci spiral
    golden_angle = 2.39996
    radii = np.sqrt(np.arange(13) / 13.0)
    angles = np.arange(13) * golden_angle
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(pts)
    
    return candidates

def optimize_barycentric(hull_pts, interior_count, rng, time_limit):
    """
    Optimize interior points using barycentric coordinates relative to a triangulation
    of the fixed convex hull.
    
    hull_pts: (13, 2) array of hull vertices (regular 13-gon)
    interior_count: number of interior points (0 to 10)
    """
    if interior_count == 0:
        return hull_pts.copy()
    
    # Create a simple triangulation: fan from vertex 0
    # Triangles: (0, i, i+1) for i = 1, ..., 11
    # This gives us 11 triangles
    tri_indices = [(0, i, i+1) for i in range(1, 12)]
    
    # For each interior point, assign it to a triangle and use barycentric coords
    # We'll randomly assign interior points to triangles
    assignments = rng.integers(0, len(tri_indices), size=interior_count)
    
    # Initialize barycentric coordinates (must be non-negative and sum to 1)
    # Use Dirichlet distribution for initialization
    bary_coords = []
    for _ in range(interior_count):
        # Random Dirichlet(1,1,1)
        b = rng.dirichlet([1, 1, 1])
        bary_coords.append(b)
    
    bary_coords = np.array(bary_coords)
    
    def bary_to_points(bary_flat):
        """Convert barycentric coordinates to actual point positions."""
        points = hull_pts.copy()
        for i in range(interior_count):
            b = bary_flat[i*3:(i+1)*3]
            # Ensure non-negativity and normalization
            b = np.maximum(b, 0)
            s = np.sum(b)
            if s < 1e-10:
                b = np.array([1/3, 1/3, 1/3])
            else:
                b = b / s
            tri = tri_indices[assignments[i]]
            p0, p1, p2 = hull_pts[tri[0]], hull_pts[tri[1]], hull_pts[tri[2]]
            point = b[0]*p0 + b[1]*p1 + b[2]*p2
            points[13 - interior_count + i] = point
        return points
    
    def objective(bary_flat):
        """Objective function to minimize: -soft_min_score"""
        points = bary_to_points(bary_flat)
        return soft_min_score(points, temperature=0.01)
    
    # Initial point
    x0 = bary_coords.flatten()
    
    # Bounds: barycentric coords in [0, 1] (will be normalized anyway)
    bounds = [(0, 1)] * (interior_count * 3)
    
    # Use L-BFGS-B with constraints
    start_time = time.time()
    
    # Multiple restarts
    best_x = x0.copy()
    best_f = objective(x0)
    
    n_restarts = 3
    time_per_restart = (time_limit - (time.time() - start_time)) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 1:
            break
        
        # Perturb initial conditions
        if restart > 0:
            x0_perturbed = bary_coords.flatten() + rng.normal(0, 0.1, size=x0.shape)
            x0_perturbed = np.clip(x0_perturbed, 0, 1)
        else:
            x0_perturbed = x0.copy()
        
        # Optimize
        result = minimize(objective, x0_perturbed, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.fun < best_f:
            best_f = result.fun
            best_x = result.x.copy()
    
    # Convert back to points
    final_points = bary_to_points(best_x)
    
    # Polish with Nelder-Mead on the full point set
    current_time = time.time()
    if current_time < time_limit - 5:
        def obj_full(x_flat):
            pts = x_flat.reshape(13, 2)
            return soft_min_score(pts, temperature=0.005)
        
        x0_full = final_points.flatten()
        x_opt, f_opt = nelder_mead(obj_full, x0_full, max_iter=1000, 
                                   time_limit=time_limit - current_time - 2,
                                   start_time=current_time)
        polished_points = x_opt.reshape(13, 2)
        
        # Check if polished is better
        if soft_min_score(polished_points, 0.005) < soft_min_score(final_points, 0.005):
            final_points = polished_points
    
    return final_points

def heilbronn_convex13() -> np.ndarray:
    """
    Optimize 13 points to maximize the minimum triangle area ratio.
    Uses barycentric coordinate optimization with fixed regular 13-gon hull,
    plus multi-start Nelder-Mead as fallback.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 10.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Fixed regular 13-gon hull
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    hull_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Strategy 1: Barycentric optimization with different numbers of interior points
    # Try various configurations: 1-10 interior points
    # Allocate time: 40% to barycentric, 30% to multi-start NM, 20% to polish, 10% reserve
    
    bary_deadline = start_time + (deadline - start_time) * 0.4
    
    # Try different numbers of interior points
    for n_interior in [1, 2, 3, 4, 5, 6, 7, 8]:
        current_time = time.time()
        if current_time > bary_deadline - 5:
            break
        
        time_for_this = min(bary_deadline - current_time - 2, 30.0)
        
        if time_for_this < 5:
            break
        
        # Create initial configuration: hull points + interior points at center
        # For n_interior interior points, we have 13 - n_interior hull points
        # Wait, we need exactly 13 points total.
        # So if we have n_interior interior points, we have 13 - n_interior hull points.
        # But our hull is fixed at 13-gon... 
        # Actually, let's reconsider: we use all 13 hull points, and some are "interior"
        # in the sense that they're moved inward.
        
        # Better approach: Use all 13 points, but optimize their positions
        # with the constraint that they stay within the convex hull of the regular 13-gon.
        
        # For simplicity, let's just optimize all 13 points directly with NM
        # but use the barycentric structure as initialization.
        
        # Create initial points: regular 13-gon with some points moved inward
        init_pts = hull_pts.copy()
        # Move some points toward center
        n_move = min(n_interior, 13)
        for i in range(n_move):
            # Move point i toward center by a random amount
            move_factor = rng.uniform(0.3, 0.8)
            init_pts[i] = (1 - move_factor) * init_pts[i]
        
        # Optimize
        def obj_func(x_flat):
            pts = x_flat.reshape(13, 2)
            return soft_min_score(pts, temperature=0.01)
        
        x0 = init_pts.flatten()
        x_opt, f_opt = nelder_mead(obj_func, x0, max_iter=1500, 
                                   time_limit=time_for_this, start_time=current_time)
        
        pts_opt = x_opt.reshape(13, 2)
        score = compute_score(pts_opt)
        
        if score > best_score:
            best_score = score
            best_points = pts_opt.copy()
    
    # Strategy 2: Multi-start Nelder-Mead with diverse initializations
    nm_deadline = start_time + (deadline - start_time) * 0.7
    
    candidates = generate_initial_candidates(rng)
    
    # Add perturbed versions
    for base in candidates[:4]:
        for scale in [0.1, 0.3]:
            perturbed = base + rng.normal(0, scale, size=base.shape)
            candidates.append(perturbed)
    
    rng.shuffle(candidates)
    
    n_candidates = len(candidates)
    time_per_candidate = max(5.0, (nm_deadline - time.time()) / n_candidates)
    
    for i, init_pts in enumerate(candidates):
        current_time = time.time()
        if current_time > nm_deadline:
            break
        
        # Normalize
        init_pts_centered = init_pts - np.mean(init_pts, axis=0)
        max_norm = np.max(np.linalg.norm(init_pts_centered, axis=1))
        if max_norm > 1e-6:
            init_pts_scaled = init_pts_centered / max_norm
        else:
            init_pts_scaled = init_pts_centered
        
        x0 = init_pts_scaled.flatten()
        
        def obj_func(x_flat):
            pts = x_flat.reshape(13, 2)
            return soft_min_score(pts, temperature=0.01)
        
        time_for_this = min(time_per_candidate, nm_deadline - current_time - 1)
        if time_for_this < 3:
            break
            
        x_opt, f_opt = nelder_mead(obj_func, x0, max_iter=2000, 
                                   time_limit=time_for_this, start_time=current_time)
        
        pts_opt = x_opt.reshape(13, 2)
        score = compute_score(pts_opt)
        
        if score > best_score:
            best_score = score
            best_points = pts_opt.copy()
    
    # Polish the best point found
    if best_points is not None:
        polish_deadline = deadline - 5.0
        current_time = time.time()
        if current_time < polish_deadline:
            x0 = best_points.flatten()
            
            def obj_func_polish(x_flat):
                pts = x_flat.reshape(13, 2)
                return soft_min_score(pts, temperature=0.005)
                
            x_opt, f_opt = nelder_mead(obj_func_polish, x0, max_iter=2000, 
                                       time_limit=polish_deadline - current_time,
                                       start_time=current_time)
            pts_polished = x_opt.reshape(13, 2)
            
            score = compute_score(pts_polished)
            if score > best_score:
                best_score = score
                best_points = pts_polished.copy()
                
            # Local search
            current_time = time.time()
            if current_time < polish_deadline - 2:
                step_size = 0.01
                local_best = best_points.copy()
                local_score = best_score
                
                while time.time() < polish_deadline - 2:
                    improved = False
                    for idx in range(13):
                        for dx in [-step_size, 0, step_size]:
                            for dy in [-step_size, 0, step_size]:
                                if dx == 0 and dy == 0:
                                    continue
                                new_pts = local_best.copy()
                                new_pts[idx, 0] += dx
                                new_pts[idx, 1] += dy
                                s = compute_score(new_pts)
                                if s > local_score:
                                    local_best = new_pts
                                    local_score = s
                                    improved = True
                    if not improved:
                        step_size *= 0.5
                        if step_size < 1e-4:
                            break
                            
                if local_score > best_score:
                    best_score = local_score
                    best_points = local_best.copy()

    # Fallback
    if best_points is None:
        best_points = hull_pts
        
    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        best_points = hull_pts
        
    return best_points

# EVOLVE-BLOCK-END
