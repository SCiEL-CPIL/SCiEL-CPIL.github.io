# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
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

def delaunay_repair(points, max_iters=100, learning_rate=0.1, time_limit=None, start_time=None):
    """
    Delaunay triangulation repair strategy:
    1. Compute Delaunay triangulation
    2. Find smallest-area triangles
    3. Move vertices of small triangles toward centroids of adjacent triangles
    4. Repeat
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 30.0
    
    pts = points.copy()
    best_pts = pts.copy()
    best_score = compute_score(pts)
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Check if all points are in general position (no collinear issues)
        if not np.all(np.isfinite(pts)):
            break
        
        # Compute Delaunay triangulation
        try:
            delaunay = Delaunay(pts)
        except Exception:
            break
        
        # Compute areas of Delaunay triangles
        tri_idx = delaunay.simplices  # shape (n_tri, 3)
        tri_pts = pts[tri_idx]  # shape (n_tri, 3, 2)
        
        # Compute area of each Delaunay triangle
        v1 = tri_pts[:, 1] - tri_pts[:, 0]
        v2 = tri_pts[:, 2] - tri_pts[:, 0]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        tri_areas = np.abs(cross) * 0.5
        
        # Find the smallest triangles
        min_area_idx = np.argmin(tri_areas)
        min_area = tri_areas[min_area_idx]
        
        # Find all triangles with area close to minimum (within 20%)
        threshold = min_area * 1.2
        small_tri_mask = tri_areas < threshold
        small_tri_indices = np.where(small_tri_mask)[0]
        
        if len(small_tri_indices) == 0:
            break
        
        # For each small triangle, compute the centroid of adjacent triangles
        # and move vertices toward that centroid
        displacements = np.zeros_like(pts)
        
        for tri_i in small_tri_indices:
            tri = tri_idx[tri_i]  # 3 vertices
            tri_centroid = np.mean(pts[tri], axis=0)
            
            # Find adjacent triangles (sharing an edge)
            adjacent_centroids = []
            for edge in range(3):
                v1_idx = tri[edge]
                v2_idx = tri[(edge + 1) % 3]
                
                # Find triangles that share this edge
                for other_tri_i, other_tri in enumerate(tri_idx):
                    if other_tri_i == tri_i:
                        continue
                    # Check if other_tri shares the edge (v1_idx, v2_idx)
                    if ((v1_idx in other_tri) and (v2_idx in other_tri)):
                        other_centroid = np.mean(pts[other_tri], axis=0)
                        adjacent_centroids.append(other_centroid)
            
            if len(adjacent_centroids) > 0:
                adj_centroid = np.mean(adjacent_centroids, axis=0)
                
                # Move each vertex of the small triangle toward the adjacent centroid
                for v_idx in tri:
                    # Only move if this vertex is not on the convex hull
                    # (hull vertices should stay fixed to preserve hull shape)
                    # For now, move all vertices but with smaller step for hull vertices
                    direction = adj_centroid - pts[v_idx]
                    dist = np.linalg.norm(direction)
                    if dist > 1e-10:
                        direction = direction / dist
                        displacements[v_idx] += learning_rate * direction
            
            # Also move toward the global centroid of all small triangles
            global_small_centroid = np.mean(pts[tri_idx[small_tri_indices]].reshape(-1, 2), axis=0)
            for v_idx in tri:
                direction = global_small_centroid - pts[v_idx]
                dist = np.linalg.norm(direction)
                if dist > 1e-10:
                    direction = direction / dist
                    displacements[v_idx] += 0.5 * learning_rate * direction
        
        # Apply displacements
        new_pts = pts + displacements
        
        # Ensure points remain in general position and finite
        if not np.all(np.isfinite(new_pts)):
            break
        
        # Compute score
        new_score = compute_score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # If score decreased, reduce learning rate and retry
            learning_rate *= 0.8
            if learning_rate < 0.001:
                break
    
    return best_pts, best_score

def nelder_mead(f, x0, max_iter=5000, tol=1e-8, time_limit=None, start_time=None):
    """
    Nelder-Mead optimization algorithm with improved parameters.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 60.0
        
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    # Standard Nelder-Mead initialization: axis-aligned simplex
    step = 0.05
    for i in range(n):
        x_new = x0.copy()
        if abs(x0[i]) > 1e-6:
            x_new[i] *= 1.02
        else:
            x_new[i] += step
        simplex.append(x_new)
        
    # Parameters - tuned for better convergence
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
    stagnation_count = 0
    
    for iteration in range(max_iter):
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
            stagnation_count = 0
        elif f_reflected < values[-2]:
            # Accept reflection
            simplex[-1] = x_reflected
            values[-1] = f_reflected
            stagnation_count = 0
        else:
            # Contraction
            x_contracted = centroid + rho * (simplex[-1] - centroid)
            f_contracted = f(x_contracted)
            if f_contracted < values[-1]:
                simplex[-1] = x_contracted
                values[-1] = f_contracted
                stagnation_count = 0
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = f(simplex[i])
                stagnation_count += 1
        
        # Re-sort
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if values[0] < best_f:
            best_f = values[0]
            best_x = simplex[0].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # If stuck for too long, perturb the worst point
        if stagnation_count > 50:
            simplex[-1] = best_x + np.random.normal(0, 0.01, size=n)
            values[-1] = f(simplex[-1])
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            stagnation_count = 0
            
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
    # A 4x4 grid has 16 points. We want 13.
    # Let's use a distorted grid.
    xs = np.array([-1, -0.33, 0.33, 1.0])
    ys = np.array([-1, -0.33, 0.33, 1.0])
    grid = np.array([[x, y] for x in xs for y in ys])
    # Keep 13 points: remove corners? Or just take first 13.
    # Better: Randomly select 13 from a 4x4 grid and optimize.
    # For initialization, just take first 13.
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

def heilbronn_convex13() -> np.ndarray:
    """
    Optimize 13 points to maximize the minimum triangle area ratio.
    Uses Delaunay triangulation repair strategy as the primary method.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 10.0 # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate initial candidates
    candidates = generate_initial_candidates(rng)
    
    # Add some perturbed versions of the best structured candidates
    for base in candidates[:4]:
        for scale in [0.1, 0.3]:
            perturbed = base + rng.normal(0, scale, size=base.shape)
            candidates.append(perturbed)
    
    # Shuffle candidates to ensure variety
    rng.shuffle(candidates)
    
    n_candidates = len(candidates)
    
    # Time allocation: 70% for Delaunay repair, 20% for NM polish, 10% reserve
    delaunay_deadline = start_time + (deadline - start_time) * 0.7
    polish_deadline = deadline - 5.0
    
    # Phase 1: Delaunay repair on multiple initial candidates
    time_per_candidate = max(10.0, (delaunay_deadline - start_time) / n_candidates)
    
    for i, init_pts in enumerate(candidates):
        current_time = time.time()
        if current_time > delaunay_deadline:
            break
        
        # Normalize initial points to unit circle-ish scale for better behavior
        init_pts_centered = init_pts - np.mean(init_pts, axis=0)
        max_norm = np.max(np.linalg.norm(init_pts_centered, axis=1))
        if max_norm > 1e-6:
            init_pts_scaled = init_pts_centered / max_norm
        else:
            init_pts_scaled = init_pts_centered
        
        # Run Delaunay repair
        repair_deadline = min(delaunay_deadline, current_time + time_per_candidate)
        pts_repaired, score_repaired = delaunay_repair(
            init_pts_scaled, 
            max_iters=50, 
            learning_rate=0.1,
            time_limit=repair_deadline - current_time,
            start_time=current_time
        )
        
        if score_repaired > best_score:
            best_score = score_repaired
            best_points = pts_repaired.copy()
    
    # Phase 2: Nelder-Mead polish on the best result
    if best_points is not None:
        current_time = time.time()
        if current_time < polish_deadline:
            x0 = best_points.flatten()
            
            def obj_func_polish(x_flat):
                pts = x_flat.reshape(13, 2)
                # Use a tighter soft-min
                return -compute_score(pts)
                
            nm_deadline = polish_deadline
            x_opt, f_opt = nelder_mead(obj_func_polish, x0, max_iter=3000, time_limit=nm_deadline - current_time, start_time=current_time)
            pts_polished = x_opt.reshape(13, 2)
            
            score = compute_score(pts_polished)
            if score > best_score:
                best_score = score
                best_points = pts_polished.copy()
                
            # Final high-precision local search with finer steps
            current_time = time.time()
            if current_time < polish_deadline - 2:
                local_best = best_points.copy()
                local_score = best_score
                step_sizes = [0.005, 0.002, 0.001, 0.0005, 0.0002]
                
                for step_size in step_sizes:
                    if time.time() > polish_deadline - 2:
                        break
                    improved = True
                    while improved and time.time() < polish_deadline - 2:
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
                            
                if local_score > best_score:
                    best_score = local_score
                    best_points = local_best.copy()

    # Fallback
    if best_points is None:
        best_points = candidates[0]
        
    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        best_points = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 13, endpoint=False)), np.sin(np.linspace(0, 2*np.pi, 13, endpoint=False))])
        
    return best_points

# EVOLVE-BLOCK-END
