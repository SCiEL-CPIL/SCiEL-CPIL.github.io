"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
import time

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """Compute all triangle areas and return (min_area, active_indices)."""
    n = len(points)
    areas = []
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
                indices.append((i, j, k))
    areas = np.array(areas)
    min_area = np.min(areas)
    # Find triangles within 10% of minimum as "active"
    threshold = min_area * 1.1
    active_mask = areas <= threshold
    active_indices = [indices[i] for i in range(len(indices)) if active_mask[i]]
    return min_area, active_indices

def _solve_equi_area(points, n_active=5, max_iter=100, tol=1e-8):
    """
    Solve for points where n_active triangles have equal area A,
    and convex hull area = 1, using damped Newton-Raphson.
    
    Unknowns: 26 coordinates (13 points x 2)
    Equations: n_active area equations + 1 hull area equation
    """
    n = len(points)
    x = points.flatten().copy()
    
    for iteration in range(max_iter):
        pts = x.reshape(-1, 2)
        
        # Compute all triangle areas
        min_area, active_indices = _all_triangle_areas(pts)
        
        # Take the n_active smallest triangles
        all_areas = []
        all_indices = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                    all_areas.append(area)
                    all_indices.append((i, j, k))
        
        all_areas = np.array(all_areas)
        # Sort by area and take the n_active smallest
        sorted_idx = np.argsort(all_areas)
        active_idx = sorted_idx[:n_active]
        active_triangles = [all_indices[i] for i in active_idx]
        target_A = all_areas[active_idx[0]]  # Target area
        
        # Build residual vector: active triangle areas - target_A, and hull_area - 1
        residuals = []
        jacobian = np.zeros((n_active + 1, 2*n))
        
        # Residuals and Jacobian for active triangles
        for m, (i, j, k) in enumerate(active_triangles):
            p1 = pts[i]
            p2 = pts[j]
            p3 = pts[k]
            # Area = 0.5 * |cross(p2-p1, p3-p1)|
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
            area = 0.5 * abs(cross)
            residuals.append(area - target_A)
            
            # Jacobian of area w.r.t. coordinates
            # d(area)/d(x_i) = 0.5 * (p3[1]-p1[1]) * sign(cross)
            # d(area)/d(y_i) = -0.5 * (p3[0]-p1[0]) * sign(cross)
            # d(area)/d(x_j) = 0.5 * (p1[1]-p3[1]) * sign(cross)
            # d(area)/d(y_j) = -0.5 * (p1[0]-p3[0]) * sign(cross)
            # d(area)/d(x_k) = 0.5 * (p2[1]-p1[1]) * sign(cross)
            # d(area)/d(y_k) = -0.5 * (p2[0]-p1[0]) * sign(cross)
            
            sgn = np.sign(cross) if abs(cross) > 1e-15 else 1.0
            jacobian[m, 2*i] = 0.5 * (p3[1]-p1[1]) * sgn
            jacobian[m, 2*i+1] = -0.5 * (p3[0]-p1[0]) * sgn
            jacobian[m, 2*j] = 0.5 * (p1[1]-p3[1]) * sgn
            jacobian[m, 2*j+1] = -0.5 * (p1[0]-p3[0]) * sgn
            jacobian[m, 2*k] = 0.5 * (p2[1]-p1[1]) * sgn
            jacobian[m, 2*k+1] = -0.5 * (p2[0]-p1[0]) * sgn
        
        # Residual and Jacobian for hull area constraint
        hull_area = _convex_hull_area(pts)
        residuals.append(hull_area - 1.0)
        
        # Approximate Jacobian for hull area using finite differences
        for idx in range(2*n):
            x_pert = x.copy()
            x_pert[idx] += 1e-7
            pts_pert = x_pert.reshape(-1, 2)
            hull_area_pert = _convex_hull_area(pts_pert)
            jacobian[n_active, idx] = (hull_area_pert - hull_area) / 1e-7
        
        residuals = np.array(residuals)
        
        # Check convergence
        if np.max(np.abs(residuals)) < tol:
            break
        
        # Solve J * dx = -residuals using least squares (underdetermined system)
        try:
            dx, residuals_ls, rank, sv = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        # Damped step: try full step first, reduce if needed
        step_size = 1.0
        for _ in range(20):
            x_new = x + step_size * dx
            pts_new = x_new.reshape(-1, 2)
            
            # Check if this step improves the objective
            min_area_new, _ = _all_triangle_areas(pts_new)
            hull_area_new = _convex_hull_area(pts_new)
            
            if hull_area_new > 1e-10:
                score_new = min_area_new / hull_area_new
            else:
                score_new = 0.0
            
            # Also compute current score
            hull_area_cur = _convex_hull_area(pts)
            if hull_area_cur > 1e-10:
                score_cur = min_area / hull_area_cur
            else:
                score_cur = 0.0
            
            if score_new >= score_cur * 0.95:  # Allow slight decrease
                x = x_new
                break
            else:
                step_size *= 0.5
        
        if step_size < 1e-10:
            break
    
    return x.reshape(-1, 2)

def construct():
    """
    Place 13 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Approach: Use equi-area Newton-Raphson to find configurations where multiple
    triangles achieve the minimum area simultaneously, with convex hull area = 1.
    """
    best_score = 0.0
    best_points = None
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(reg_points.copy())
    
    # 2. Perturbed regular polygons with different seeds
    rng = np.random.RandomState(123)
    for seed in range(5):
        r = np.random.RandomState(seed)
        perturbed = reg_points.copy()
        for i in range(N):
            perturbed[i] += r.rand(2) * 0.15
        initial_configs.append(perturbed)
    
    # 3. Grid-like arrangement
    grid_points = []
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            grid_points.append([i/3.0, j/3.0])
    grid_points = np.array(grid_points)
    initial_configs.append(grid_points.copy())
    
    # 4. Random points
    for seed in range(3):
        r = np.random.RandomState(1000 + seed)
        rand_points = r.rand(N, 2)
        initial_configs.append(rand_points)
    
    # For each initial config, run equi-area Newton
    for init_points in initial_configs:
        # Normalize to convex hull area 1
        hull_area = _convex_hull_area(init_points)
        if hull_area > 1e-10:
            init_points = init_points / np.sqrt(hull_area)
        else:
            continue
        
        # Run equi-area solver with different numbers of active triangles
        for n_active in [3, 5, 7]:
            try:
                opt_points = _solve_equi_area(init_points, n_active=n_active, max_iter=80)
                hull_area = _convex_hull_area(opt_points)
                if hull_area > 1e-10:
                    min_area, _ = _all_triangle_areas(opt_points)
                    score = min_area / hull_area
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
            except Exception:
                pass
    
    # If no good result, fall back to regular polygon
    if best_points is None or best_score < 0.005:
        best_points = reg_points.copy()
        hull_area = _convex_hull_area(best_points)
        if hull_area > 1e-10:
            best_points = best_points / np.sqrt(hull_area)
        min_area, _ = _all_triangle_areas(best_points)
        best_score = min_area / _convex_hull_area(best_points)
    
    # Final normalization
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        best_points = best_points / np.sqrt(hull_area)
    
    return best_points
# EVOLVE-BLOCK-END
