# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time
import itertools

def _precompute_indices(n):
    """Precompute triangle indices for efficiency."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_dets(points, idx):
    """Compute 2 * signed area (determinant) for all triangles."""
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return det

def _compute_hull_area(points):
    """Compute convex hull area safely."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
        
    A_hull = _compute_hull_area(points)
    if A_hull <= 1e-12:
        return 0.0
        
    idx = _precompute_indices(n)
    dets = _compute_dets(points, idx)
    A_min = 0.5 * np.min(np.abs(dets))
    
    return A_min / A_hull

def _delaunay_repair(points, idx, max_iters=500, time_limit=500):
    """
    Delaunay-aware local repair strategy.
    1. Compute Delaunay triangulation.
    2. Identify the minimum area triangle.
    3. Move one of its vertices to increase the area of this triangle,
       while checking that no other triangle area decreases below the new minimum.
    """
    n = len(points)
    pts = points.copy()
    start_time = time.time()
    
    current_score = compute_score(pts)
    if current_score <= 0:
        return pts, current_score

    A_hull = _compute_hull_area(pts)
    dets = _compute_dets(pts, idx)
    A_min = 0.5 * np.min(np.abs(dets))
    
    # Indices of triangles in Delaunay triangulation
    try:
        tri = Delaunay(pts)
        delaunay_indices = tri.simplices
    except Exception:
        return pts, current_score

    # Map from triangle vertices (sorted tuple) to index in delaunay_indices
    # This is complex, so we just iterate over delaunay triangles.
    
    best_pts = pts
    best_score = current_score
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Recompute score and min area triangle
        score = compute_score(best_pts)
        if score <= 0:
            break
            
        A_hull_cur = _compute_hull_area(best_pts)
        dets_cur = _compute_dets(best_pts, idx)
        A_min_cur = 0.5 * np.min(np.abs(dets_cur))
        
        # Find the index of the minimum area triangle in the full set
        min_idx_full = np.argmin(np.abs(dets_cur))
        min_tri_vertices = idx[min_idx_full]
        
        # Check if this triangle is in the Delaunay triangulation
        # We want to repair the "worst" triangle. If it's not in Delaunay, 
        # it's likely an ear or internal, but Delaunay covers the convex hull.
        # Actually, the minimum area triangle might not be in the Delaunay triangulation 
        # if the points are not in convex position? No, all points are in the set.
        # Delaunay triangulates the convex hull. If a point is inside, it's part of triangles.
        
        # Let's try to move one of the vertices of the minimum area triangle.
        # We will try moving each of the 3 vertices slightly towards the centroid of the triangle 
        # or in a direction that increases the area.
        
        # Direction to increase area:
        # For triangle (A, B, C), area is 0.5 * |det(B-A, C-A)|.
        # If we move A, we change the base or height.
        # A simple heuristic: move A towards the midpoint of BC? No, that decreases area.
        # Move A away from BC?
        
        # Better: Use a local line search.
        # For each vertex v in the min triangle, try moving it in the direction of the gradient of the area.
        
        moved = False
        for v_idx in min_tri_vertices:
            # Direction: gradient of area w.r.t. v
            # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            # Let v = (x1, y1). 
            # grad_x = 0.5 * (y2 - y3)
            # grad_y = 0.5 * (x3 - x2)
            # This is the vector perpendicular to the opposite edge.
            
            other1, other2 = [v for v in min_tri_vertices if v != v_idx]
            p_other1 = best_pts[other1]
            p_other2 = best_pts[other2]
            
            # Vector from v to the midpoint of the opposite edge?
            # Or just the normal to the opposite edge.
            edge_vec = p_other2 - p_other1
            # Normal to edge_vec
            normal = np.array([-edge_vec[1], edge_vec[0]])
            
            # Determine sign to increase area
            # Current area vector (signed)
            v1 = p_other1 - best_pts[v_idx]
            v2 = p_other2 - best_pts[v_idx]
            signed_det = v1[0]*v2[1] - v1[1]*v2[0]
            
            # If we move v in direction d, new det is det + dot(grad, d)
            # We want to increase |det|.
            # If signed_det > 0, we want to increase det.
            # If signed_det < 0, we want to decrease det (make more negative).
            
            # The gradient of the signed area w.r.t. v is actually the normal to the opposite edge.
            # Let's check:
            # A = 0.5 * (x1(y2-y3) + ...)
            # dA/dx1 = 0.5(y2-y3)
            # dA/dy1 = 0.5(x3-x2)
            # This is 0.5 * (p_other2 - p_other1) rotated 90 deg?
            # p_other2 - p_other1 = (x2-x1, y2-y1) ... wait indices are relative to triangle vertices.
            # Let triangle be (0, 1, 2). Moving 0.
            # dA/dx0 = 0.5(y1-y2)
            # dA/dy0 = 0.5(x2-x1)
            # Vector: 0.5 * (-(y1-y2), (x2-x1)) ? No.
            # Vector: (0.5(y1-y2), 0.5(x2-x1)).
            # This is perpendicular to (x1-x2, y1-y2).
            
            grad = np.array([0.5 * (best_pts[other1][1] - best_pts[other2][1]),
                             0.5 * (best_pts[other2][0] - best_pts[other1][0])])
            
            # Ensure we move in the direction that increases the absolute area
            if signed_det < 0:
                grad = -grad
                
            # Line search
            best_step = 0.0
            best_local_score = score
            
            # Try a few step sizes
            # Scale step based on point spread
            scale = np.std(best_pts)
            for step_size in [0.01 * scale, 0.005 * scale, 0.001 * scale, 0.02 * scale]:
                new_pts = best_pts.copy()
                new_pts[v_idx] += step_size * grad
                
                # Check validity
                if not np.all(np.isfinite(new_pts)):
                    continue
                    
                new_score = compute_score(new_pts)
                
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_step = step_size
                    best_new_pts = new_pts.copy()
            
            if best_step > 0:
                best_pts = best_new_pts
                best_score = best_local_score
                moved = True
                break # Restart loop with new configuration
        
        if not moved:
            # If no local improvement found with this simple gradient step, try Nelder-Mead polish
            break

    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    idx = _precompute_indices(n)
    
    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric polygons (1 center, 5 inner, 7 outer)
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    inits.append(np.vstack([center, inner_pts, outer_pts]))
    
    # 3. Random
    rng = np.random.default_rng(seed=42)
    inits.append(rng.random((n, 2)) * 2 - 1)
    
    # 4. Grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:13]
    pts += rng.random((13, 2)) * 0.05
    inits.append(pts)
    
    # 5. Hexagonal lattice snippet
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / (np.abs(hex_pts).max() + 1e-9)
    hex_pts = hex_pts[:13]
    inits.append(hex_pts)
    
    # 6. Fibonacci spiral
    angles_fib = np.arange(n) * 2 * np.pi * 0.6180339887
    radii_fib = np.sqrt(np.arange(1, n + 1) / n)
    inits.append(np.column_stack([radii_fib * np.cos(angles_fib), radii_fib * np.sin(angles_fib)]))

    # Strategy:
    # 1. For each init, run Delaunay repair.
    # 2. Then run Nelder-Mead polish on the result.
    
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Phase 1: Delaunay Repair
        try:
            repaired_pts, repaired_score = _delaunay_repair(init_pts, idx, max_iters=100, time_limit=60)
            
            if repaired_score > best_score:
                best_score = repaired_score
                best_points = repaired_pts.copy()
            else:
                # If repair didn't improve, maybe the init was better?
                init_score = compute_score(init_pts)
                if init_score > best_score:
                    best_score = init_score
                    best_points = init_pts.copy()
        except Exception:
            init_score = compute_score(init_pts)
            if init_score > best_score:
                best_score = init_score
                best_points = init_pts.copy()

        # Phase 2: Nelder-Mead Polish on the best result so far
        if best_points is not None:
            base_pts = best_points.copy()
            
            def nm_objective(x):
                pts = x.reshape((n, 2))
                s = compute_score(pts)
                if s == 0.0:
                    return 1.0
                return -s
            
            try:
                res_nm = minimize(nm_objective, base_pts.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
                final_pts_nm = res_nm.x.reshape((n, 2))
                final_score_nm = compute_score(final_pts_nm)
                
                if final_score_nm > best_score:
                    best_score = final_score_nm
                    best_points = final_pts_nm.copy()
                    
            except Exception:
                pass

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
