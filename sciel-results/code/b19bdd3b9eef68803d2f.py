import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def project_points(points, normals):
    """Project points onto normals. Returns array of shape (N_points, N_normals)."""
    # points: (N, 2), normals: (M, 2)
    return points @ normals.T

def sat_overlap(v1, v2):
    """
    Check if two convex polygons (v1, v2) overlap using Separating Axis Theorem.
    Returns 0 if disjoint, positive value (approximate penetration depth) if overlapping.
    """
    # Get edges and normals for v1 and v2
    def get_normals(poly):
        normals = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i+1) % len(poly)]
            edge = p2 - p1
            # Normal perpendicular to edge
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                normals.append(n / norm)
        return normals

    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_overlap = float('inf')
    
    for n in all_normals:
        # Project v1 and v2 onto n
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            return 0.0 # Disjoint
        
        # Calculate overlap depth
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def compute_min_side_and_center(all_verts):
    """
    Compute minimal side length of a hexagon (angle 0) containing all_verts,
    and its optimal center.
    """
    sqrt3 = np.sqrt(3)
    # Normals for angle 0 hexagon: 0, 60, 120, 180, 240, 300 degrees
    # Note: The "width" directions are the normals to the faces.
    # For a hexagon with vertices at 0, 60..., the face normals are at 30, 90...
    # Wait, let's check the definition.
    # Vertices at theta = i * pi/3.
    # Edge between i=0 and i=1 is at angle 30 deg from x-axis?
    # Midpoint of edge 0-1: angle 30. Normal is 30.
    # So normals are at 30, 90, 150, 210, 270, 330.
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # We want to find center (cx, cy) that minimizes S
    # S = max_i (max_v | (v - c) dot n_i |) / (sqrt(3)/2)
    # This is a convex optimization problem in c.
    # We can use a simple grid search or Nelder-Mead for c.
    # Since N is small, let's use a coarse grid search around the centroid.
    
    cx0 = all_verts[:, 0].mean()
    cy0 = all_verts[:, 1].mean()
    
    best_side = float('inf')
    best_cx = cx0
    best_cy = cy0
    
    # Coarse search
    for dx in np.linspace(-2, 2, 9):
        for dy in np.linspace(-2, 2, 9):
            cx = cx0 + dx
            cy = cy0 + dy
            c = np.array([cx, cy])
            
            # Project vertices relative to center
            rel_verts = all_verts - c
            projs = rel_verts @ normals.T
            
            # Max absolute projection for each normal
            max_abs_projs = np.abs(projs).max(axis=0)
            
            # Required side for each normal
            sides = max_abs_projs / (sqrt3 / 2)
            
            side = sides.max()
            
            if side < best_side:
                best_side = side
                best_cx = cx
                best_cy = cy
                
    # Fine search around best
    for dx in np.linspace(-0.2, 0.2, 5):
        for dy in np.linspace(-0.2, 0.2, 5):
            cx = best_cx + dx
            cy = best_cy + dy
            c = np.array([cx, cy])
            
            rel_verts = all_verts - c
            projs = rel_verts @ normals.T
            max_abs_projs = np.abs(projs).max(axis=0)
            sides = max_abs_projs / (sqrt3 / 2)
            side = sides.max()
            
            if side < best_side:
                best_side = side
                best_cx = cx
                best_cy = cy
                
    return best_side, best_cx, best_cy

def objective(params):
    """
    Objective function for differential_evolution.
    params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, cx, cy]
    Total 35 params.
    """
    # Unpack
    inner_params = params[:33]
    cx = params[33]
    cy = params[34]
    
    # Reconstruct inner hexagons
    centers = inner_params[0::3]
    ys = inner_params[1::3]
    angles = inner_params[2::3]
    
    # Check for overlaps
    total_overlap = 0.0
    verts_list = []
    for i in range(N):
        v = hex_vertices(centers[i], ys[i], 1.0, angles[i])
        verts_list.append(v)
        
    for i in range(N):
        for j in range(i + 1, N):
            ov = sat_overlap(verts_list[i], verts_list[j])
            if ov > 0:
                total_overlap += ov
                # Early exit if overlap is huge
                if total_overlap > 1.0:
                    return 1000.0 + total_overlap
    
    if total_overlap > 1e-6:
        return 100.0 + total_overlap * 100
    
    # Compute minimal side length
    all_verts = np.vstack(verts_list)
    side, _, _ = compute_min_side_and_center(all_verts)
    
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: Hexagonal grid
    sqrt3 = np.sqrt(3)
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    init_centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        init_centers.append((x, y))
        
    init_params = []
    for cx, cy in init_centers:
        init_params.extend([cx, cy, 0.0])
        
    # Initial outer center estimate
    init_cx = 0.0
    init_cy = 0.0
    init_params.extend([init_cx, init_cy])
    
    # Bounds
    # Inner centers: roughly within [-5, 5]
    # Inner angles: [0, 60]
    # Outer center: [-5, 5]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
        bounds.append((0.0, 60.0)) # angle
    
    bounds.append((-5.0, 5.0)) # outer cx
    bounds.append((-5.0, 5.0)) # outer cy
    
    # Run Differential Evolution
    # maxiter and popsize tuned for speed
    result = differential_evolution(
        objective, 
        bounds, 
        x0=np.array(init_params),
        maxiter=50, 
        popsize=15,
        tol=1e-5,
        seed=42,
        polish=True
    )
    
    if time.time() > deadline - 10:
        # Fallback to initial grid if DE took too long
        best_params = np.array(init_params)
        best_side = objective(best_params)
    else:
        best_params = result.x
        best_side = result.fun
        
    # Extract best solution
    inner_centers = best_params[0::3]
    inner_ys = best_params[1::3]
    inner_angles = best_params[2::3]
    outer_cx = best_params[33]
    outer_cy = best_params[34]
    
    # Verify and refine
    verts_list = []
    for i in range(N):
        v = hex_vertices(inner_centers[i], inner_ys[i], 1.0, inner_angles[i])
        verts_list.append(v)
        
    # Check validity
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if sat_overlap(verts_list[i], verts_list[j]) > 1e-5:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If invalid, fall back to the initial grid
        inner_centers = np.array([c[0] for c in init_centers])
        inner_ys = np.array([c[1] for c in init_centers])
        inner_angles = np.zeros(N)
        outer_cx = 0.0
        outer_cy = 0.0
        verts_list = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in init_centers]
        
    all_verts = np.vstack(verts_list)
    
    # Final precise calculation of side and center
    # We use the center found by DE, but refine it
    side, cx_final, cy_final = compute_min_side_and_center(all_verts)
    
    # Ensure no violation with a tiny epsilon
    # Check max projection
    sqrt3 = np.sqrt(3)
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    c_final = np.array([cx_final, cy_final])
    rel_verts = all_verts - c_final
    projs = rel_verts @ normals.T
    max_abs_projs = np.abs(projs).max(axis=0)
    required_side = (max_abs_projs / (sqrt3 / 2)).max()
    
    if required_side > side:
        side = required_side
        
    # Add small safety margin
    side += 1e-7
    
    inner = [[float(inner_centers[i]), float(inner_ys[i]), float(inner_angles[i])] for i in range(N)]
    
    return inner, (float(cx_final), float(cy_final)), float(side), 0.0

