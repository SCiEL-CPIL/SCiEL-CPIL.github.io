import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) side s rotated angle_deg."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    for i in range(6):
        theta = i * np.pi / 3
        if lx * np.cos(theta) + ly * np.sin(theta) > s * np.sqrt(3) / 2 + 1e-12:
            return False
    return True

def hexagons_disjoint(v1, v2, s, eps=1e-9):
    """Check if two hexagons (vertex arrays) are disjoint (no interior intersection)."""
    def point_in_poly(px, py, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    for v in v1:
        if point_in_poly(v[0], v[1], v2):
            return False
    for v in v2:
        if point_in_poly(v[0], v[1], v1):
            return False
    def seg_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
    
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if seg_intersect(p1, p2, p3, p4):
                return False
    return True

def min_enclosing_side(points, center, angle_deg):
    """Compute minimum side length of hexagon at center/angle to contain all points."""
    a = np.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + np.pi/6 + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    
    required = np.zeros(6)
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        required[i] = max(proj.max(), -proj.min())
    
    return np.max(required) / (np.sqrt(3) / 2)

def compute_all_vertices(centers, angles, side=1.0):
    """Compute all vertices of all inner hexagons."""
    verts = []
    for i in range(len(centers)):
        v = hex_vertices(centers[i, 0], centers[i, 1], side, angles[i])
        verts.append(v)
    return np.vstack(verts)

def optimize_outer(centers, angles, max_iterations=100):
    """Optimize outer hexagon center and angle to minimize side length."""
    all_verts = compute_all_vertices(centers, angles)
    
    # Initial center: centroid of all vertices
    cx0 = all_verts[:, 0].mean()
    cy0 = all_verts[:, 1].mean()
    
    best_side = float('inf')
    best_center = (cx0, cy0)
    best_angle = 0.0
    
    # Try different angles
    for outer_angle in np.linspace(0, 60, 121):
        side = min_enclosing_side(all_verts, (cx0, cy0), outer_angle)
        if side < best_side:
            best_side = side
            best_center = (cx0, cy0)
            best_angle = outer_angle
    
    # Refine center using gradient-free optimization
    def obj(params):
        cx, cy, ang = params
        return min_enclosing_side(all_verts, (cx, cy), ang)
    
    x0 = np.array([best_center[0], best_center[1], best_angle])
    result = minimize(obj, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iterations, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if result.fun < best_side:
        best_center = (result.x[0], result.x[1])
        best_angle = result.x[2]
        best_side = result.fun
    
    return best_center, best_side, best_angle

def check_validity(centers, angles, outer_center, side, outer_angle, eps=1e-7):
    """Check that all inner hexagons are disjoint and inside the outer hexagon."""
    # Check containment
    all_verts = compute_all_vertices(centers, angles)
    a = np.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + np.pi/6 + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    dx = all_verts[:, 0] - outer_center[0]
    dy = all_verts[:, 1] - outer_center[1]
    
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        if proj.max() > side * np.sqrt(3) / 2 + eps:
            return False
        if -proj.min() > side * np.sqrt(3) / 2 + eps:
            return False
    
    # Check disjointness
    inner_verts = [hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                return False
    
    return True

def force_directed_relax(centers, angles, outer_center, side, outer_angle, max_iter=500, deadline=None):
    """Use force-directed relaxation to improve the packing."""
    centers = centers.copy()
    angles = angles.copy()
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(centers)
        angle_forces = np.zeros_like(angles)
        
        # Repulsion between overlapping hexagons
        for i in range(N):
            for j in range(i + 1, N):
                v1 = hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
                v2 = hex_vertices(centers[j, 0], centers[j, 1], 1.0, angles[j])
                
                # Check if they overlap
                overlap = not hexagons_disjoint(v1, v2, 1.0)
                if overlap:
                    # Push apart
                    dx = centers[j, 0] - centers[i, 0]
                    dy = centers[j, 1] - centers[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > 1e-10:
                        fx = dx / dist * 0.01
                        fy = dy / dist * 0.01
                        forces[i] += [-fx, -fy]
                        forces[j] += [fx, fy]
        
        # Attraction to center
        for i in range(N):
            forces[i, 0] += (outer_center[0] - centers[i, 0]) * 0.001
            forces[i, 1] += (outer_center[1] - centers[i, 1]) * 0.001
        
        # Update positions
        for i in range(N):
            centers[i, 0] += forces[i, 0]
            centers[i, 1] += forces[i, 1]
    
    return centers, angles

def simulate_annealing(centers, angles, max_iter=200, deadline=None):
    """Simulated annealing to find better arrangement."""
    centers = centers.copy()
    angles = angles.copy()
    
    # Get initial score
    outer_center, side, angle = optimize_outer(centers, angles, max_iterations=50)
    current_score = side
    
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_score = current_score
    
    T = 0.1
    T_min = 0.001
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Propose move
        i = np.random.randint(N)
        move_type = np.random.randint(3)
        
        old_cx, old_cy = centers[i, 0], centers[i, 1]
        old_angle = angles[i]
        
        if move_type == 0:
            centers[i, 0] += np.random.randn() * T
            centers[i, 1] += np.random.randn() * T
        elif move_type == 1:
            angles[i] += np.random.randn() * T
        else:
            centers[i, 0] += np.random.randn() * T
            centers[i, 1] += np.random.randn() * T
            angles[i] += np.random.randn() * T
        
        # Evaluate
        outer_center, new_side, new_angle = optimize_outer(centers, angles, max_iterations=30)
        new_score = new_side
        
        # Accept or reject
        if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_centers = centers.copy()
                best_angles = angles.copy()
        else:
            centers[i, 0] = old_cx
            centers[i, 1] = old_cy
            angles[i] = old_angle
        
        T = max(T_min, T * 0.95)
    
    return best_centers, best_angles, best_score

def generate_initial_configurations():
    """Generate several initial configurations."""
    configs = []
    sqrt3 = np.sqrt(3)
    
    # Config 1: Hex grid pattern (current)
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append([x, y])
    configs.append((np.array(centers), np.zeros(N)))
    
    # Config 2: Compact cluster with some rotation
    centers2 = []
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 1.2 if i > 0 else 0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers2.append([x, y])
    angles2 = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0])
    configs.append((np.array(centers2), angles2))
    
    # Config 3: Random perturbation of hex grid
    centers3 = np.array(centers) + np.random.randn(N, 2) * 0.1
    configs.append((centers3, np.zeros(N)))
    
    return configs

def get_valid_fallback():
    """Return a valid fallback configuration."""
    sqrt3 = np.sqrt(3)
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append([x, y])
    centers = np.array(centers)
    angles = np.zeros(N)
    
    outer_center, side, angle = optimize_outer(centers, angles, max_iterations=100)
    
    # Add safety margin
    side *= 1.001
    
    return centers, angles, outer_center, side, angle

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate initial configurations
    configs = generate_initial_configurations()
    
    best_side = float('inf')
    best_result = None
    
    for centers, angles in configs:
        if time.time() > deadline - 10:
            break
        
        # Optimize outer hexagon
        outer_center, side, angle = optimize_outer(centers, angles, max_iterations=100)
        
        # Check validity
        if check_validity(centers, angles, outer_center, side * 1.001, angle):
            if side < best_side:
                best_side = side
                best_result = (centers, angles, outer_center, side, angle)
        
        # Force-directed relaxation
        if time.time() < deadline - 30:
            centers_relaxed, angles_relaxed = force_directed_relax(
                centers, angles, outer_center, side, angle, 
                max_iter=200, deadline=deadline)
            c, a, oc, side, oa = optimize_outer(centers_relaxed, angles_relaxed, max_iterations=50)
            if check_validity(centers_relaxed, angles_relaxed, oc, side * 1.001, oa):
                if side < best_side:
                    best_side = side
                    best_result = (centers_relaxed, angles_relaxed, oc, side, oa)
        
        # Simulated annealing
        if time.time() < deadline - 20:
            centers_sa, angles_sa, sa_score = simulate_annealing(
                centers, angles, max_iter=100, deadline=deadline)
            c, a, oc, side, oa = optimize_outer(centers_sa, angles_sa, max_iterations=50)
            if check_validity(centers_sa, angles_sa, oc, side * 1.001, oa):
                if side < best_side:
                    best_side = side
                    best_result = (centers_sa, angles_sa, oc, side, oa)
    
    # If no valid result found, use fallback
    if best_result is None:
        centers, angles, outer_center, side, angle = get_valid_fallback()
        best_result = (centers, angles, outer_center, side, angle)
    
    centers, angles, outer_center, side, angle = best_result
    
    # Final polish: high-precision optimization of outer hexagon
    if time.time() < deadline - 5:
        outer_center, side, angle = optimize_outer(centers, angles, max_iterations=200)
    
    # Add safety margin
    side *= 1.001
    
    # Build inner list
    inner = [[float(cx), float(cy), float(ang)] for cx, cy, ang in zip(centers[:,0], centers[:,1], angles)]
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(side), float(angle)
