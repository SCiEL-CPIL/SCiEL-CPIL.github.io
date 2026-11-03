import numpy as np
import time
from scipy.optimize import minimize
import math

N = 11
START_TIME = time.time()
DEADLINE = 120.0

def hex_vertices(cx, cy, angle, size=1.0):
    """
    Returns the 6 vertices of a regular hexagon of given size,
    centered at (cx, cy) and rotated by angle (radians).
    """
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack((cx + size * np.cos(angles), cy + size * np.sin(angles)))

def point_in_hexagon(p, verts):
    """
    Check if point p is inside convex polygon verts.
    Returns True if inside or on boundary.
    """
    # Cross product method for convex polygon
    # Check if p is on the same side of all edges
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        # Vector p1->p2
        v1 = p2 - p1
        # Vector p1->p
        v2 = p - p1
        # Cross product z-component
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        # For a counter-clockwise polygon, cross should be >= 0 for points inside
        # We assume verts are generated CCW
        if cross < -1e-9:
            return False
    return True

def segments_intersect(p1, p2, p3, p4):
    """
    Check if segment p1-p2 intersects segment p3-p4.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    def on_segment(a, b, c):
        # Check if point b is on segment a-c
        return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and \
               min(a[1], c[1]) <= b[1] <= max(a[1], c[1])

    if not (ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)):
        return False
        
    # Check collinear cases
    if ccw(p1, p3, p4) == 0 and on_segment(p1, p3, p4): return True
    if ccw(p1, p2, p3) == 0 and on_segment(p1, p3, p2): return True
    if ccw(p1, p2, p4) == 0 and on_segment(p1, p4, p2): return True
    if ccw(p3, p1, p4) == 0 and on_segment(p3, p1, p4): return True
    
    return False

def polygons_overlap(verts1, verts2):
    """
    Check if two convex polygons overlap.
    """
    n1 = len(verts1)
    n2 = len(verts2)
    
    # Check if any vertex of 1 is in 2
    for v in verts1:
        if point_in_hexagon(v, verts2):
            return True
    # Check if any vertex of 2 is in 1
    for v in verts2:
        if point_in_hexagon(v, verts1):
            return True
            
    # Check edge intersections
    for i in range(n1):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % n1]
        for j in range(n2):
            p3 = verts2[j]
            p4 = verts2[(j + 1) % n2]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def min_enclosing_hex_side(vertices, center=None):
    """
    Compute the minimum side length of a regular hexagon enclosing the given vertices.
    vertices: (N, 2) array
    center: (2,) array or None (uses centroid)
    """
    if center is None:
        center = np.mean(vertices, axis=0)
    
    min_side = float('inf')
    # Scan rotation from 0 to 60 degrees
    # Step 1 degree for speed, can be refined later
    for phi_deg in range(0, 61, 1):
        phi = np.deg2rad(phi_deg)
        # Normals to the sides of the enclosing hexagon
        # The sides are perpendicular to these normals
        # Normals are at angles phi, phi + 60, phi + 120
        normals = []
        for k in range(3):
            angle = phi + k * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_radius = 0
        for n_vec in normals:
            # Project vertices onto normal
            projections = (vertices - center) @ n_vec
            min_proj = np.min(projections)
            max_proj = np.max(projections)
            # The distance from center to the furthest point in this direction
            # The hexagon must contain the interval [min_proj, max_proj]
            # The apothem (distance from center to side) must be at least max(|min_proj|, |max_proj|)
            # Wait, the center of the enclosing hexagon is 'center'.
            # The side is at distance 'apothem' from center.
            # So apothem >= max(|min_proj|, |max_proj|)
            radius = max(abs(min_proj), abs(max_proj))
            max_radius = max(max_radius, radius)
        
        # Side length S = 2 * apothem / sqrt(3)
        side = 2 * max_radius / np.sqrt(3)
        if side < min_side:
            min_side = side
            
    return min_side

def compute_objective(params):
    """
    params: flattened array of [x1, y1, theta1, x2, y2, theta2, ...]
    """
    if time.time() - START_TIME > DEADLINE:
        return float('inf')
        
    x = params[0::3]
    y = params[1::3]
    theta = params[2::3]
    
    # Compute vertices for all inner hexagons
    all_verts = []
    hex_verts_list = []
    for i in range(N):
        verts = hex_vertices(x[i], y[i], theta[i])
        hex_verts_list.append(verts)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    
    # 1. Min enclosing hexagon side
    side = min_enclosing_hex_side(all_verts)
    
    # 2. Overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            # Quick check: center distance
            dist = np.linalg.norm([x[i] - x[j], y[i] - y[j]])
            if dist < np.sqrt(3) - 0.1:
                # Likely overlap, do detailed check
                if polygons_overlap(hex_verts_list[i], hex_verts_list[j]):
                    # Penalty proportional to overlap depth (approx by dist violation)
                    # We want to push them apart
                    # Use a quadratic penalty on the distance violation
                    violation = np.sqrt(3) - dist
                    if violation > 0:
                        penalty += violation ** 2 * 100
            elif dist < np.sqrt(3) + 0.5:
                # Borderline, check carefully
                if polygons_overlap(hex_verts_list[i], hex_verts_list[j]):
                    # Even if dist > sqrt(3), they might overlap if rotated badly?
                    # No, min center dist is sqrt(3).
                    # But if dist is slightly larger, they shouldn't overlap.
                    # This check is for safety.
                    violation = 0.1 
                    penalty += violation ** 2 * 100

    return side + penalty

def construct():
    if time.time() - START_TIME > DEADLINE:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Initial configuration: Hexagonal cluster
    # 1 center, 6 neighbors, 4 more
    # Using axial coordinates
    coords = []
    # Center
    coords.append((0, 0))
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        coords.append((np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)))
    # Ring 2 (partial)
    # Add 4 more to make 11
    # Let's just add 4 neighbors to some of the ring 1 hexagons
    # Or use a 3x3 grid rotated?
    # Simpler: Just take the first 11 points from a hexagonal lattice
    lattice_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Convert axial to cartesian
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            lattice_points.append((x, y))
    
    # Sort by distance from origin to get the most compact cluster
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take first 11
    init_coords = lattice_points[:N]
    
    # Initialize params
    params0 = []
    for x, y in init_coords:
        params0.extend([x, y, 0.0])
    params0 = np.array(params0)
    
    # Optimize
    best_params = params0
    best_score = compute_objective(params0)
    
    try:
        res = minimize(compute_objective, params0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
        
    # Extract best result
    x = best_params[0::3]
    y = best_params[1::3]
    theta = best_params[2::3]
    
    inner = []
    for i in range(N):
        inner.append([x[i], y[i], np.rad2deg(theta[i])])
    
    # Calculate final outer hexagon parameters
    all_verts = []
    for i in range(N):
        verts = hex_vertices(x[i], y[i], theta[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Find optimal center and rotation for the enclosing hexagon
    # We used centroid in the objective, but we can refine it
    # For now, use centroid and the best phi from the scan
    center = np.mean(all_verts, axis=0)
    
    # Refine phi
    best_phi = 0
    best_side = float('inf')
    for phi_deg in range(0, 61, 1):
        phi = np.deg2rad(phi_deg)
        normals = []
        for k in range(3):
            angle = phi + k * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_radius = 0
        for n_vec in normals:
            projections = (all_verts - center) @ n_vec
            min_proj = np.min(projections)
            max_proj = np.max(projections)
            radius = max(abs(min_proj), abs(max_proj))
            max_radius = max(max_radius, radius)
            
        side = 2 * max_radius / np.sqrt(3)
        if side < best_side:
            best_side = side
            best_phi = phi_deg
            
    return inner, (center[0], center[1]), best_side, best_phi

