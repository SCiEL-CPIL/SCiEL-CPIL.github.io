import numpy as np
import time
from itertools import product

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
    # Rotate point into hexagon's local frame
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Regular hexagon with flat top: |x| <= s, |y| <= s*sqrt(3)/2, and |x|/s + |y|/(s*sqrt(3)/2) <= 1 for the hex shape
    # Actually for a regular hexagon with vertices at angles 0,60,...: the boundary is max of 6 half-planes
    # Half-plane i: cos(theta_i)*lx + sin(theta_i)*ly <= s * cos(pi/6) = s*sqrt(3)/2
    for i in range(6):
        theta = i * np.pi / 3
        if lx * np.cos(theta) + ly * np.sin(theta) > s * np.sqrt(3) / 2 + 1e-12:
            return False
    return True

def hexagons_disjoint(v1, v2, s, eps=1e-9):
    """Check if two hexagons (vertex arrays) are disjoint (no interior intersection)."""
    # Check if any vertex of one is inside the other
    for v in v1:
        if point_in_hex(v[0], v[1], 0, 0, s, 0) and False:
            pass
    # Better: check edge intersections and point-in-polygon
    # For simplicity, check if any vertex of v1 is strictly inside v2 and vice versa
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
    # Check edge intersections
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Place 11 hexagons in a compact hexagonal cluster
    # Hex grid: center + 6 neighbors + 4 more
    # In hex grid coordinates, neighbors of (0,0) are at:
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) in axial coords
    # Convert axial to pixel: x = s*sqrt(3)*(q + r/2), y = s*3/2*r  (for pointy-top)
    # For flat-top: x = s*3/2*q, y = s*sqrt(3)*(r + q/2)
    
    # Let's use flat-top hexagons. Side length = 1.
    # Distance between centers of edge-adjacent flat-top hexagons = sqrt(3)
    # Axial to pixel (flat-top): 
    #   x = 1.5 * q
    #   y = sqrt(3) * (r + 0.5*q)
    
    sqrt3 = np.sqrt(3)
    
    # 11 hexagons in hex pattern:
    # Layer 0: (0,0)
    # Layer 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Layer 2: pick 4 from the 12 positions at distance 2
    # Let's use a symmetric selection
    
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    # That's 11 positions
    
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # All inner hexagons have same orientation (0 degrees) and side 1
    inner_angles = [0.0] * N
    
    # Compute bounding box of all hexagon vertices to find good outer hex center
    all_verts = []
    for cx, cy in centers:
        verts = hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    cx_mean = all_verts[:, 0].mean()
    cy_mean = all_verts[:, 1].mean()
    
    # Try different outer hexagon rotations to minimize side length
    best_side = float('inf')
    best_center = (cx_mean, cy_mean)
    best_angle = 0.0
    
    # For a given outer hexagon rotation, the minimal side length to contain a set of points
    # is determined by the maximum "distance" in the 6 normal directions
    # For a regular hexagon with center (cx,cy), side s, rotation a:
    # The 6 face normals are at angles a + i*pi/3
    # A point (px,py) is inside if for all i: 
    #   (px-cx)*cos(n_i) + (py-cy)*sin(n_i) <= s * sqrt(3)/2
    # So s >= max over all points and all i of [(px-cx)*cos(n_i) + (py-cy)*sin(n_i)] / (sqrt(3)/2)
    
    for outer_angle in np.linspace(0, 60, 121):  # 0 to 60 degrees
        a = np.radians(outer_angle)
        normals = []
        for i in range(6):
            theta = a + np.pi/6 + i * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        dx = all_verts[:, 0] - cx_mean
        dy = all_verts[:, 1] - cy_mean
        
        projections = np.zeros((N * 6, 6))
        for i in range(6):
            projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
        
        max_proj = projections.max(axis=0)
        min_proj = projections.min(axis=0)
        required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
        side = required_s.max()
        
        if side < best_side:
            best_side = side
            best_center = (cx_mean, cy_mean)
            best_angle = outer_angle
    
    # Now verify that inner hexagons are actually disjoint
    # Check all pairs
    valid = True
    inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: increase spacing slightly
        # Scale positions by a small factor
        scale = 1.001
        centers_scaled = centers * scale
        inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers_scaled]
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            all_verts = np.vstack(inner_verts)
            cx_mean = all_verts[:, 0].mean()
            cy_mean = all_verts[:, 1].mean()
            
            for outer_angle in np.linspace(0, 60, 121):
                a = np.radians(outer_angle)
                normals = []
                for i in range(6):
                    theta = a + np.pi/6 + i * np.pi / 3
                    normals.append([np.cos(theta), np.sin(theta)])
                normals = np.array(normals)
                
                dx = all_verts[:, 0] - cx_mean
                dy = all_verts[:, 1] - cy_mean
                
                projections = np.zeros((N * 6, 6))
                for i in range(6):
                    projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
                
                max_proj = projections.max(axis=0)
                min_proj = projections.min(axis=0)
                required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
                side = required_s.max()
                
                if side < best_side:
                    best_side = side
                    best_center = (cx_mean, cy_mean)
                    best_angle = outer_angle
            centers = centers_scaled
    
    # Build inner list
    inner = [[float(cx), float(cy), float(0.0)] for cx, cy in centers]
    
    # Final verification: ensure all inner hex vertices are inside outer hex
    # Add small safety margin
    best_side *= 1.001
    inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
    all_check_verts = np.vstack(inner_verts)
    
    a = np.radians(best_angle)
    normals = []
    for i in range(6):
        theta = a + np.pi/6 + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    dx = all_check_verts[:, 0] - best_center[0]
    dy = all_check_verts[:, 1] - best_center[1]
    
    max_violation = 0
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        max_violation = max(max_violation, proj.max() - best_side * sqrt3 / 2)
        max_violation = max(max_violation, -proj.min() - best_side * sqrt3 / 2)
    
    if max_violation > 1e-9:
        best_side += max_violation / (sqrt3 / 2) * 1.001
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)
