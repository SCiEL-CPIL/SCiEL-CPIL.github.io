import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon with given center, rotation, and side length."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def polygon_area_overlap(v1, v2):
    """Check if two convex polygons overlap using separating axis theorem.
    Returns the minimum penetration depth (positive if overlapping, 0 if not)."""
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    min_dist = np.inf
    for edges in [edges1, edges2]:
        for i in range(len(edges)):
            edge = edges[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-15)
            
            # Project both polygons onto this normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0  # Separated
            
            # Overlap on this axis
            overlap = min(max1, max2) - max(min1, min2)
            min_dist = min(min_dist, overlap)
    
    return min_dist

def check_valid(inner, center, side, angle_deg, tol=1e-6):
    """Check if configuration is valid (no overlaps, all inside outer hexagon)."""
    # Check inner hexagons don't overlap
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            vi = hex_vertices(xi, yi, ai)
            vj = hex_vertices(xj, yj, aj)
            if polygon_area_overlap(vi, vj) > tol:
                return False
    
    # Check all inner hexagons are inside outer hexagon
    outer_verts = hex_vertices(center[0], center[1], angle_deg, side)
    for i in range(N):
        xi, yi, ai = inner[i]
        vi = hex_vertices(xi, yi, ai)
        # Check if all vertices of inner hex are inside outer hex
        for v in vi:
            if not point_in_polygon(v, outer_verts):
                return False
    
    return True

def point_in_polygon(p, poly):
    """Check if point p is inside convex polygon poly using cross product method."""
    n = len(poly)
    sign = 0
    for i in range(n):
        v1 = poly[i]
        v2 = poly[(i+1) % n]
        cross = (v2[0]-v1[0])*(p[1]-v1[1]) - (v2[1]-v1[1])*(p[0]-v1[0])
        if cross > 1e-10:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-10:
            if sign > 0:
                return False
            sign = -1
    return True

def objective(params):
    """Objective function: minimize outer side length with penalty for violations."""
    # Unpack parameters
    # For each inner hex: x, y, angle (3 params) -> 33 params
    # Outer: cx, cy, side, angle (4 params)
    inner_params = params[:N*3].reshape(N, 3)
    outer_params = params[N*3:]
    outer_cx, outer_cy, outer_side, outer_angle = outer_params
    
    if outer_side < 1.0:
        return 1e6
    
    penalty = 0.0
    
    # Penalty for inner hexagon overlaps
    for i in range(N):
        for j in range(i+1, N):
            vi = hex_vertices(inner_params[i,0], inner_params[i,1], inner_params[i,2])
            vj = hex_vertices(inner_params[j,0], inner_params[j,1], inner_params[j,2])
            overlap = polygon_area_overlap(vi, vj)
            if overlap > 0:
                penalty += overlap * overlap * 100.0
    
    # Penalty for inner hexagons outside outer hexagon
    outer_verts = hex_vertices(outer_cx, outer_cy, outer_angle, outer_side)
    for i in range(N):
        vi = hex_vertices(inner_params[i,0], inner_params[i,1], inner_params[i,2])
        for v in vi:
            if not point_in_polygon(v, outer_verts):
                # Compute distance outside
                dist = 0
                for k in range(6):
                    v1 = outer_verts[k]
                    v2 = outer_verts[(k+1)%6]
                    edge = v2 - v1
                    edge_len = np.linalg.norm(edge)
                    if edge_len < 1e-15:
                        continue
                    normal = np.array([-edge[1], edge[0]]) / edge_len
                    # Check which side
                    d = (v - v1) @ normal
                    if d < 0:
                        dist = max(dist, -d)
                penalty += dist * dist * 100.0
    
    return outer_side + penalty

def initial_config():
    """Create initial compact configuration."""
    # Central hexagon at origin
    # 6 hexagons around it at distance 2 (touching)
    # 4 more hexagons in gaps
    
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # 6 surrounding at angle 0, 60, 120, 180, 240, 300
    for i in range(6):
        angle = i * 60.0
        x = 2.0 * np.cos(np.radians(angle))
        y = 2.0 * np.sin(np.radians(angle))
        inner.append([x, y, angle])
    
    # 4 more in the gaps (at 30, 150, 210, 330 degrees, further out)
    for i in range(4):
        angle = 30.0 + i * 120.0
        x = 3.5 * np.cos(np.radians(angle))
        y = 3.5 * np.sin(np.radians(angle))
        inner.append([x, y, angle])
    
    # Outer hexagon centered at origin, large enough to contain all
    outer_center = [0.0, 0.0]
    outer_side = 6.0
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # Safety margin
    
    best_result = None
    best_score = np.inf
    
    # Try multiple initial configurations
    configs = []
    
    # Config 1: Standard compact
    inner1, oc1, os1, oa1 = initial_config()
    configs.append((inner1, oc1, os1, oa1))
    
    # Config 2: Slightly different arrangement
    inner2 = [[0.0, 0.0, 0.0]]
    for i in range(6):
        angle = i * 60.0
        x = 2.0 * np.cos(np.radians(angle))
        y = 2.0 * np.sin(np.radians(angle))
        inner2.append([x, y, angle])
    # Add 4 more in different positions
    for i in range(4):
        angle = 60.0 + i * 90.0
        x = 3.4 * np.cos(np.radians(angle))
        y = 3.4 * np.sin(np.radians(angle))
        inner2.append([x, y, angle])
    configs.append((inner2, [0.0, 0.0], 6.0, 0.0))
    
    # Config 3: Rotated version
    inner3 = [[0.0, 0.0, 30.0]]
    for i in range(6):
        angle = 30.0 + i * 60.0
        x = 2.0 * np.cos(np.radians(angle))
        y = 2.0 * np.sin(np.radians(angle))
        inner3.append([x, y, angle])
    for i in range(4):
        angle = 90.0 + i * 120.0
        x = 3.5 * np.cos(np.radians(angle))
        y = 3.5 * np.sin(np.radians(angle))
        inner3.append([x, y, angle])
    configs.append((inner3, [0.0, 0.0], 6.0, 30.0))
    
    for cfg_idx, (inner, oc, os, oa) in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Build parameter vector
        params = np.zeros(N*3 + 4)
        for i in range(N):
            params[i*3] = inner[i][0]
            params[i*3+1] = inner[i][1]
            params[i*3+2] = inner[i][2]
        params[N*3] = oc[0]
        params[N*3+1] = oc[1]
        params[N*3+2] = os
        params[N*3+3] = oa
        
        # Optimize
        try:
            result = minimize(objective, params, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < best_score:
                best_params = result.x
                best_score = result.fun
                
                # Extract results
                inner_params = best_params[:N*3].reshape(N, 3)
                outer_params = best_params[N*3:]
                outer_cx, outer_cy, outer_side, outer_angle = outer_params
                
                inner_list = [[float(inner_params[i,0]), float(inner_params[i,1]), float(inner_params[i,2])] for i in range(N)]
                
                # Check validity
                if check_valid(inner_list, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle)):
                    if best_result is None or float(outer_side) < best_result[2]:
                        best_result = (inner_list, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle))
        except Exception as e:
            pass
    
    # If no valid result found, return a safe default
    if best_result is None:
        # Return the initial config scaled to be valid
        inner, oc, os, oa = initial_config()
        # Make sure it's valid by increasing side if needed
        for _ in range(100):
            if check_valid(inner, oc, os, oa):
                break
            os *= 1.05
        return inner, oc, os, oa
    
    return best_result
