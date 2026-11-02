import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12
SQRT3 = np.sqrt(3.0)

def hex_vertices(center, side, angle):
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + angle
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def get_hex_axes(angle):
    # Returns the 6 normal vectors for a regular hexagon with given rotation
    # Edges are at angles angle + 0, 60, 120, ...
    # Normals are perpendicular to edges
    # Edge direction: cos(a), sin(a)
    # Normal direction: -sin(a), cos(a)
    base_angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    normals = np.column_stack((-np.sin(base_angles + angle), np.cos(base_angles + angle)))
    return normals

def sat_distance(poly1, poly2):
    # Returns minimum distance if separated, or negative overlap depth if intersecting
    # Using SAT with vectorized projections
    axes1 = get_hex_axes(np.radians(poly1[2])) # Assuming poly is (x, y, angle_deg)
    axes2 = get_hex_axes(np.radians(poly2[2]))
    axes = np.vstack([axes1, axes2])
    
    # Unique axes (optional, but vstack is fast enough for 12x12 pairs)
    # Project poly1 and poly2 onto all axes
    # poly1 center: c1, poly2 center: c2
    # Vertices of poly1: c1 + R1 * V
    # Projection of vertex v onto axis a: dot(v, a)
    # Min/Max projection of a polygon onto an axis can be computed from center and vertices
    
    # Actually, let's just use the vertices.
    v1 = hex_vertices((poly1[0], poly1[1]), 1.0, np.radians(poly1[2]))
    v2 = hex_vertices((poly2[0], poly2[1]), 1.0, np.radians(poly2[2]))
    
    # Project v1 and v2 onto all axes
    # v1: (6, 2), axes: (12, 2)
    # proj1: (6, 12) = v1 @ axes.T
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    min1 = np.min(proj1, axis=0)
    max1 = np.max(proj1, axis=0)
    min2 = np.min(proj2, axis=0)
    max2 = np.max(proj2, axis=0)
    
    # Separation for each axis
    # sep = min(max1, max2) - max(min1, min2)
    # If sep < 0 for any axis, they are separated by -sep (distance)
    # If sep >= 0 for all axes, they intersect.
    # We want the minimum distance.
    # If they intersect, distance is 0 (or negative penalty based on overlap).
    # For penalty, we want to minimize overlap.
    
    overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
    
    # If any overlap is < 0, they are separated. The distance is -min(overlaps).
    # If all overlaps >= 0, they intersect.
    
    if np.any(overlaps < 0):
        return -np.min(overlaps) # Positive distance
    else:
        # They intersect. We need a penalty proportional to overlap.
        # Simple penalty: sum of overlaps? Or max overlap?
        # Using sum of positive overlaps as a measure of intersection depth
        return -np.sum(overlaps) # Negative value indicating overlap

def penalty_function(params):
    outer_side = params[0]
    outer_cx, outer_cy = params[1:3]
    outer_angle = params[3]
    
    if outer_side <= 0:
        return 1e10
        
    inner_params = params[4:]
    
    total_penalty = outer_side
    
    # Check outer containment
    outer_verts = hex_vertices((outer_cx, outer_cy), outer_side, np.radians(outer_angle))
    
    # For each inner hex, check if all vertices are inside outer
    # This is O(N * 6 * 6) = O(432) point-in-poly checks.
    # Can be vectorized?
    # Point in polygon is tricky to vectorize directly without loops or complex math.
    # Given N=12, a loop is fine if the check is fast.
    
    for i in range(N):
        x, y, a = inner_params[3*i:3*i+3]
        v = hex_vertices((x, y), 1.0, np.radians(a))
        for pt in v:
            # Fast point in polygon check for convex polygon
            # Cross product of edge vector and point vector must have same sign
            inside = True
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j+1)%6]
                cross = (p2[0]-p1[0])*(pt[1]-p1[1]) - (p2[1]-p1[1])*(pt[0]-p1[0])
                if cross < -1e-9:
                    inside = False
                    break
            if not inside:
                # Calculate distance to boundary for penalty
                # Approximate penalty: distance to closest edge
                min_dist = np.inf
                for j in range(6):
                    p1 = outer_verts[j]
                    p2 = outer_verts[(j+1)%6]
                    # Distance from pt to segment p1-p2
                    ab = p2 - p1
                    ap = pt - p1
                    ab_len_sq = np.dot(ab, ab)
                    if ab_len_sq < 1e-9: continue
                    t = np.dot(ap, ab) / ab_len_sq
                    t = np.clip(t, 0, 1)
                    proj = p1 + t * ab
                    d = np.linalg.norm(pt - proj)
                    if d < min_dist: min_dist = d
                total_penalty += min_dist * min_dist * 100.0

    # Check inner-inner distances
    for i in range(N):
        for j in range(i+1, N):
            p1 = inner_params[3*i:3*i+3]
            p2 = inner_params[3*j:3*j+3]
            d = sat_distance(p1, p2)
            if d < 0:
                # Overlap
                total_penalty += (-d) * 50.0
            elif d < 0.1:
                # Too close
                total_penalty += (0.1 - d) * 10.0

    return total_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = np.inf
    best_params = None
    
    def init_honeycomb():
        # Standard honeycomb packing
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        # 12 hexagons in a hexagonal arrangement?
        # Center 1, ring 1 (6), ring 2 (5) -> 12 total?
        # Or 3x4 grid?
        # Let's try a hexagonal cluster
        idx = 0
        params[4 + 3*idx] = 0
        params[4 + 3*idx+1] = 0
        params[4 + 3*idx+2] = 0
        idx += 1
        
        for k in range(6):
            ang = np.radians(k * 60)
            dist = 2.0 * SQRT3 # Distance between centers of touching hexagons
            params[4 + 3*idx] = dist * np.cos(ang)
            params[4 + 3*idx+1] = dist * np.sin(ang)
            params[4 + 3*idx+2] = 0.0
            idx += 1
            
        # 5 more for the second ring?
        # This is getting complicated to place manually.
        # Let's just use a grid for now and let the optimizer fix it.
        return params

    def init_grid():
        params = np.zeros(4 + 3*N)
        params[0] = 10.0
        params[1] = 0.0
        params[2] = 0.0
        params[3] = 0.0
        
        for i in range(N):
            col = i % 4
            row = i // 4
            ix = col * 3.0 * SQRT3
            iy = row * 3.0
            params[4 + 3*i] = ix
            params[4 + 3*i+1] = iy
            params[4 + 3*i+2] = 0.0
        return params

    initializations = [init_grid()]
    
    for init_params in initializations:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(penalty_function, init_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        except Exception:
            pass
            
    if best_params is None:
        # Fallback
        inner = []
        for i in range(N):
            col = i % 4
            row = i // 4
            x = col * 3.5 - 5.0
            y = row * 3.0 - 1.5
            inner.append((x, y, 0))
        return inner, (0, 0), 10.0, 0.0

    outer_side = best_params[0]
    outer_cx = best_params[1]
    outer_cy = best_params[2]
    outer_angle = best_params[3]
    
    inner = []
    for i in range(N):
        x = best_params[4 + 3*i]
        y = best_params[4 + 3*i+1]
        a = best_params[4 + 3*i+2]
        inner.append((x, y, a))
        
    outer_center = (outer_cx, outer_cy)
    
    return inner, outer_center, outer_side, outer_angle
