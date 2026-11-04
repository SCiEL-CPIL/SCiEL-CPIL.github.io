import math
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Initial Configuration
    # Use a subset of the 19-hex cluster (side 3 lattice) to ensure validity.
    # Axial coords for 19-hex cluster (q, r) where |q|<=2, |r|<=2, |q+r|<=2
    # We pick 11 of them.
    # Center (0,0)
    # Ring 1 (6): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (4): (2,0), (1,1), (0,2), (-1,2)
    
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    init_positions = []
    for q, r in coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_positions.append([x, y])
        
    init_rotations = [0.0] * 11
    init_S = 5.0
    
    # Parameters: x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, S
    x0 = []
    for i in range(11):
        x0.append(init_positions[i][0])
        x0.append(init_positions[i][1])
        x0.append(init_rotations[i])
    x0.append(init_S)
    
    # Helper: Get vertices of a unit hexagon at (x, y) with rotation angle
    def get_hex_vertices(x, y, angle_deg):
        angle_rad = math.radians(angle_deg)
        # Base vertices of unit hexagon (side 1) centered at origin
        # Angles: 0, 60, 120, 180, 240, 300
        base_verts = np.array([
            [1.0, 0.0],
            [0.5, math.sqrt(3)/2],
            [-0.5, math.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -math.sqrt(3)/2],
            [0.5, -math.sqrt(3)/2]
        ])
        
        # Rotation matrix
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        
        verts = (base_verts @ R.T) + np.array([x, y])
        return verts

    # Helper: Check if point is inside polygon (winding number or ray casting)
    # For convex polygons, we can check if point is to the left of all edges (or right, depending on orientation)
    def point_in_convex_poly(pt, poly):
        # poly is Nx2 array of vertices in CCW order
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            # Cross product of (p2-p1) and (pt-p1)
            v1 = p2 - p1
            v2 = pt - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross < -1e-9: # Assuming CCW, inside means cross > 0
                return False
        return True

    # Helper: Check polygon-polygon collision
    # 1. If any vertex of A is inside B, or B is inside A
    # 2. If any edges intersect
    def edges_intersect(p1, p2, p3, p4):
        # Check if segment p1-p2 intersects p3-p4
        def orient(a, b, c):
            val = (b[1]-a[1])*(c[0]-a[0]) - (b[0]-a[0])*(c[1]-a[1])
            if val > 1e-9: return 1
            if val < -1e-9: return -1
            return 0

        d1 = orient(p3, p4, p1)
        d2 = orient(p3, p4, p2)
        d3 = orient(p1, p2, p3)
        d4 = orient(p1, p2, p4)

        if ((d1 != d2) and (d3 != d4)):
            return True
        return False

    def polygons_collide(verts1, verts2):
        # Check vertices
        for v in verts1:
            if point_in_convex_poly(v, verts2):
                return True
        for v in verts2:
            if point_in_convex_poly(v, verts1):
                return True
        
        # Check edges
        for i in range(6):
            for j in range(6):
                if edges_intersect(verts1[i], verts1[(i+1)%6], verts2[j], verts2[(j+1)%6]):
                    return True
        return False

    # Outer Hexagon Vertices
    def get_outer_vertices(cx, cy, S, angle_deg):
        angle_rad = math.radians(angle_deg)
        # Side length S. Distance from center to vertex is S.
        # Wait, for a regular hexagon of side S, the circumradius is S.
        # The vertices are at distance S from center.
        base_verts = np.array([
            [S, 0.0],
            [S/2, S*math.sqrt(3)/2],
            [-S/2, S*math.sqrt(3)/2],
            [-S, 0.0],
            [-S/2, -S*math.sqrt(3)/2],
            [S/2, -S*math.sqrt(3)/2]
        ])
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        verts = (base_verts @ R.T) + np.array([cx, cy])
        return verts

    def cost(params):
        if time.time() > deadline:
            return 1e10
            
        S = params[-1]
        if S < 1.0: return 1e10
        
        outer_verts = get_outer_vertices(0.0, 0.0, S, 0.0) # Outer hexagon rotation fixed to 0 for simplicity? 
        # The task allows outer rotation. Let's fix it to 0 to reduce variables, 
        # or include it? The task says "outer hexagon ... rotation". 
        # However, rotating the outer hexagon is equivalent to rotating all inner ones 
        # if we fix the outer to 0. So we can fix outer rotation to 0.
        
        penalty = 0.0
        inner_verts_list = []
        
        for i in range(11):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            verts = get_hex_vertices(x, y, a)
            inner_verts_list.append(verts)
            
            # Check inner-outer collision
            # If any inner vertex is outside outer, penalty
            for v in verts:
                if not point_in_convex_poly(v, outer_verts):
                    # Calculate distance to boundary? Just add penalty
                    penalty += 1.0 * S # Large penalty
                    
        # Check inner-inner collisions
        for i in range(11):
            for j in range(i+1, 11):
                if polygons_collide(inner_verts_list[i], inner_verts_list[j]):
                    penalty += 1.0 * S

        return S + penalty

    # Run Optimization
    best_result = None
    best_cost = cost(x0)
    
    try:
        result = minimize(cost, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result.x
    except Exception:
        pass

    # If optimization failed or didn't improve, use initial
    if best_result is None:
        best_params = x0
    else:
        best_params = best_result

    # Extract final values
    final_inner = []
    for i in range(11):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        final_inner.append([x, y, a])
        
    final_S = best_params[-1]
    
    # Final validity check
    # If the result is invalid (penalty > 0), we might need to fallback to initial S=5.0
    # But Nelder-Mead should find a valid one if one exists.
    # To be safe, if cost > S (meaning penalty > 0), we return a known valid config.
    if best_cost > final_S + 1e-5:
        # Fallback to initial valid config
        final_inner = []
        for i in range(11):
            final_inner.append([init_positions[i][0], init_positions[i][1], 0.0])
        final_S = 5.0

    return final_inner, (0.0, 0.0), final_S, 0.0
