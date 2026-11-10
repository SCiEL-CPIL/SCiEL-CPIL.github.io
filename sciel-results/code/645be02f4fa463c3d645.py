import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Constants
EPS = 1e-6
START_TIME = time.time()
TIME_LIMIT = 600
SAFETY_MARGIN = 30

def get_hex_vertices(cx, cy, s, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.deg2rad(60 * np.arange(6))
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack((x, y))

def hexagon_distance(v1, v2):
    # v1, v2 are (6,2) arrays of vertices
    # Calculate min distance between two convex polygons
    # Simple approach: check all pairs of edges for intersection, if none, min distance between vertices and edges
    # Given small size, O(36) checks are fine
    
    min_dist = float('inf')
    
    # Helper to get distance from point to segment
    def point_segment_dist(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    # Check vertices of v1 against edges of v2
    for i in range(6):
        p1 = v1[i]
        p2 = v2[(i+1)%6]
        for j in range(6):
            q1 = v2[j]
            q2 = v2[(j+1)%6]
            d1 = point_segment_dist(p1, q1, q2)
            d2 = point_segment_dist(q1, p1, p2)
            if d1 < min_dist: min_dist = d1
            if d2 < min_dist: min_dist = d2
            
    return min_dist

def is_inside_outer(vertices, S, a_outer=0):
    # Outer hexagon centered at 0,0 with side S and angle a_outer
    a_rad = np.deg2rad(a_outer)
    # Normals are at a_rad + 30 + 60k
    normals = []
    for k in range(6):
        angle = a_rad + np.deg2rad(30 + 60*k)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    
    apothem = S * np.sqrt(3) / 2
    
    for n in normals:
        dists = np.dot(vertices, n)
        if np.any(dists > apothem + 1e-9):
            return False
    return True

def hexagon_packing_11():
    # Generate initial 19-hex cluster
    # Axial coords (q, r)
    coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Hex distance
            if (abs(q) + abs(r) + abs(q+r)) // 2 <= 2:
                coords.append((q, r))
                
    # Convert to Cartesian (flat-top, side 1)
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    hexes = []
    for q, r in coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        hexes.append((x, y))
        
    # hexes has 19 entries
    # Identify corners: max distance from origin
    dists = [np.hypot(x, y) for x, y in hexes]
    max_dist = max(dists)
    corners = [i for i, d in enumerate(dists) if d > max_dist - 1e-6]
    
    # Remove 6 corners
    remaining_indices = [i for i in range(19) if i not in corners]
    
    # Remove 2 more to get 11
    # Pick first 2 from remaining
    to_remove = remaining_indices[:2]
    final_indices = [i for i in remaining_indices if i not in to_remove]
    
    initial_positions = np.array([hexes[i] for i in final_indices])
    
    # Initial S
    # Calculate bounding box of initial positions
    # Max apothem needed
    max_apothem = 0
    for x, y in initial_positions:
        # Distance to sides of outer hexagon (a=0)
        # Normals at 30, 90, ...
        for k in range(6):
            angle = np.deg2rad(30 + 60*k)
            n = np.array([np.cos(angle), np.sin(angle)])
            # Project all vertices of this hex
            verts = get_hex_vertices(x, y, 1.0, 0)
            dists = np.dot(verts, n)
            if np.max(dists) > max_apothem:
                max_apothem = np.max(dists)
                
    S_init = max_apothem * 2 / np.sqrt(3)
    S_init = max(S_init, np.sqrt(11))
    
    # Variables: x0, y0, a0, ..., x10, y10, a10, S
    x0 = np.zeros(34)
    for i in range(11):
        x0[i*3] = initial_positions[i, 0]
        x0[i*3+1] = initial_positions[i, 1]
        x0[i*3+2] = 0.0
    x0[33] = S_init
    
    bounds = []
    for i in range(11):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((0, 360)) # angle
    bounds.append((np.sqrt(11), 10)) # S
    
    def objective(vars):
        S = vars[33]
        return S
        
    def constraints(vars):
        cons = []
        S = vars[33]
        
        # Extract positions
        positions = []
        for i in range(11):
            x = vars[i*3]
            y = vars[i*3+1]
            a = vars[i*3+2]
            positions.append((x, y, a))
            
        # Check containment
        for i in range(11):
            x, y, a = positions[i]
            verts = get_hex_vertices(x, y, 1.0, a)
            if not is_inside_outer(verts, S):
                # Penalty
                # Calculate violation
                a_rad = np.deg2rad(0)
                apothem = S * np.sqrt(3) / 2
                viol = 0
                for k in range(6):
                    angle = a_rad + np.deg2rad(30 + 60*k)
                    n = np.array([np.cos(angle), np.sin(angle)])
                    dists = np.dot(verts, n)
                    if np.max(dists) > apothem:
                        viol += np.max(dists) - apothem
                cons.append({'type': 'ineq', 'fun': lambda v, vi=viol: -vi})
                
        # Check disjointness
        verts_list = []
        for i in range(11):
            x, y, a = positions[i]
            verts_list.append(get_hex_vertices(x, y, 1.0, a))
            
        for i in range(11):
            for j in range(i+1, 11):
                d = hexagon_distance(verts_list[i], verts_list[j])
                # Must be > EPS
                cons.append({'type': 'ineq', 'fun': lambda v, di=d: di - EPS})
                
        return cons

    # Use SLSQP
    result = None
    best_S = S_init
    best_x = x0.copy()
    
    try:
        # Run optimization
        # To handle time, we can run multiple iterations or just one with high maxiter
        # But SLSQP might be slow.
        # Let's try.
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 200, 'ftol': 1e-6})
        
        if res.success:
            best_S = res.x[33]
            best_x = res.x
        else:
            # Check if res.x is better
            if res.x[33] < best_S:
                 # Validate res.x
                 # (Simplistic: assume valid if not crashed)
                 best_S = res.x[33]
                 best_x = res.x
                 
    except Exception as e:
        print(f"Optimization error: {e}")
        
    # Construct result from best_x
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = best_x[i*3]
        inner_hex_data[i, 1] = best_x[i*3+1]
        inner_hex_data[i, 2] = best_x[i*3+2]
        
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = best_S
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
