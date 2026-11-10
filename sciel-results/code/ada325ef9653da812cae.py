import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(np.arange(0, 360, 60))
    return np.array([cx + s * np.cos(angles), cy + s * np.sin(angles)]).T

def _get_outer_normals(a_deg):
    a = np.deg2rad(a_deg)
    # Normals to edges are at a + 30 + 60k
    angles = a + np.deg2rad(30) + np.deg2rad(np.arange(0, 360, 60))
    return np.array([np.cos(angles), np.sin(angles)]).T

def _pair_overlap(v1, v2):
    # v1, v2 are (6,2) arrays
    # Returns penalty if overlapping
    # Simple check: min distance between edges?
    # For hexagons, if any vertex of v1 is inside v2 or vice versa, or edges intersect.
    # For speed in optimization, we can use a simpler metric:
    # Distance between centers. If dist < 2*sqrt(3)/2 * something?
    # No, hexagons touch at distance sqrt(3).
    # If dist < sqrt(3), they overlap.
    # But they can overlap even if dist > sqrt(3) if rotated?
    # Actually, for regular hexagons, the minimum distance between centers for non-overlap is sqrt(3) if aligned.
    # If rotated, the minimum distance can be smaller? No, the "width" is constant.
    # The support function of a regular hexagon of side 1 is constant in the 6 normal directions?
    # No, the radius to vertex is 1, radius to edge is sqrt(3)/2.
    # So the distance for non-overlap depends on relative orientation.
    # To be safe, use polygon intersection check.
    
    # Bounding box check
    if not (np.all(v1[:,0] < v2[:,0].max()+1e-6) and np.all(v1[:,0] > v2[:,0].min()-1e-6) and
            np.all(v1[:,1] < v2[:,1].max()+1e-6) and np.all(v1[:,1] > v2[:,1].min()-1e-6)):
        return 0.0
    
    # Point in poly check
    def point_in_poly(p, poly):
        x, y = p
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # Check vertices
    for p in v1:
        if point_in_poly(p, v2):
            return 1.0
    for p in v2:
        if point_in_poly(p, v1):
            return 1.0
            
    # Edge intersection
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
                return 1.0
    return 0.0

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    
    # Initial guess: 7-hex cluster + 4
    sqrt3 = np.sqrt(3)
    centers = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [2*sqrt3, 0],
        [sqrt3, 3.0],
        [-2*sqrt3, 0],
        [-sqrt3, -3.0]
    ])
    
    angles = np.zeros(11)
    outer_center = np.array([0, 0])
    outer_angle = 0
    outer_side = 5.5
    
    def unpack(params):
        inner = params[:33].reshape(11, 3)
        outer_c = params[33:35]
        outer_a = params[35]
        outer_s = params[36]
        return inner, outer_c, outer_a, outer_s

    def objective(params):
        if time.time() > deadline - 5:
            return 1e10
        inner, outer_c, outer_a, outer_s = unpack(params)
        
        # Ensure side is positive
        if outer_s <= 0:
            return 1e10
            
        inner_verts = []
        for i in range(11):
            v = _hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
            inner_verts.append(v)
        
        outer_v = _hex_vertices(outer_c[0], outer_c[1], outer_s, outer_a)
        
        penalty = 0.0
        # Overlap penalty
        for i in range(11):
            for j in range(i+1, 11):
                if _pair_overlap(inner_verts[i], inner_verts[j]):
                    penalty += 100.0
                    
        # Containment penalty
        # Check if all inner vertices are inside outer
        # Use linear inequalities for hexagon containment for speed/robustness
        # Outer hexagon with center C, side S, angle A.
        # Normals n_k. Distance from center to edge is S*sqrt(3)/2.
        # A point P is inside if dot(P - C, n_k) <= S*sqrt(3)/2 for all k.
        normals = _get_outer_normals(outer_a)
        dist_to_edge = outer_s * sqrt3 / 2.0
        
        all_inner_verts = np.vstack(inner_verts)
        # Shift by outer center
        shifted = all_inner_verts - outer_c
        projections = shifted @ normals.T
        # Max projection must be <= dist_to_edge
        max_proj = np.max(projections, axis=0)
        violations = np.maximum(0, max_proj - dist_to_edge)
        penalty += np.sum(violations**2) * 10.0
        
        return outer_s + penalty

    p0 = np.concatenate([
        np.column_stack([centers, angles]).flatten(),
        [outer_center[0], outer_center[1], outer_angle, outer_side]
    ])
    
    best_result = None
    best_score = -1.0
    
    # Run optimization
    # Use Nelder-Mead for robustness
    try:
        res = minimize(objective, p0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < 1e10:
            inner, outer_c, outer_a, outer_s = unpack(res.x)
            # Validate
            inner_verts = [_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(11)]
            outer_v = _hex_vertices(outer_c[0], outer_c[1], outer_s, outer_a)
            
            valid = True
            for i in range(11):
                for j in range(i+1, 11):
                    if _pair_overlap(inner_verts[i], inner_verts[j]):
                        valid = False
                        break
            if valid:
                # Check containment
                normals = _get_outer_normals(outer_a)
                dist_to_edge = outer_s * sqrt3 / 2.0
                all_inner_verts = np.vstack(inner_verts)
                shifted = all_inner_verts - outer_c
                projections = shifted @ normals.T
                max_proj = np.max(projections, axis=0)
                if np.any(max_proj > dist_to_edge + 1e-6):
                    valid = False
            
            if valid:
                inner_hex_data = inner
                outer_hex_data = np.array([outer_c[0], outer_c[1], outer_a])
                outer_hex_side_length = outer_s
                
                score = 1.0 / outer_hex_side_length
                if score > best_score:
                    best_score = score
                    best_result = (inner_hex_data, outer_hex_data, outer_hex_side_length)
                    
    except Exception as e:
        pass

    if best_result:
        return best_result
    
    # Fallback
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length
