import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    # Precompute normals and vertices for outer hexagon (angle 0)
    # Vertices at 0, 60, 120, ... deg
    # Inward normals: perpendicular to edges, pointing inward
    # Edge between vertex 0 (at 0 deg) and vertex 1 (at 60 deg)
    # Midpoint of edge is at 30 deg. Normal points toward center, so at 210 deg?
    # Let's use the property: distance from center to any edge is r = S * sqrt(3)/2.
    # The normal direction for edge k is at angle (k * 60 + 30) degrees?
    # No, the vertices are at k*60. The edge midpoint is at k*60 + 30.
    # The normal vector pointing INWARD is from edge midpoint to center, i.e., opposite to edge midpoint direction.
    # So inward normal angle = k*60 + 30 + 180.
    # Or simply: the condition for a point P to be inside is:
    # dot(P, n_k) <= r for all k, where n_k is the outward normal.
    # Outward normal angle for edge k (between vertex k and k+1) is k*60 + 30.
    outward_normals = np.cos(np.deg2rad(30 + np.arange(6)*60))
    outward_normals_y = np.sin(np.deg2rad(30 + np.arange(6)*60))
    normals_x = outward_normals
    normals_y = outward_normals_y

    def hex_verts(cx, cy, s, rot_deg):
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def point_in_hexagon(p, cx, cy, s, rot_deg):
        """Check if point p is inside hexagon (cx, cy, s, rot)."""
        # Transform p to hexagon local coords
        p_local = np.array([p[0] - cx, p[1] - cy])
        # Rotate back
        rot_rad = -np.deg2rad(rot_deg)
        c = math.cos(rot_rad)
        s_r = math.sin(rot_rad)
        px = p_local[0]*c - p_local[1]*s_r
        py = p_local[0]*s_r + p_local[1]*c
        
        # Check against 6 half-planes
        # Inward normals for unit hexagon at angle 0 (pointy right)
        # Outward normals are at 30, 90, 150, 210, 270, 330.
        # Inward: 210, 270, 330, 30, 90, 150.
        # Distance from center to edge is r = s * sqrt(3)/2.
        r = s * SQRT3 / 2.0
        
        for k in range(6):
            angle = np.deg2rad(30 + k*60) # Outward normal angle
            nx = math.cos(angle)
            ny = math.sin(angle)
            # dot(p, n) <= r
            if px * nx + py * ny > r + 1e-9:
                return False
        return True

    def min_dist_polygons(v1, v2):
        """
        Returns minimum distance between two convex polygons.
        If they intersect, returns 0.
        """
        # Check intersection first (vertex in polygon)
        for p in v1:
            if point_in_hexagon(p, 0,0,0,0): # dummy, we need local logic
                pass
        # Actually, let's just use the distance between edges if no intersection.
        # For convex polygons, min distance is between an edge of one and vertex of other, or edge-edge.
        # If they intersect, we return 0.
        
        # Quick check: if any vertex of v1 is in v2 or vice versa.
        # We need a general point_in_polygon for the specific hexagons.
        # Let's implement a specific check.
        
        def v_in_hex(v, verts):
            # v is point, verts are hex vertices
            # Use cross product method
            sign = 0
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i+1)%6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross > 1e-9:
                    if sign == 0: sign = 1
                    elif sign == -1: return False
                elif cross < -1e-9:
                    if sign == 0: sign = -1
                    elif sign == 1: return False
            return True

        if v_in_hex(v1[0], v2) or v_in_hex(v2[0], v1):
            return 0.0

        min_d = float('inf')
        # Vertex of v1 to edges of v2
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        # Vertex of v2 to edges of v1
        for i in range(6):
            p1 = v2[i]
            for j in range(6):
                p2 = v1[j]
                p3 = v1[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        
        return min_d

    def evaluate_violation(params, outer_s):
        """
        params: [x1, y1, a1, ..., x12, y12, a12]
        a in radians.
        """
        # Reshape
        pos = params.reshape(12, 3)
        xs = pos[:,0]
        ys = pos[:,1]
        rots = pos[:,2]
        
        # Precompute vertices for all inner hexagons
        inner_verts = []
        for i in range(12):
            v = hex_verts(xs[i], ys[i], 1.0, np.rad2deg(rots[i]))
            inner_verts.append(v)
        
        violation = 0.0
        
        # 1. Containment
        r = outer_s * SQRT3 / 2.0
        for i in range(12):
            for v in inner_verts[i]:
                # dot(v, n) <= r
                # n is outward normal
                # We have normals_x, normals_y arrays
                dists = v[0]*normals_x + v[1]*normals_y
                out = np.maximum(0, dists - r)
                violation += np.sum(out)
        
        # 2. Overlap
        for i in range(12):
            for j in range(i+1, 12):
                # Check distance between centers first for speed?
                # If centers are far enough, skip detailed check.
                # Max distance from center to vertex of hex is 1.0 (radius).
                # So if dist > 2.0, they definitely don't touch.
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_center = math.sqrt(dx*dx + dy*dy)
                if d_center > 2.1:
                    continue
                
                d = min_dist_polygons(inner_verts[i], inner_verts[j])
                if d < 1e-6:
                    # Intersecting or touching
                    # Need a penalty that pushes them apart.
                    # If d is 0, they overlap. We need to estimate penetration depth?
                    # For convex polygons, if they intersect, min_dist returns 0.
                    # We can use a penalty based on d.
                    # If d < 0 (impossible here as we return 0 for intersect), 
                    # we treat intersect as a large penalty.
                    # But we need a gradient.
                    # Let's use a quadratic penalty on "negative distance".
                    # Since we can't easily get negative distance, we approximate.
                    # If they intersect, penalty is large.
                    # To make it smooth, we can check if they are "close".
                    # But 0 is a cliff.
                    # Alternative: use distance between centers minus minimum possible distance?
                    # Min distance between centers for non-overlapping unit hexagons depends on orientation.
                    # It varies from sqrt(3) (flat to flat) to 2.0 (point to point).
                    # Average is around 1.732.
                    # Let's use a penalty: max(0, min_center_dist - d_center)^2 ?
                    # This is not accurate but provides gradient.
                    
                    # Better: if d < 0.01, they are effectively touching/overlapping.
                    # We want to push them apart.
                    # We can compute the vector between centers and penalize closeness.
                    # Let's use a fixed threshold for "collision".
                    # If d < 0.05, add penalty.
                    violation += (0.05 - d) ** 2 * 10.0
                    # Also add a center-based penalty to guide away
                    # Min separation is ~1.732. 
                    # If d_center < 1.732, add penalty.
                    if d_center < 1.732:
                        violation += (1.732 - d_center) ** 2 * 5.0
                        
        return violation

    # Binary Search Strategy
    LB = 3.0
    UB = 8.0
    
    best_solution = None
    best_s = float('inf')
    
    def get_initial_positions():
        # Grid 4x3
        pos = np.zeros((12, 3))
        for i in range(12):
            col = i % 4
            row = i // 4
            pos[i,0] = (col - 1.5) * 1.8
            pos[i,1] = (row - 1.0) * 1.6
            pos[i,2] = 0.0
        return pos.flatten()

    max_iter_nm = 300
    tol = 1e-4
    
    # We will binary search S
    # For each S, we run optimization.
    
    for step in range(25):
        if time.time() > deadline:
            break
            
        mid_s = (LB + UB) / 2.0
        current_outer_s = mid_s
        
        found_valid = False
        best_violation = float('inf')
        best_params = None
        
        # 2 restarts
        for restart in range(2):
            if time.time() > deadline:
                break
                
            x0 = get_initial_positions()
            x0 += np.random.normal(0, 0.5, x0.shape)
            
            try:
                res = minimize(evaluate_violation, x0, args=(current_outer_s,),
                               method='Nelder-Mead',
                               options={'maxiter': max_iter_nm, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.fun < best_violation:
                    best_violation = res.fun
                    best_params = res.x
                    
                if res.fun < 1e-3:
                    found_valid = True
                    break
            except Exception:
                pass
        
        if found_valid:
            UB = mid_s
            best_solution = best_params.copy()
            best_s = mid_s
        else:
            LB = mid_s
            
    if best_solution is None:
        # Fallback
        inner = []
        for i in range(12):
            x = (i % 4) * 2.5 - 3.75
            y = (i // 4) * 2.2 - 2.2
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    # Extract
    pos = best_solution.reshape(12, 3)
    inner = []
    for i in range(12):
        inner.append([pos[i,0], pos[i,1], np.rad2deg(pos[i,2])])
        
    return inner, (0.0, 0.0), best_s, 0.0

