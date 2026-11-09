"""Strategy: Structured lattice initialization + smooth SAT-based penalty minimization with scipy L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def remaining_time():
        return max(0.0, deadline - time.time())

    # Helper: vertices of a unit regular hexagon centered at (cx, cy) with rotation angle (radians)
    def hex_vertices(cx, cy, angle):
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + angle
        xs = cx + np.cos(angles)
        ys = cy + np.sin(angles)
        return np.column_stack((xs, ys))

    # Helper: check if point (px, py) is inside hexagon centered at (cx, cy) with rotation angle (radians)
    # The hexagon has side length 1. The distance from center to edge (inradius) is sqrt(3)/2.
    # The normals to the edges are at angles angle + 30, 90, 150, 210, 270, 330 degrees.
    # A point is inside if its projection onto each normal is <= sqrt(3)/2.
    def point_in_hex(px, py, cx, cy, angle):
        dx = px - cx
        dy = py - cy
        # Rotate point into local frame
        ca, sa = np.cos(-angle), np.sin(-angle)
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        
        # Normals in local frame are at 30, 90, 150, 210, 270, 330 degrees
        normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
        nx = np.cos(normals)
        ny = np.sin(normals)
        
        dists = lx * nx + ly * ny
        inradius = np.sqrt(3) / 2.0
        
        return np.all(dists <= inradius + 1e-9)

    # Helper: check if two hexagons are disjoint
    # We use a SAT-like approach:
    # 1. Check if any vertex of A is inside B
    # 2. Check if any vertex of B is inside A
    # 3. Check if any edges intersect
    
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

    def seg_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

    def are_disjoint(v1, v2):
        # v1, v2 are (6, 2) arrays of vertices
        for v in v1:
            if point_in_poly(v[0], v[1], v2):
                return False
        for v in v2:
            if point_in_poly(v[0], v[1], v1):
                return False
        
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if seg_intersect(p1, p2, p3, p4):
                    return False
        return True

    # Helper: check if a set of hexagons is inside the outer hexagon
    # Outer hexagon: center (cx, cy), side S, rotation angle (radians)
    # Inradius = S * sqrt(3) / 2
    def is_inside_outer(px, py, cx, cy, S, angle):
        dx = px - cx
        dy = py - cy
        ca, sa = np.cos(-angle), np.sin(-angle)
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        
        normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
        nx = np.cos(normals)
        ny = np.sin(normals)
        
        dists = lx * nx + ly * ny
        inradius = S * np.sqrt(3) / 2.0
        
        return np.all(dists <= inradius + 1e-9)

    # Objective function for L-BFGS-B
    # Variables: x = [x_1, y_1, a_1, ..., x_N, y_N, a_N, cx, cy, S, a_out]
    # We want to minimize S, subject to constraints.
    # Since L-BFGS-B doesn't handle hard constraints well, we use a penalty method.
    # Penalty = sum of violations^2
    
    def objective(x):
        # Unpack
        centers = x[:N*2].reshape(N, 2)
        angles = x[N*2:N*3]
        cx = x[N*3]
        cy = x[N*3+1]
        S = x[N*3+2]
        a_out = x[N*3+3]
        
        # Penalty for outer boundary violations
        penalty = 0.0
        for i in range(N):
            verts = hex_vertices(centers[i,0], centers[i,1], angles[i])
            for v in verts:
                # Distance to boundary in local frame
                dx = v[0] - cx
                dy = v[1] - cy
                ca, sa = np.cos(-a_out), np.sin(-a_out)
                lx = ca * dx - sa * dy
                ly = sa * dx + ca * dy
                
                normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
                nx = np.cos(normals)
                ny = np.sin(normals)
                
                dists = lx * nx + ly * ny
                inradius = S * np.sqrt(3) / 2.0
                
                # Violation is max(0, dist - inradius)
                viol = np.maximum(0, dists - inradius)
                penalty += np.sum(viol**2)
        
        # Penalty for pairwise overlaps
        # For each pair, we need a smooth measure of overlap.
        # Approximation: if distance between centers < 2, they might overlap.
        # A better smooth penalty: use the minimum distance between edges.
        # For speed, we can use a proxy: if centers are close, push them apart.
        # Let's use a soft potential: f = max(0, 2.0 - dist)^2
        # This is not exact but helps guide the optimizer.
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[j,0] - centers[i,0]
                dy = centers[j,1] - centers[i,1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Threshold 2.0 is the max distance between centers of touching hexagons
                # (actually it's sqrt(3) for edge-to-edge, but 2.0 is safe for vertex-to-vertex)
                # Let's use 1.8 as a threshold to be tighter.
                if dist < 1.8:
                    penalty += (1.8 - dist)**2 * 10.0
        
        # Return S + penalty
        # We want to minimize S, but the penalty ensures validity.
        # As S decreases, the penalty for boundary violations increases, balancing the trade-off.
        return S + penalty

    # Initial guess: Hexagonal lattice
    sqrt3 = np.sqrt(3)
    
    # 11 hexagons in a compact cluster
    # Axial coordinates for a 3x3 block minus corners?
    # Let's use:
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # (2,0), (2,-1), (1,-2)
    # Total 11
    
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2),
    ]
    
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Flatten initial state
    # x = [centers, angles, cx, cy, S, a_out]
    x0 = np.zeros(N*3 + 4)
    x0[:N*2] = centers.flatten()
    x0[N*2:N*3] = 0.0 # Angles
    x0[N*3] = 0.0 # cx
    x0[N*3+1] = 0.0 # cy
    x0[N*3+2] = 5.0 # S (initial guess)
    x0[N*3+3] = 0.0 # a_out
    
    # Run L-BFGS-B
    best_result = None
    best_S = float('inf')
    
    # Try a few random restarts for angles and S
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb initial angles
        angles_init = np.random.uniform(0, 2*np.pi, N)
        # Perturb initial S
        S_init = 4.0 + np.random.uniform(0, 2.0)
        
        x_init = x0.copy()
        x_init[N*2:N*3] = angles_init
        x_init[N*3+2] = S_init
        
        # Optimize
        res = minimize(objective, x_init, method='L-BFGS-B', options={'maxiter': 1000, 'disp': False})
        
        if res.success or res.fun < best_S:
            # Check validity
            centers_opt = res.x[:N*2].reshape(N, 2)
            angles_opt = res.x[N*2:N*3]
            cx_opt = res.x[N*3]
            cy_opt = res.x[N*3+1]
            S_opt = res.x[N*3+2]
            a_out_opt = res.x[N*3+3]
            
            # Verify validity with strict checks
            valid = True
            verts_list = []
            for i in range(N):
                verts = hex_vertices(centers_opt[i,0], centers_opt[i,1], angles_opt[i])
                verts_list.append(verts)
                for v in verts:
                    if not is_inside_outer(v[0], v[1], cx_opt, cy_opt, S_opt, a_out_opt):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        if not are_disjoint(verts_list[i], verts_list[j]):
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                if S_opt < best_S:
                    best_S = S_opt
                    best_result = (centers_opt, angles_opt, cx_opt, cy_opt, S_opt, a_out_opt)
        
        if time.time() > deadline - 10.0:
            break
    
    if best_result is None:
        # Fallback: return the initial lattice with a large S
        centers = x0[:N*2].reshape(N, 2)
        angles = x0[N*2:N*3]
        cx = 0.0
        cy = 0.0
        S = 10.0
        a_out = 0.0
    else:
        centers, angles, cx, cy, S, a_out = best_result
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([centers[i,0], centers[i,1], np.degrees(angles[i])])
    
    return inner, (cx, cy), S, np.degrees(a_out)
