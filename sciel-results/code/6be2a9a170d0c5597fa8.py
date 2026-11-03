import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Vertices of a regular hexagon with given center, side length, and rotation."""
    angle = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def support_function_hex(point, cx, cy, side, angle_deg=0.0):
    """Support function of a regular hexagon: max dot product of point direction with vertices."""
    angle = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 6) + angle
    verts_x = cx + side * np.cos(angles)
    verts_y = cy + side * np.sin(angles)
    return np.max(point[0] * verts_x + point[1] * verts_y)

def min_enclosing_side(inner_hexes, outer_center, outer_angle_deg, tol=1e-9):
    """
    Find minimum side length of outer hexagon (at given center and angle) 
    that contains all inner hexagons.
    Uses the fact that for containment, for all 6 directions of the outer hexagon,
    the support of the outer must be >= support of each inner.
    """
    outer_angle = np.radians(outer_angle_deg)
    # 6 outward normals of the outer hexagon
    normals = np.column_stack([
        np.cos(outer_angle + np.linspace(0, 2*np.pi, 7)[:-1]),
        np.sin(outer_angle + np.linspace(0, 2*np.pi, 7)[:-1])
    ])
    
    # For each inner hexagon, compute its support in each of the 6 normal directions
    # The outer hexagon with center oc, side s has support = dot(oc, n) + s * (inradius of hex in that direction)
    # For a regular hexagon with side s, the support in direction making angle theta with a vertex direction is s * cos(theta) for |theta| <= 60 deg
    # Actually, the support function of a regular hexagon with side s centered at origin:
    # h(n) = s * max over vertices of dot(n, v) where v are unit vertex positions
    # For regular hexagon, vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # h(n) = s * max(cos(n_angle - k*pi/3)) for k=0..5
    
    # For containment: for each normal n_i of outer, 
    #   dot(oc, n_i) + s * h_unit(n_i - outer_angle) >= max over inner j of support_j(n_i)
    # where h_unit is support of unit hexagon
    
    # h_unit for a unit regular hexagon in direction with angle phi relative to its own axes:
    # The vertices are at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
    # h_unit(phi) = max_k cos(phi - k*pi/3)
    
    # For the outer hexagon rotated by outer_angle, its "local" direction for normal n_i is:
    # phi_i = angle(n_i) - outer_angle = (i*pi/3 + outer_angle) - outer_angle = i*pi/3
    # So h_unit at local angle i*pi/3 = max_k cos(i*pi/3 - k*pi/3) = 1 (when k=i)
    # Wait, that's not right. Let me reconsider.
    
    # Actually for a regular hexagon with side length s, centered at origin, with one vertex at angle 0:
    # vertices at (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5
    # Support in direction (cos(phi), sin(phi)): h(phi) = s * max_k cos(phi - k*pi/3)
    
    # For the outer hexagon rotated by outer_angle, normal n_i has global angle (i*pi/3 + outer_angle)
    # In the outer hexagon's local frame, this is at angle i*pi/3
    # h_local(i*pi/3) = max_k cos(i*pi/3 - k*pi/3) = 1 (achieved at k=i)
    # So the support of the outer hexagon in direction n_i is: dot(oc, n_i) + s * 1 = dot(oc, n_i) + s
    
    # For inner hexagon j with center c_j, side 1, rotation a_j:
    # Its support in direction n_i: dot(c_j, n_i) + 1 * max_k cos(n_i_angle - a_j - k*pi/3)
    
    required = []
    for i in range(6):
        n = normals[i]
        n_angle = np.arctan2(n[1], n[0])
        max_inner_support = 0
        for j in range(len(inner_hexes)):
            cx_j, cy_j, a_j_deg = inner_hexes[j]
            a_j = np.radians(a_j_deg)
            # Support of unit hexagon centered at (cx_j, cy_j) with rotation a_j in direction n
            local_angle = n_angle - a_j
            # h_unit(local_angle) = max_k cos(local_angle - k*pi/3)
            k_angles = np.linspace(0, 2*np.pi, 6)
            h_val = np.max(np.cos(local_angle - k_angles))
            sup = cx_j * n[0] + cy_j * n[1] + h_val
            if sup > max_inner_support:
                max_inner_support = sup
        # Need: dot(oc, n_i) + s >= max_inner_support
        # s >= max_inner_support - dot(oc, n_i)
        req = max_inner_support - (outer_center[0] * n[0] + outer_center[1] * n[1])
        required.append(req)
    
    return max(required)

def objective(params, inner_hexes_fixed=None):
    """
    params: [cx_0, cy_0, a_0, cx_1, cy_1, a_1, ..., cx_N-1, cy_N-1, a_N-1, ox, oy, oa]
    where a are in radians, oa is outer angle in radians
    """
    if time.time() > DEADLINE:
        return 1e10
    
    n = N
    # Parse inner hexagons
    cx = params[0:3*n:3]
    cy = params[1:3*n:3]
    a = params[2:3*n:3]
    ox = params[3*n]
    oy = params[3*n+1]
    oa = params[3*n+2]
    
    inner = list(zip(cx.tolist(), cy.tolist(), (a * 180/np.pi).tolist()))
    
    # Non-overlap: for each pair, check minimum distance
    # Two unit hexagons with relative position d and relative angle delta don't overlap
    # if the Minkowski difference doesn't contain origin
    # For hexagons, non-overlap: for all 6 directions of one hex, 
    # the support of the Minkowski sum >= distance
    # Simpler: check if distance between centers > some threshold based on relative angle
    
    # Use smooth penalty for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[j] - cx[i]
            dy = cy[j] - cy[i]
            dist = np.sqrt(dx*dx + dy*dy)
            # Relative rotation
            rel_angle = a[j] - a[i]
            # Minimum distance for non-overlap of two unit hexagons
            # This is complex; use a conservative estimate
            # The maximum extent of a unit hexagon in any direction is 1 (circumradius)
            # So if dist < 2, they might overlap. Use smooth penalty.
            # Better: compute actual Minkowski sum check
            # For now, use: if dist < 1.8, add penalty
            if dist < 1.8:
                # Smooth penalty: (1.8 - dist)^2 if positive
                penalty += (1.8 - dist)**2
    
    # Containment: outer side must be large enough
    side = min_enclosing_side(inner, (ox, oy), np.degrees(oa))
    
    return side + 10.0 * penalty

def construct():
    start_time = time.time()
    
    # Try multiple initializations
    best_result = None
    best_score = float('inf')
    
    # Hexagonal lattice initialization
    # Place hexagons in a hexagonal pattern
    # For N=11, we can use a 3-3-3-2 arrangement or similar
    
    initializations = []
    
    # Init 1: Hexagonal lattice centered at origin
    # Rows: -2, -1, 0, 1, 2 with appropriate counts
    # For 11 hexagons: row -2: 2, row -1: 3, row 0: 3, row 1: 3, row 2: 2
    # Or: center + 6 around + 4 more
    # Let's try: one center, 6 in first ring, 4 in second ring
    
    # Actually, let's use a structured approach
    # Hexagonal grid positions with spacing ~2 (just non-overlapping)
    # Then optimize
    
    # Init 1: Linear arrangement (like seed but better)
    init1 = []
    for k in range(-5, 6):
        init1.append([2.1 * k, 0.0, 0.0])
    initializations.append(init1)
    
    # Init 2: Hexagonal pattern
    # Center at origin, 6 around, then 4 more
    init2 = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = k * np.pi / 3
        init2.append([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0])
    # Add 4 more in second ring
    for k in [0, 1, 3, 4]:
        angle = k * np.pi / 3
        init2.append([4.0 * np.cos(angle), 4.0 * np.sin(angle), 0.0])
    initializations.append(init2)
    
    # Init 3: Two-row pattern
    init3 = []
    for k in range(-2, 3):
        init3.append([2.0 * k, 0.0, 0.0])
        if k < 2:
            init3.append([2.0 * k + 1.0, 1.732, 0.0])
    initializations.append(init3)
    
    # Init 4: Compact cluster
    init4 = []
    positions = [(0,0), (2,0), (1,1.732), (-1,1.732), (-2,0), (-1,-1.732), (1,-1.732),
                 (3,0), (2,1.732), (0,3.464), (-2,1.732), (-3,0)]
    for x, y in positions:
        init4.append([float(x), float(y), 0.0])
    initializations.append(init4)
    
    for init_idx, init in enumerate(initializations):
        if time.time() > DEADLINE - 5:
            break
        
        # Build initial params
        params0 = []
        for x, y, a_deg in init:
            params0.extend([x, y, np.radians(a_deg)])
        # Outer hexagon: center at centroid, angle 0
        cx_vals = [p[0] for p in init]
        cy_vals = [p[1] for p in init]
        ox0 = np.mean(cx_vals)
        oy0 = np.mean(cy_vals)
        params0.extend([ox0, oy0, 0.0])
        
        # Optimize
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                             options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.success or result.fun < best_score:
                p = result.x
                cx = p[0:3*N:3]
                cy = p[1:3*N:3]
                a = p[2:3*N:3]
                ox = p[3*N]
                oy = p[3*N+1]
                oa = p[3*N+2]
                
                inner = list(zip(cx.tolist(), cy.tolist(), (a * 180/np.pi).tolist()))
                side = min_enclosing_side(inner, (ox, oy), np.degrees(oa))
                
                # Verify no overlaps
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = inner[j][0] - inner[i][0]
                        dy = inner[j][1] - inner[i][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 1.7:  # conservative check
                            valid = False
                
                if valid and side < best_score:
                    best_score = side
                    best_result = (inner, (ox, oy), side, np.degrees(oa))
        except Exception as e:
            pass
    
    # Fallback if optimization failed
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

construct()
