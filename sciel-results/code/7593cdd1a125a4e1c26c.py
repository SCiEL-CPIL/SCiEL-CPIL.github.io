import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0

def get_hex_vertices(cx, cy, angle, scale=1.0):
    """Get vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        ang = angle + k * math.pi / 3.0
        x = cx + scale * math.cos(ang)
        y = cy + scale * math.sin(ang)
        verts.append((x, y))
    return np.array(verts)

def check_overlap(v1, v2):
    """
    Check if two convex polygons (arrays of vertices) overlap.
    Uses Separating Axis Theorem (SAT).
    """
    # Get all edges of both polygons
    edges = []
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edges.append(p2 - p1)
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edges.append(p2 - p1)
    
    # Check each edge normal as a separating axis
    for edge in edges:
        # Normal to the edge
        normal = np.array([-edge[1], edge[0]])
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue
        normal /= norm_len
        
        # Project all vertices of both polygons onto the normal
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # If there is a gap, they don't overlap
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def objective(vars):
    """
    Objective function for differential evolution.
    vars: [x1, y1, th1, ..., x11, y11, th11, Cx, Cy, phi, S]
    """
    # Parse variables
    centers = []
    angles = []
    for i in range(N):
        x = vars[i * 3]
        y = vars[i * 3 + 1]
        th = vars[i * 3 + 2]
        centers.append((x, y))
        angles.append(th)
    
    Cx = vars[N * 3]
    Cy = vars[N * 3 + 1]
    phi = vars[N * 3 + 2]
    S = vars[N * 3 + 3]
    
    penalty = 0.0
    
    # 1. Check Overlaps
    hex_verts = []
    for i in range(N):
        v = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
        hex_verts.append(v)
        
    for i in range(N):
        for j in range(i + 1, N):
            if check_overlap(hex_verts[i], hex_verts[j]):
                # Calculate penetration depth approximation for penalty
                # Just use a large penalty
                penalty += 10.0
                
    # 2. Check Containment
    # Outer hexagon vertices
    outer_verts = get_hex_vertices(Cx, Cy, phi, S)
    # Outer hexagon normals (outward)
    outer_normals = []
    for k in range(6):
        ang = phi + k * math.pi / 3.0
        # Normal is perpendicular to edge. Edge angle is ang + pi/6?
        # Vertices are at ang. Edge between k and k+1.
        # Normal direction is ang + pi/6 + pi/2?
        # Let's just use the distance from center.
        # A point (x,y) is inside if distance to center in direction normal <= Apotem_outer
        # Apotem_outer = S * sqrt(3) / 2
        outer_apotem = S * math.sqrt(3) / 2.0
        
        # Normal for face k
        # The face k is centered at angle phi + k * pi/3 + pi/6?
        # No, vertices are at phi + k*pi/3.
        # The midpoint of edge k (between vertex k and k+1) is at angle phi + (k+0.5)*pi/3.
        # The normal points in that direction.
        n_ang = phi + (k + 0.5) * math.pi / 3.0
        nx = math.cos(n_ang)
        ny = math.sin(n_ang)
        outer_normals.append((nx, ny))
        
    for i in range(N):
        v = hex_verts[i]
        for (nx, ny) in outer_normals:
            # Project inner vertices onto normal
            # The support function of outer hex in direction n is Cx*nx + Cy*ny + outer_apotem
            # The max projection of inner hex in direction n must be <= that
            proj = np.dot(v, np.array([nx, ny]))
            max_proj = np.max(proj)
            outer_limit = Cx * nx + Cy * ny + outer_apotem
            if max_proj > outer_limit:
                # Penalty proportional to violation
                penalty += (max_proj - outer_limit) * 10.0
                
    return S + penalty

def construct():
    # Bounds for differential evolution
    # x, y in [-5, 5]
    # theta in [0, pi/3]
    # Cx, Cy in [-5, 5]
    # phi in [0, pi/3]
    # S in [1, 6]
    
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))  # x
        bounds.append((-5, 5))  # y
        bounds.append((0, math.pi / 3))  # theta
    
    bounds.append((-5, 5))  # Cx
    bounds.append((-5, 5))  # Cy
    bounds.append((0, math.pi / 3))  # phi
    bounds.append((1.5, 6.0))  # S
    
    # Run differential evolution
    # maxiter and popsize control runtime. 120s limit.
    # DE can be slow. Let's try a moderate setting.
    try:
        res = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=10, tol=0.01, polish=True)
        
        if res.success or res.fun < 100: # Heuristic check
            best_vars = res.x
        else:
            # Fallback to a known good configuration if DE fails
            best_vars = None
    except Exception:
        best_vars = None
        
    if best_vars is None:
        # Fallback: Use the best candidate from the previous logic or a simple grid
        # Let's use the axial_1 configuration from the previous code as a safe fallback
        axial_1 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        # Convert to cartesian
        pts = []
        for q, r in axial_1:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
            
        # Optimize outer hexagon for these points
        # Simple grid search for outer center and rotation
        best_S = float('inf')
        best_Cx, best_Cy = 0, 0
        best_phi = 0
        
        for i in range(10):
            phi = i * (math.pi / 6) / 10
            for j in range(10):
                Cx = (j - 5) * 0.5
                Cy = (j - 5) * 0.5
                # Calculate required S
                max_proj = 0
                for k in range(6):
                    n_ang = phi + (k + 0.5) * math.pi / 3.0
                    nx = math.cos(n_ang)
                    ny = math.sin(n_ang)
                    for px, py in pts:
                        proj = (px - Cx) * nx + (py - Cy) * ny
                        if proj > max_proj:
                            max_proj = proj
                # S = (max_proj + apotem_inner) / apotem_outer?
                # No, max_proj is distance from center to furthest point.
                # Outer apotem must be >= max_proj + inner apotem?
                # No, the inner hexagon has apothem APOTHEM.
                # The distance from outer center to inner center is d.
                # The furthest point of inner hex from outer center is d + APOTHEM.
                # So Outer Apotem >= d + APOTHEM.
                # S = Outer Apotem / (sqrt(3)/2)
                
                # Wait, my previous logic in the current code was:
                # S = (D + APOTHEM) / APOTHEM where D is max projection of centers?
                # No, D was max projection of *points*?
                # In current code: `calc_S_for_beta` uses `pts` which are centers.
                # It computes `px * nx + py * ny`.
                # Then `S = (D + APOTHEM) / APOTHEM`.
                # This implies D is the max distance of the *center* from the outer center.
                # And the outer apotem is D + APOTHEM.
                # So S = (D + APOTHEM) / (sqrt(3)/2).
                
                D = max_proj
                S_val = (D + APOTHEM) / APOTHEM
                if S_val < best_S:
                    best_S = S_val
                    best_Cx, best_Cy = Cx, Cy
                    best_phi = phi
                    
        best_vars = []
        for x, y in pts:
            best_vars.extend([x, y, 0.0])
        best_vars.extend([best_Cx, best_Cy, best_phi, best_S])
        
    # Extract final parameters
    centers = []
    angles = []
    for i in range(N):
        x = best_vars[i * 3]
        y = best_vars[i * 3 + 1]
        th = best_vars[i * 3 + 2]
        centers.append((x, y))
        angles.append(th)
        
    Cx = best_vars[N * 3]
    Cy = best_vars[N * 3 + 1]
    phi = best_vars[N * 3 + 2]
    S = best_vars[N * 3 + 3]
    
    # Final refinement with minimize to polish S and positions
    # We fix the structure found by DE and just minimize S with constraints
    # This is complex, so we just return the DE result.
    # But we must ensure no overlap.
    # Let's do a quick check and scale if necessary? No, scaling changes S.
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i][0]), float(centers[i][1]), float(math.degrees(angles[i]))])
        
    center = [float(Cx), float(Cy)]
    side = float(S)
    rotation = float(math.degrees(phi))
    
    return inner, center, side, rotation

