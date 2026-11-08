import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _support_unit_hex(angle_deg, dir_vec):
    """
    Support function of a unit regular hexagon centered at origin,
    rotated by angle_deg, in direction dir_vec (unit vector).
    """
    a = np.radians(angle_deg)
    # Vertices of unit hex
    angles = a + np.radians(60 * np.arange(6))
    vs = np.array([np.cos(angles), np.sin(angles)]).T
    dots = vs @ dir_vec
    return np.max(dots)

def _min_outer_side(params, outer_angle_deg):
    """
    Calculate the minimum side length of a regular hexagon (centered at 0,0)
    with rotation outer_angle_deg that contains all N unit hexagons defined by params.
    params: [x0, y0, a0, x1, y1, a1, ...]
    """
    outer_a = np.radians(outer_angle_deg)
    # 6 outward normals of the outer hexagon
    normals = np.array([
        [np.cos(outer_a + np.radians(60 * i)),
         np.sin(outer_a + np.radians(60 * i))]
        for i in range(6)
    ])
    
    max_proj = 0.0
    for k in range(N):
        cx = params[3*k]
        cy = params[3*k+1]
        ang = params[3*k+2]
        
        for ni in range(6):
            n = normals[ni]
            # Projection of center
            proj_c = cx * n[0] + cy * n[1]
            # Support of unit hex in direction n
            h = _support_unit_hex(ang, n)
            total_proj = proj_c + h
            if total_proj > max_proj:
                max_proj = total_proj
                
    # Apothem A = max_proj
    # Side S = 2 * A / sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def _overlap_penalty(params):
    """
    Calculate total overlap penalty for all pairs of hexagons.
    Uses Separating Axis Theorem (SAT) for regular hexagons.
    """
    penalty = 0.0
    # Precompute vertices for all hexagons? 
    # It might be faster to compute projections directly.
    
    # Extract centers and angles
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i+1]
        angles[i] = params[3*i+2]
        
    # Precompute unit vertex vectors for each hexagon (relative to center)
    # v_k[i] = (cos(a_k + 60i), sin(a_k + 60i))
    unit_verts = np.zeros((N, 6, 2))
    for k in range(N):
        a = np.radians(angles[k])
        unit_verts[k, :, 0] = np.cos(a + np.radians(60 * np.arange(6)))
        unit_verts[k, :, 1] = np.sin(a + np.radians(60 * np.arange(6)))
        
    # Edge normals for each hexagon
    # The normals are perpendicular to edges.
    # For a regular hexagon, the outward normals point to the midpoints of edges.
    # The angle of the normal for edge i is a + 60i + 30 degrees? 
    # Let's check: Vertex 0 is at angle a. Edge 0-1 is between a and a+60.
    # Midpoint is at a+30. Normal is at a+30.
    # So normals are at a + 30 + 60i.
    
    normals = np.zeros((N, 6, 2))
    for k in range(N):
        a = np.radians(angles[k])
        normals[k, :, 0] = np.cos(a + np.radians(30 + 60 * np.arange(6)))
        normals[k, :, 1] = np.sin(a + np.radians(30 + 60 * np.arange(6)))

    for i in range(N):
        for j in range(i+1, N):
            # Check separation using normals of i and j
            # If separated on any axis, no overlap.
            separated = False
            min_overlap = float('inf')
            
            # Axes from i
            for ni in range(6):
                n = normals[i, ni]
                # Project centers
                d_c = (centers[j] - centers[i]) @ n
                # Project extents
                # Extent of hex i in direction n is the max dot of vertices with n
                # Since n is a normal of i, the extent is the apothem of unit hex = sqrt(3)/2
                # Wait, the support function in the normal direction of a regular hexagon is indeed the apothem.
                # Let's verify. Vertices are at angles a+60k. Normal at a+30+60m.
                # Dot product is cos(a+60k - (a+30+60m)) = cos(60(k-m) - 30).
                # Max value of cos(theta) for theta in {30, 90, 150, 210, 270, 330} is cos(30) = sqrt(3)/2.
                # So extent_i = sqrt(3)/2.
                # Similarly extent_j = max dot of vertices of j with n.
                
                ext_i = np.sqrt(3.0)/2.0
                ext_j = np.max(unit_verts[j] @ n)
                
                # Separation condition: |d_c| > ext_i + ext_j
                # Overlap depth: ext_i + ext_j - |d_c|
                if abs(d_c) > ext_i + ext_j:
                    separated = True
                    break
                else:
                    overlap = ext_i + ext_j - abs(d_c)
                    if overlap < min_overlap:
                        min_overlap = overlap
                        
            if separated:
                continue
                
            # Axes from j
            for nj in range(6):
                n = normals[j, nj]
                d_c = (centers[i] - centers[j]) @ n
                ext_j = np.sqrt(3.0)/2.0
                ext_i = np.max(unit_verts[i] @ n)
                
                if abs(d_c) > ext_i + ext_j:
                    separated = True
                    break
                else:
                    overlap = ext_i + ext_j - abs(d_c)
                    if overlap < min_overlap:
                        min_overlap = overlap
                        
            if not separated:
                penalty += min_overlap
                
    return penalty

def _objective(params, outer_angle_deg):
    S = _min_outer_side(params, outer_angle_deg)
    overlap = _overlap_penalty(params)
    return S + 100.0 * overlap

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    best_S = float('inf')
    best_params = None
    best_outer_angle = 0.0
    
    # Initial guess: Hex lattice
    # 3 rows, 4 cols
    hex_positions = []
    for r in range(3):
        for c in range(4):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * np.sqrt(3)
            hex_positions.append([x, y, 0.0])
            
    # Center
    cx_mean = np.mean([p[0] for p in hex_positions])
    cy_mean = np.mean([p[1] for p in hex_positions])
    
    init_params = []
    for (x, y, a) in hex_positions:
        init_params.extend([x - cx_mean, y - cy_mean, a])
    init_params = np.array(init_params)
    
    bounds = []
    for i in range(N):
        bounds.append((-6.0, 6.0))
        bounds.append((-6.0, 6.0))
        bounds.append((0.0, 60.0))
        
    for outer_angle in [0.0, 30.0]:
        if time.time() > deadline - 30.0:
            break
            
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(outer_angle,),
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                polish=True,
                init='latinhypercube'
            )
            opt_params = result.x
            S_val = _min_outer_side(opt_params, outer_angle)
            overlap_pen = _overlap_penalty(opt_params)
            
            if overlap_pen < 1e-6 and S_val < best_S:
                best_S = S_val
                best_params = opt_params.copy()
                best_outer_angle = outer_angle
        except Exception:
            continue
        
        # Polish
        if time.time() < deadline - 10.0 and best_params is not None:
            try:
                result2 = minimize(
                    _objective,
                    best_params,
                    args=(best_outer_angle,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                opt2 = result2.x
                S2 = _min_outer_side(opt2, best_outer_angle)
                op2 = _overlap_penalty(opt2)
                if op2 < 1e-6 and S2 < best_S:
                    best_S = S2
                    best_params = opt2.copy()
            except Exception:
                pass

    if best_params is None:
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 6.0, 0.0

    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        ang = float(best_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
        
    outer_verts = _hex_verts(0.0, 0.0, best_S, best_outer_angle)
    
    def point_in_hex(pt, outer_verts):
        for i in range(6):
            v1 = outer_verts[i]
            v2 = outer_verts[(i+1) % 6]
            edge = v2 - v1
            rel = pt - v1
            cross = edge[0]*rel[1] - edge[1]*rel[0]
            if cross < -1e-10:
                return False
        return True
        
    valid = True
    for iv in inner_verts_list:
        for pt in iv:
            if not point_in_hex(pt, outer_verts):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        best_S *= 1.05
        
    return inner, (0.0, 0.0), float(best_S), float(best_outer_angle)
