"""Combined program: honeycomb patterns + minimal enclosing hexagon + local refinement."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
sqrt3 = np.sqrt(3.0)

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(np.arange(6) * 60.0)
    verts = s * np.column_stack([np.cos(angles), np.sin(angles)])
    return verts + np.array([cx, cy])

def _hex_face_normals(angle_deg):
    """Get the 6 outward face normals of a regular hexagon with rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(30.0 + np.arange(6) * 60.0)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides):
    """
    Find minimal regular hexagon (center, side, angle) containing all inner hexagons.
    Uses vertex-based approach: collect all vertices, try many outer angles,
    compute widths, find feasible center via least squares.
    """
    all_verts = []
    for (cx, cy), ang, s in zip(inner_positions, inner_angles, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_center = None
    best_angle = 0.0
    
    # Try many outer hexagon angles (restricted to [0, 60) by symmetry)
    for phi_deg in np.linspace(0, 60, 600):
        phi = np.radians(phi_deg)
        normals = _hex_face_normals(phi_deg)  # 6x2
        
        # Compute projections of all vertices onto each normal
        projs = all_verts @ normals.T  # (num_verts, 6)
        max_projs = projs.max(axis=0)  # (6,)
        min_projs = projs.min(axis=0)  # (6,)
        widths = max_projs - min_projs  # (6,)
        
        # Required apothem: a >= max(widths) / 2
        required_apothem = np.max(widths) / 2.0
        s_candidate = required_apothem * 2.0 / sqrt3
        
        # Find center c such that dot(c, n_i) is in [min_projs[i] + a, max_projs[i] - a]
        lo = min_projs + required_apothem
        hi = max_projs - required_apothem
        
        if np.any(lo > hi + 1e-12):
            continue
        
        # Use midpoint as target
        mid = (lo + hi) / 2.0
        
        # Solve least squares: normals @ c = mid
        c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
        
        # Verify
        c_projs = normals @ c
        if np.any(c_projs < lo - 1e-9) or np.any(c_projs > hi + 1e-9):
            continue
        
        if s_candidate < best_side:
            best_side = s_candidate
            best_center = c
            best_angle = phi_deg
    
    if best_center is None:
        centroid = all_verts.mean(axis=0)
        return centroid[0], centroid[1], 100.0, 0.0
    
    return best_center[0], best_center[1], best_side, best_angle

def _check_disjointness(positions, angles, sides, tol=1e-6):
    """Check that all pairs of hexagons are disjoint (no vertex inside another, no edge intersection)."""
    n = len(positions)
    unit_angles = np.radians(np.arange(6) * 60.0)
    unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])
    
    # Precompute vertices for each hexagon
    verts_list = []
    for j in range(n):
        cx, cy = positions[j]
        a = np.radians(angles[j])
        s = sides[j]
        cos_a, sin_a = np.cos(a), np.sin(a)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rot_verts = unit_verts @ rot_mat.T
        verts_j = s * rot_verts + np.array([cx, cy])
        verts_list.append(verts_j)
    
    def point_in_hex(px, py, cx_h, cy_h, angle_deg_h, side_h):
        normals_h = _hex_face_normals(angle_deg_h)
        apothem_h = side_h * sqrt3 / 2.0
        for i in range(6):
            d_i = normals_h[i]
            val = d_i[0] * (px - cx_h) + d_i[1] * (py - cy_h)
            if val > apothem_h + tol:
                return False
        return True
    
    for j in range(n):
        for k in range(j + 1, n):
            # Check if any vertex of j is inside k
            for v in verts_list[j]:
                if point_in_hex(v[0], v[1], positions[k][0], positions[k][1], angles[k], sides[k]):
                    return False
            # Check if any vertex of k is inside j
            for v in verts_list[k]:
                if point_in_hex(v[0], v[1], positions[j][0], positions[j][1], angles[j], sides[j]):
                    return False
    return True

def _refine(positions, angles, outer_center, outer_side, outer_angle, time_limit=10.0):
    """
    Local refinement using physics-based repulsion to fix any small overlaps
    while maintaining containment in the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    n = len(positions)
    sides = [1.0] * n
    
    # Build initial parameter vector
    x0 = np.zeros(3 * n)
    for j in range(n):
        x0[3*j] = positions[j][0]
        x0[3*j+1] = positions[j][1]
        x0[3*j+2] = angles[j]
    
    outer_normals = _hex_face_normals(outer_angle)
    apothem = outer_side * sqrt3 / 2.0
    
    unit_angles = np.radians(np.arange(6) * 60.0)
    unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])
    
    def energy(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        
        total = 0.0
        pos = []
        angs = []
        for j in range(n):
            x = params[3*j]
            y = params[3*j+1]
            ang = params[3*j+2]
            pos.append((x, y))
            angs.append(ang)
        
        # Containment penalty
        for j in range(n):
            x_j, y_j = pos[j]
            a_j = np.radians(angs[j])
            cos_a, sin_a = np.cos(a_j), np.sin(a_j)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rot_verts = unit_verts @ rot_mat.T
            verts_j = rot_verts + np.array([x_j, y_j])
            for v in verts_j:
                for i in range(6):
                    d_i = outer_normals[i]
                    val = d_i[0] * (v[0] - outer_center[0]) + d_i[1] * (v[1] - outer_center[1])
                    if val > apothem:
                        total += (val - apothem) ** 2 * 100.0
        
        # Repulsion: push overlapping hexagons apart
        for j in range(n):
            for k in range(j + 1, n):
                x_j, y_j = pos[j]
                x_k, y_k = pos[k]
                dx = x_j - x_k
                dy = y_j - y_k
                dist = np.sqrt(dx*dx + dy*dy)
                # Minimum center distance for non-overlap (conservative)
                min_dist = 1.732  # sqrt(3) ~ 1.732
                if dist < min_dist:
                    if dist < 1e-6:
                        dist = 1e-6
                        dx, dy = 1.0, 0.0
                    overlap = min_dist - dist
                    total += overlap ** 2 * 50.0
        
        return total
    
    # Use Nelder-Mead for local refinement
    res = minimize(energy, x0, method='Nelder-Mead',
                   options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
    
    new_positions = [(res.x[3*j], res.x[3*j+1]) for j in range(n)]
    new_angles = [res.x[3*j+2] for j in range(n)]
    
    # Verify validity
    if _check_disjointness(new_positions, new_angles, sides):
        # Check containment
        valid = True
        for j in range(n):
            x_j, y_j = new_positions[j]
            a_j = np.radians(new_angles[j])
            cos_a, sin_a = np.cos(a_j), np.sin(a_j)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rot_verts = unit_verts @ rot_mat.T
            verts_j = rot_verts + np.array([x_j, y_j])
            for v in verts_j:
                for i in range(6):
                    d_i = outer_normals[i]
                    val = d_i[0] * (v[0] - outer_center[0]) + d_i[1] * (v[1] - outer_center[1])
                    if val > apothem + 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return new_positions, new_angles
    
    return positions, angles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_result = None
    best_score = float('inf')
    
    # Generate honeycomb positions with various patterns
    # Scale factor 1.001 ensures slight separation for numerical safety
    scale = 1.001
    
    def honeycomb_positions(count, pattern):
        """Generate positions for 'count' hexagons in honeycomb pattern."""
        positions = []
        for j, row_len in enumerate(pattern):
            for i in range(row_len):
                x = (i + 0.5 * (j % 2)) * sqrt3 * scale
                y = j * 1.5 * scale
                positions.append((x, y))
        return positions[:count]
    
    # Try several row patterns that sum to 12
    patterns = [
        [3, 4, 3, 2],
        [4, 3, 3, 2],
        [2, 3, 4, 3],
        [3, 3, 4, 2],
        [4, 4, 2, 2],
        [2, 2, 4, 4],
        [5, 3, 2, 2],
        [2, 2, 3, 5],
        [6, 3, 3],
        [3, 3, 6],
        [4, 4, 4],
        [2, 4, 4, 2],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [3, 5, 4],
        [4, 5, 3],
        [5, 4, 3],
        [3, 4, 5],
        [5, 4, 2, 1],
        [1, 2, 4, 5],
        [2, 5, 4, 1],
        [4, 4, 3, 1],
        [1, 3, 4, 4],
        [3, 3, 3, 3],
        [6, 6],
        [2, 3, 3, 4],
        [4, 3, 3, 2],
        [3, 2, 3, 4],
        [2, 4, 3, 3],
    ]
    
    for pattern in patterns:
        if time.time() > deadline - 20.0:
            break
        
        if sum(pattern) != 12:
            continue
        if len(pattern) > 8:
            continue
        
        positions = honeycomb_positions(12, pattern)
        if len(positions) != 12:
            continue
        
        # Try a few rotation angles for inner hexagons
        for rot in [0.0, 30.0, 15.0, 45.0]:
            if time.time() > deadline - 15.0:
                break
            
            angles = [rot] * 12
            sides = [1.0] * 12
            
            cx, cy, side, angle = _min_enclosing_hexagon(positions, angles, sides)
            
            # Add small safety margin
            side_safe = side * 1.0001 + 1e-6
            
            if side_safe < best_score:
                # Try local refinement to potentially improve
                refined_pos, refined_ang = _refine(positions, angles, (cx, cy), side_safe, angle, time_limit=8.0)
                
                # Recompute enclosing hexagon with refined positions
                cx2, cy2, side2, angle2 = _min_enclosing_hexagon(refined_pos, refined_ang, sides)
                side2_safe = side2 * 1.0001 + 1e-6
                
                if side2_safe < side_safe:
                    side_safe = side2_safe
                    positions = refined_pos
                    angles = refined_ang
                    cx, cy = cx2, cy2
                    angle = angle2
                
                best_score = side_safe
                best_result = (positions, angles, (cx, cy), side_safe, angle)
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [(4.0 * k, 0.0, 0.0) for k in range(-6, 6)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    positions, angles, (cx, cy), side, angle = best_result
    inner = [[positions[j][0], positions[j][1], angles[j]] for j in range(N)]
    return inner, (cx, cy), side, angle
