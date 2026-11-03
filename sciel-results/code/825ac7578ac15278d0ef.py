import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 110.0  # safety margin before 120s limit

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3.0
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append([x, y])
    return np.array(vertices)

def point_in_polygon(px, py, poly):
    """Ray casting point-in-polygon test."""
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

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (including touching)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    # Check collinear overlap
    if d1 == 0 and ccw(p1, p2, p3) >= -1e-10 and ccw(p1, p2, p3) <= 1e-10:
        return True
    if d2 == 0 and ccw(p1, p2, p4) >= -1e-10 and ccw(p1, p2, p4) <= 1e-10:
        return True
    if d3 == 0 and ccw(p3, p4, p1) >= -1e-10 and ccw(p3, p4, p1) <= 1e-10:
        return True
    if d4 == 0 and ccw(p3, p4, p2) >= -1e-10 and ccw(p3, p4, p2) <= 1e-10:
        return True
    return False

def hexagons_overlap(v1, v2, gap=0.001):
    """Check if two hexagons (as vertex arrays) overlap or touch within gap."""
    # Check if any vertex of v1 is inside v2 (with gap)
    for pt in v1:
        if point_in_polygon(pt[0], pt[1], v2):
            return True
    for pt in v2:
        if point_in_polygon(pt[0], pt[1], v1):
            return True
    # Check edge intersections
    for i in range(6):
        for j in range(6):
            if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def all_inside_hexagon(inner_verts_list, outer_verts):
    """Check all inner hexagon vertices are inside outer hexagon."""
    for verts in inner_verts_list:
        for pt in verts:
            if not point_in_polygon(pt[0], pt[1], outer_verts):
                return False
    return True

def are_disjoint(inner_verts_list, gap=0.001):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner_verts_list)
    for i in range(n):
        for j in range(i+1, n):
            if hexagons_overlap(inner_verts_list[i], inner_verts_list[j], gap):
                return False
    return True

def compute_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side length needed to contain all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_verts = []
    for (x, y, ang) in zip(inner_positions[:, 0], inner_positions[:, 1], inner_angles):
        verts = hexagon_vertices(x, y, 1.0, ang)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    # For a regular hexagon centered at outer_center with rotation outer_angle,
    # the distance from center to each edge (inradius) must be >= max projection
    # of all points onto the normal directions of the 6 edges.
    angle_rad = np.radians(outer_angle)
    # The 6 edge normals for a hexagon with "flat top" orientation
    # Edge normals point outward at angles: 0, 60, 120, 180, 240, 300 (for pointy-top)
    # For a regular hexagon, the inradius r = side * sqrt(3)/2
    # A point (px, py) relative to center is inside if for all 6 normals n_k:
    #   dot(point - center, n_k) <= r
    # So we need r >= max over all points and all 6 normals of dot(point-center, n_k)
    
    rel = all_verts - np.array(outer_center)
    max_proj = 0.0
    for k in range(6):
        a = angle_rad + k * np.pi / 3.0
        nx = np.cos(a)
        ny = np.sin(a)
        projs = rel[:, 0] * nx + rel[:, 1] * ny
        max_proj = max(max_proj, np.max(projs))
    
    # inradius r = side * sqrt(3)/2, so side = r / (sqrt(3)/2) = 2r/sqrt(3)
    side = 2.0 * max_proj / np.sqrt(3.0)
    return side

def initial_hexagonal_packing():
    """Create initial hexagonal packing for 11 hexagons: rows of 1,2,3,2,1 or 3,4,3 etc."""
    # Try a compact arrangement: center + 6 around + 4 more
    # Actually for 11, let's do a hex-like pattern
    # Row structure: 3, 4, 3 (offset) or similar
    # Spacing between hex centers when packed: 2*inradius = sqrt(3) for same orientation
    # For pointy-top hexagons, horizontal spacing = sqrt(3), vertical = 1.5
    
    positions = []
    angles = []
    
    # Central hexagon
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 surrounding (pointy-top orientation)
    # Neighbors at distance sqrt(3) in 6 directions
    d = np.sqrt(3.0)
    for i in range(6):
        a = i * np.pi / 3.0
        positions.append([d * np.cos(a), d * np.sin(a)])
        angles.append(0.0)
    
    # 4 more to make 11 - place them further out
    # Add at larger radius
    d2 = 2.0 * np.sqrt(3.0)
    for i in range(4):
        a = (2*i + 1) * np.pi / 6.0  # 30, 90, 150, 210 degrees
        positions.append([d2 * np.cos(a), d2 * np.sin(a)])
        angles.append(0.0)
    
    return np.array(positions), np.array(angles)

def construct():
    start_time = time.time()
    
    # Initial packing
    positions, angles = initial_hexagonal_packing()
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Compute initial outer side
    current_side = compute_outer_side(positions, angles, outer_center, outer_angle)
    best_side = current_side
    best_positions = positions.copy()
    best_angles = angles.copy()
    
    # Verify initial solution
    inner_verts_list = []
    for i in range(N):
        verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        inner_verts_list.append(verts)
    
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], current_side, outer_angle)
    
    if all_inside_hexagon(inner_verts_list, outer_verts) and are_disjoint(inner_verts_list):
        pass
    else:
        # Fallback: spread out more
        for i in range(N):
            a = i * 2 * np.pi / N
            r = 3.0
            positions[i, 0] = r * np.cos(a)
            positions[i, 1] = r * np.sin(a)
        current_side = compute_outer_side(positions, angles, outer_center, outer_angle)
        best_side = current_side
        best_positions = positions.copy()
        best_angles = angles.copy()
    
    # Optimization: minimize outer side by adjusting positions and angles
    # Use a penalty-based approach with scipy minimize
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            return best_side + 1000  # Signal timeout
        
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:]
        
        # Compute outer side
        side = compute_outer_side(pos, ang, outer_center, outer_angle)
        
        # Check constraints with penalty
        inner_verts_list = []
        for i in range(N):
            verts = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
            inner_verts_list.append(verts)
        
        outer_verts = hexagon_vertices(outer_center[0], outer_center[1], side, outer_angle)
        
        penalty = 0.0
        # Inside constraint
        if not all_inside_hexagon(inner_verts_list, outer_verts):
            penalty += 10.0 * side
        # Disjointness constraint
        if not are_disjoint(inner_verts_list):
            penalty += 10.0 * side
        
        return side + penalty
    
    # Try optimization
    x0 = np.concatenate([positions.flatten(), angles])
    
    try:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 30:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < best_side:
                new_pos = result.x[:2*N].reshape(N, 2)
                new_ang = result.x[2*N:]
                new_side = compute_outer_side(new_pos, new_ang, outer_center, outer_angle)
                
                # Verify
                inner_verts_list = []
                for i in range(N):
                    verts = hexagon_vertices(new_pos[i, 0], new_pos[i, 1], 1.0, new_ang[i])
                    inner_verts_list.append(verts)
                outer_verts = hexagon_vertices(outer_center[0], outer_center[1], new_side, outer_angle)
                
                if all_inside_hexagon(inner_verts_list, outer_verts) and are_disjoint(inner_verts_list):
                    best_side = new_side
                    best_positions = new_pos
                    best_angles = new_ang
    except Exception:
        pass
    
    # Try different outer orientations
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        for test_angle in [0, 30, 60, 90, 120, 150]:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
            oa = float(test_angle)
            side_test = compute_outer_side(best_positions, best_angles, outer_center, oa)
            if side_test < best_side:
                # Verify
                inner_verts_list = []
                for i in range(N):
                    verts = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
                    inner_verts_list.append(verts)
                outer_verts = hexagon_vertices(outer_center[0], outer_center[1], side_test, oa)
                if all_inside_hexagon(inner_verts_list, outer_verts) and are_disjoint(inner_verts_list):
                    best_side = side_test
                    outer_angle = oa
    
    # Final refinement with tighter optimization
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15:
        try:
            x0 = np.concatenate([best_positions.flatten(), best_angles])
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_side:
                new_pos = result.x[:2*N].reshape(N, 2)
                new_ang = result.x[2*N:]
                new_side = compute_outer_side(new_pos, new_ang, outer_center, outer_angle)
                
                inner_verts_list = []
                for i in range(N):
                    verts = hexagon_vertices(new_pos[i, 0], new_pos[i, 1], 1.0, new_ang[i])
                    inner_verts_list.append(verts)
                outer_verts = hexagon_vertices(outer_center[0], outer_center[1], new_side, outer_angle)
                
                if all_inside_hexagon(inner_verts_list, outer_verts) and are_disjoint(inner_verts_list):
                    best_side = new_side
                    best_positions = new_pos
                    best_angles = new_ang
        except Exception:
            pass
    
    # Build result
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    return inner, (outer_center[0], outer_center[1]), float(best_side), float(outer_angle)

