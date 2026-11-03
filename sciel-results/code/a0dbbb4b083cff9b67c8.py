import math
import time
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 3.0

def hex_vertices(cx, cy, angle_deg=0.0):
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        pts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return pts

def min_enclosing_hexagon_side_fast(pts, angle_deg=0.0):
    """
    Fast computation of minimal enclosing regular hexagon side length.
    For a hexagon with rotation angle, the 6 edge normals are fixed.
    The optimal center minimizes max_k (M_k - c·n_k) where M_k = max_p(p·n_k).
    """
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        theta = a + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Compute M_k = max over all points of p · n_k
    M = np.zeros(6)
    for k in range(6):
        nx, ny = normals[k]
        M[k] = max(px * nx + py * ny for px, py in pts)
    
    # Find optimal center: minimize max_k (M_k - c·n_k)
    # Due to hexagon symmetry, try center at origin first, then refine
    # The optimal c satisfies that the max is achieved by at least 2 opposing pairs
    
    # Try all pairs of normals to find candidate centers
    best_S = float('inf')
    best_c = (0.0, 0.0)
    
    # Candidate 1: centroid
    cx = sum(px for px, py in pts) / len(pts)
    cy = sum(py for px, py in pts) / len(pts)
    S_val = max(M[k] - (cx * normals[k][0] + cy * normals[k][1]) for k in range(6)) / APOTHEM
    if S_val < best_S:
        best_S = S_val
        best_c = (cx, cy)
    
    # Candidate 2: intersection of opposing normal constraints
    # For each pair (k, k+3), the constraint is c·n_k ≤ M_k - S*APOTHEM and c·n_{k+3} ≤ M_{k+3} - S*APOTHEM
    # Since n_{k+3} = -n_k, this gives: c·n_k ≤ M_k - S*APOTHEM and -c·n_k ≤ M_{k+3} - S*APOTHEM
    # So: c·n_k ∈ [M_{k+3} - S*APOTHEM, M_k - S*APOTHEM]
    # For the hexagon to be valid, we need M_k - M_{k+3} ≤ 2*S*APOTHEM for all k
    
    # Better approach: use the fact that optimal c makes at least 2 constraints tight
    # Try all pairs of non-opposing normals
    for i in range(6):
        for j in range(i+1, 6):
            if (j - i) % 6 == 3:
                continue  # Skip opposing pairs, they give a range not a point
            # Solve: c·n_i = M_i - S*APOTHEM and c·n_j = M_j - S*APOTHEM
            # This is a system: n_i·c = M_i - S*APOTHEM, n_j·c = M_j - S*APOTHEM
            # We can eliminate S: from first eq, S*APOTHEM = M_i - c·n_i
            # Substitute: c·n_j = M_j - M_i + c·n_i => c·(n_j - n_i) = M_j - M_i
            # So c = alpha * (n_j - n_i) where alpha is determined by perpendicular constraint
            # Actually, we have 2 equations in 3 unknowns (cx, cy, S). 
            # Let's parametrize: for a given c, S = max_k(M_k - c·n_k)/APOTHEM
            # We want to minimize this. Use gradient descent or try a grid.
            pass
    
    # Simple approach: try a small grid of centers around centroid
    for dcx in np.linspace(-1.0, 1.0, 11):
        for dcy in np.linspace(-1.0, 1.0, 11):
            cx_try = cx + dcx
            cy_try = cy + dcy
            S_val = max(M[k] - (cx_try * normals[k][0] + cy_try * normals[k][1]) for k in range(6)) / APOTHEM
            if S_val < best_S:
                best_S = S_val
                best_c = (cx_try, cy_try)
    
    # Refine with finer grid around best
    for dcx in np.linspace(-0.2, 0.2, 21):
        for dcy in np.linspace(-0.2, 0.2, 21):
            cx_try = best_c[0] + dcx
            cy_try = best_c[1] + dcy
            S_val = max(M[k] - (cx_try * normals[k][0] + cy_try * normals[k][1]) for k in range(6)) / APOTHEM
            if S_val < best_S:
                best_S = S_val
                best_c = (cx_try, cy_try)
    
    return best_S, best_c[0], best_c[1]

def polygon_overlap_area(v1, v2):
    """Compute overlap area between two convex polygons using Sutherland-Hodgman clipping."""
    if not v1 or not v2:
        return 0.0
    
    def clip_polygon(subject, clip):
        output = list(subject)
        for i in range(len(clip)):
            if not output:
                break
            input_list = output
            output = []
            start = clip[i]
            end = clip[(i + 1) % len(clip)]
            for j in range(len(input_list)):
                current = input_list[j]
                prev = input_list[(j - 1) % len(input_list)]
                if (current[0] - start[0]) * (end[1] - start[1]) - (current[1] - start[1]) * (end[0] - start[0]) >= 0:
                    if (prev[0] - start[0]) * (end[1] - start[1]) - (prev[1] - start[1]) * (end[0] - start[0]) < 0:
                        # Intersection
                        d1 = (prev[0] - start[0]) * (end[1] - start[1]) - (prev[1] - start[1]) * (end[0] - start[0])
                        d2 = (current[0] - start[0]) * (end[1] - start[1]) - (current[1] - start[1]) * (end[0] - start[0])
                        t = d1 / (d1 - d2)
                        ix = prev[0] + t * (current[0] - prev[0])
                        iy = prev[1] + t * (current[1] - prev[1])
                        output.append((ix, iy))
                    output.append(current)
        return output
    
    def polygon_area(poly):
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0
    
    clipped = clip_polygon(v1, v2)
    if not clipped:
        return 0.0
    return polygon_area(clipped)

def check_overlap(pos1, angle1, pos2, angle2):
    """Check if two unit hexagons overlap. Returns overlap area."""
    v1 = hex_vertices(pos1[0], pos1[1], angle1)
    v2 = hex_vertices(pos2[0], pos2[1], angle2)
    return polygon_overlap_area(v1, v2)

def force_directed_relaxation_v2(initial_positions, initial_angles, max_iter=3000, dt=0.02):
    """
    Force-directed relaxation with proper overlap detection.
    """
    n = len(initial_positions)
    pos = np.array(initial_positions, dtype=float)
    angles = np.array(initial_angles, dtype=float)
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        forces = np.zeros_like(pos)
        total_overlap = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                overlap = check_overlap((pos[i, 0], pos[i, 1]), angles[i], 
                                        (pos[j, 0], pos[j, 1]), angles[j])
                if overlap > 1e-6:
                    total_overlap += overlap
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-10:
                        dx = (np.random.random() - 0.5) * 0.01
                        dy = (np.random.random() - 0.5) * 0.01
                        dist = math.sqrt(dx*dx + dy*dy)
                    # Push apart proportional to overlap
                    force_mag = 5.0 * overlap / (dist + 1e-10)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        pos += dt * forces
        dt *= 0.998
        
        if total_overlap < 1e-8:
            break
    
    return pos, angles

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Strategy 1: Hex grid pattern (7 in center ring, 4 in outer)
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # 7 hexagons: center + 6 neighbors
            axial = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1)]
            # Add 4 more at distance 2
            axial.extend([(2,0), (0,2), (2,-2), (-1,2)])
            
            pts = []
            for q, r in axial[:N]:
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                pts.append((x, y))
            
            angles = [0.0] * N
            pos, angles = force_directed_relaxation_v2(pts, angles, max_iter=2000)
            
            for angle_deg in np.linspace(0, 60, 31):
                S, cx, cy = min_enclosing_hexagon_side_fast(pos, angle_deg)
                if S < best_S:
                    best_S = S
                    best_positions = pos.copy()
                    best_angles = angles.copy()
                    best_angle = angle_deg
                    best_center = (cx, cy)
    except Exception:
        pass
    
    # Strategy 2: Ring + center with rotation
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            pts = [(0.0, 0.0)]
            for k in range(6):
                theta = k * math.pi / 3.0
                pts.append((math.sqrt(3) * math.cos(theta), math.sqrt(3) * math.sin(theta)))
            for k in range(4):
                theta = math.pi/6 + k * math.pi / 3.0
                pts.append((2 * math.sqrt(3) * math.cos(theta), 2 * math.sqrt(3) * math.sin(theta)))
            
            angles = [0.0] * N
            pos, angles = force_directed_relaxation_v2(pts, angles, max_iter=2000)
            
            for angle_deg in np.linspace(0, 60, 31):
                S, cx, cy = min_enclosing_hexagon_side_fast(pos, angle_deg)
                if S < best_S:
                    best_S = S
                    best_positions = pos.copy()
                    best_angles = angles.copy()
                    best_angle = angle_deg
                    best_center = (cx, cy)
    except Exception:
        pass
    
    # Strategy 3: 3x3 grid minus corners
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            axial = [(q, r) for q in range(-1, 2) for r in range(-1, 2)]
            # That's 9, add 2 more
            axial.extend([(2, 0), (-2, 0)])
            
            pts = []
            for q, r in axial[:N]:
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                pts.append((x, y))
            
            angles = [0.0] * N
            pos, angles = force_directed_relaxation_v2(pts, angles, max_iter=2000)
            
            for angle_deg in np.linspace(0, 60, 31):
                S, cx, cy = min_enclosing_hexagon_side_fast(pos, angle_deg)
                if S < best_S:
                    best_S = S
                    best_positions = pos.copy()
                    best_angles = angles.copy()
                    best_angle = angle_deg
                    best_center = (cx, cy)
    except Exception:
        pass
    
    # Final: verify no overlap and compute final S
    if best_positions is not None:
        try:
            # Verify no overlap
            has_overlap = False
            for i in range(N):
                for j in range(i+1, N):
                    ov = check_overlap((best_positions[i,0], best_positions[i,1]), best_angles[i],
                                       (best_positions[j,0], best_positions[j,1]), best_angles[j])
                    if ov > 1e-6:
                        has_overlap = True
                        break
                if has_overlap:
                    break
            
            if has_overlap:
                # Relax more
                pos, angles = force_directed_relaxation_v2(best_positions.tolist(), best_angles.tolist(), max_iter=5000)
                for angle_deg in np.linspace(0, 60, 51):
                    S, cx, cy = min_enclosing_hexagon_side_fast(pos, angle_deg)
                    if S < best_S:
                        best_S = S
                        best_positions = pos.copy()
                        best_angles = angles.copy()
                        best_angle = angle_deg
                        best_center = (cx, cy)
            
            S_final, cx_final, cy_final = min_enclosing_hexagon_side_fast(best_positions, best_angle)
        except Exception:
            S_final = best_S
            cx_final, cy_final = best_center
    
    # Fallback
    if best_positions is None or best_S == float('inf'):
        best_positions = np.array([(0.0, 0.0)] + [(math.sqrt(3)*math.cos(k*math.pi/3), math.sqrt(3)*math.sin(k*math.pi/3)) for k in range(6)] + [(2*math.sqrt(3)*math.cos(k*math.pi/3), 2*math.sqrt(3)*math.sin(k*math.pi/3)) for k in range(4)])
        best_angles = [0.0] * N
        best_angle = 0.0
        S_final, cx_final, cy_final = min_enclosing_hexagon_side_fast(best_positions, best_angle)
    
    inner = []
    for i in range(len(best_positions)):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    center = [float(cx_final), float(cy_final)]
    side = float(S_final)
    rotation = float(best_angle)
    
    return inner, center, side, rotation
