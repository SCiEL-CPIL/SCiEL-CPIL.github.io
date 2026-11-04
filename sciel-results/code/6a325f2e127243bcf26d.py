import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        pts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return pts

def polygon_intersection_area(poly1, poly2):
    """Compute area of intersection of two convex polygons using Sutherland-Hodgman."""
    def clip_polygon(subject, clip):
        def is_inside(pt, edge_start, edge_end):
            x1, y1 = edge_start
            x2, y2 = edge_end
            x, y = pt
            return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= -1e-10
        
        def line_intersect(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return p1
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        
        output = list(subject)
        for i in range(len(clip)):
            input_list = output
            output = []
            if not input_list:
                break
            edge_start = clip[i]
            edge_end = clip[(i + 1) % len(clip)]
            
            for j in range(len(input_list)):
                current = input_list[j]
                previous = input_list[j - 1]
                if is_inside(current, edge_start, edge_end):
                    if not is_inside(previous, edge_start, edge_end):
                        output.append(line_intersect(previous, current, edge_start, edge_end))
                    output.append(current)
                elif is_inside(previous, edge_start, edge_end):
                    output.append(line_intersect(previous, current, edge_start, edge_end))
        return output
    
    def polygon_area(pts):
        if len(pts) < 3:
            return 0.0
        area = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0
    
    clipped = clip_polygon(list(poly1), list(poly2))
    return polygon_area(clipped)

def hexagon_overlap_distance(c1, a1, c2, a2):
    """
    Check if two unit hexagons overlap. Returns 0 if no overlap, >0 if overlap.
    c1, c2 are centers as (x,y), a1, a2 are angles in degrees.
    """
    v1 = hex_vertices(c1[0], c1[1], a1)
    v2 = hex_vertices(c2[0], c2[1], a2)
    
    # Quick bounding box check
    min_x1 = min(p[0] for p in v1)
    max_x1 = max(p[0] for p in v1)
    min_y1 = min(p[1] for p in v1)
    max_y1 = max(p[1] for p in v1)
    min_x2 = min(p[0] for p in v2)
    max_x2 = max(p[0] for p in v2)
    min_y2 = min(p[1] for p in v2)
    max_y2 = max(p[1] for p in v2)
    
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return 0.0
    
    area = polygon_intersection_area(v1, v2)
    if area < 1e-6:
        return 0.0
    return area

def min_enclosing_hexagon_side_fast(pts, angle_deg=0.0):
    """
    Find the minimum side length S of a regular hexagon (with given rotation)
    that encloses all points. Returns (S, center_x, center_y).
    Uses linear programming via scipy on convex hull vertices only.
    """
    if len(pts) < 3:
        return 1.0, 0.0, 0.0
    
    # Compute convex hull using scipy if available, otherwise use all points
    try:
        from scipy.spatial import ConvexHull
        pts_arr = np.array(pts)
        hull = ConvexHull(pts_arr)
        hull_pts = pts_arr[hull.vertices]
    except:
        hull_pts = np.array(pts)
    
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        theta = a + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in hull_pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx, cy = res.x[0], res.x[1]
            S = (D + APOTHEM) / APOTHEM
            return S, cx, cy
        else:
            return float('inf'), 0.0, 0.0
    except Exception:
        return float('inf'), 0.0, 0.0

def min_enclosing_hexagon_side(pts, angle_deg=0.0):
    """Wrapper for backward compatibility."""
    return min_enclosing_hexagon_side_fast(pts, angle_deg)

def check_no_overlap(positions, angle_deg=0.0, tol=1e-4):
    """Check that no two hexagons overlap."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            c1 = (positions[i][0], positions[i][1])
            c2 = (positions[j][0], positions[j][1])
            overlap = hexagon_overlap_distance(c1, angle_deg, c2, angle_deg)
            if overlap > tol:
                return False
    return True

def force_directed_relaxation(initial_positions, angle_deg=0.0, max_iter=5000, dt=0.002, repulsion_strength=3.0):
    """
    Relax positions using force-directed method with hexagon-aware repulsion.
    Uses adaptive step size and cooling.
    """
    n = len(initial_positions)
    pos = np.array(initial_positions, dtype=float)
    current_dt = dt
    min_dt = 0.0001
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        forces = np.zeros_like(pos)
        max_overlap = 0.0
        total_overlap = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-10:
                    dx = (np.random.random() - 0.5) * 0.01
                    dy = (np.random.random() - 0.5) * 0.01
                    dist = math.sqrt(dx*dx + dy*dy)
                
                # Check actual overlap
                c1 = (pos[i, 0], pos[i, 1])
                c2 = (pos[j, 0], pos[j, 1])
                overlap = hexagon_overlap_distance(c1, angle_deg, c2, angle_deg)
                
                if overlap > 1e-6:
                    max_overlap = max(max_overlap, overlap)
                    total_overlap += overlap
                    # Strong repulsion when overlapping
                    force_mag = repulsion_strength * (1.0 + overlap) / (dist + 1e-10)
                elif dist < 1.9:
                    # Mild repulsion to maintain separation
                    force_mag = repulsion_strength * 0.05 * (1.9 - dist) / (dist + 1e-10)
                else:
                    force_mag = 0.0
                
                if force_mag > 0:
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Normalize forces to prevent explosion
        force_norm = np.linalg.norm(forces, axis=1)
        max_force = 5.0
        for i in range(n):
            if force_norm[i] > max_force:
                forces[i] *= max_force / force_norm[i]
        
        # Update positions
        pos += current_dt * forces
        current_dt = max(min_dt, current_dt * 0.999)
        
        # Stop if converged
        if max_overlap < 1e-6 and total_overlap < 1e-5:
            break
    
    return pos, max_overlap

def generate_initial_configurations():
    """Generate multiple initial configurations for the 11 hexagons."""
    configs = []
    sqrt3 = math.sqrt(3)
    
    # Config 1: Perfect hex grid - center + 6 neighbors + 4 more in second ring
    pts = [(0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0
        pts.append((sqrt3 * math.cos(theta), sqrt3 * math.sin(theta)))
    # Add 4 more from second ring at optimal positions
    for k in range(4):
        theta = math.pi/6 + k * math.pi / 3.0
        pts.append((2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)))
    configs.append(np.array(pts))
    
    # Config 2: Axial coordinate hex grid
    axial = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1), (2,0), (0,2)]
    pts = []
    for q, r in axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    configs.append(np.array(pts))
    
    # Config 3: Tighter cluster with slight rotation
    pts = [(0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0 + 0.05
        pts.append((1.72 * math.cos(theta), 1.72 * math.sin(theta)))
    for k in range(4):
        theta = math.pi/6 + k * math.pi / 3.0
        pts.append((2.9 * math.cos(theta), 2.9 * math.sin(theta)))
    configs.append(np.array(pts))
    
    # Config 4: Greedy densest packing - place each hex to maximize minimum distance
    np.random.seed(42)
    pts = [(0.0, 0.0)]
    for _ in range(N - 1):
        best_pos = None
        best_score = -1
        for _ in range(200):
            x = np.random.uniform(-3.5, 3.5)
            y = np.random.uniform(-3.5, 3.5)
            # Score: minimize max overlap, maximize min distance
            min_dist = min(math.sqrt((x-px)**2 + (y-py)**2) for px, py in pts)
            # Penalize if too close
            if min_dist < 1.5:
                score = -10 + min_dist
            else:
                score = min_dist
            if score > best_score:
                best_score = score
                best_pos = (x, y)
        if best_pos:
            pts.append(best_pos)
    configs.append(np.array(pts))
    
    # Config 5: Ring structure - center + 6 + 4 at different radii
    pts = [(0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0
        pts.append((1.73 * math.cos(theta), 1.73 * math.sin(theta)))
    for k in range(4):
        theta = math.pi/6 + k * math.pi / 3.0 + 0.1
        pts.append((2.85 * math.cos(theta), 2.85 * math.sin(theta)))
    configs.append(np.array(pts))
    
    # Config 6: Random restart with seed
    np.random.seed(123)
    pts = []
    for _ in range(N):
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)
        pts.append((x, y))
    configs.append(np.array(pts))
    
    return configs

def optimize_configuration(pos, angle_deg=0.0, max_iter=3000):
    """Optimize a configuration using force-directed relaxation and local search."""
    pos, _ = force_directed_relaxation(pos, angle_deg, max_iter=max_iter)
    
    # Local search: try small perturbations with adaptive step size
    n = len(pos)
    step_size = 0.05
    current_obj = float('inf')
    
    # Compute initial objective
    best_S = float('inf')
    for a in np.linspace(0, 60, 13):
        S, _, _ = min_enclosing_hexagon_side_fast(pos, a)
        if S < best_S:
            best_S = S
    current_obj = best_S
    
    no_improve_count = 0
    for it in range(50):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        improved = False
        for i in range(n):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = pos.copy()
                    new_pos[i, 0] += dx
                    new_pos[i, 1] += dy
                    
                    # Check if valid (no overlap)
                    if not check_no_overlap(new_pos, angle_deg, tol=1e-3):
                        continue
                    
                    # Compute objective
                    best_local_S = float('inf')
                    for a in np.linspace(0, 60, 7):
                        S, _, _ = min_enclosing_hexagon_side_fast(new_pos, a)
                        if S < best_local_S:
                            best_local_S = S
                    
                    if best_local_S < current_obj - 1e-6:
                        pos = new_pos
                        current_obj = best_local_S
                        improved = True
        
        if not improved:
            step_size *= 0.7
            no_improve_count += 1
            if no_improve_count > 3 or step_size < 0.003:
                break
        else:
            no_improve_count = 0
    
    return pos, current_obj

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    best_S = float('inf')
    best_positions = None
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    configs = generate_initial_configurations()
    
    for config_idx, pos in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        try:
            # Try multiple angles for the inner hexagons
            for angle_deg in [0, 15, 30, 45]:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                pos_opt, obj = optimize_configuration(pos.copy(), angle_deg, max_iter=2000)
                
                # Find best enclosing hexagon angle
                best_local_S = float('inf')
                best_local_angle = angle_deg
                best_local_center = (0.0, 0.0)
                for a in np.linspace(0, 60, 25):
                    S, cx, cy = min_enclosing_hexagon_side_fast(pos_opt, a)
                    if S < best_local_S:
                        best_local_S = S
                        best_local_angle = a
                        best_local_center = (cx, cy)
                
                if best_local_S < best_S:
                    best_S = best_local_S
                    best_positions = pos_opt
                    best_angle = best_local_angle
                    best_center = best_local_center
        except Exception:
            continue
    
    # Fallback
    if best_positions is None or best_S == float('inf'):
        best_positions = np.array([(0.0, 0.0)] + [(math.sqrt(3)*math.cos(k*math.pi/3), math.sqrt(3)*math.sin(k*math.pi/3)) for k in range(6)] + [(2*math.sqrt(3)*math.cos(k*math.pi/3), 2*math.sqrt(3)*math.sin(k*math.pi/3)) for k in range(4)])
        best_angle = 0.0
        S, cx, cy = min_enclosing_hexagon_side_fast(best_positions, best_angle)
        best_S = S
        best_center = (cx, cy)
    
    # Verify no overlap and adjust if needed
    if not check_no_overlap(best_positions, best_angle, tol=1e-3):
        # Scale down from center
        cx, cy = best_center
        for scale in [0.99, 0.98, 0.97, 0.95, 0.93]:
            pts_scaled = np.array([cx + (x - cx) * scale, cy + (y - cy) * scale] for x, y in best_positions)
            if check_no_overlap(pts_scaled, best_angle, tol=1e-4):
                best_positions = pts_scaled
                break
    
    # Final computation
    S_final, cx_final, cy_final = min_enclosing_hexagon_side_fast(best_positions, best_angle)
    
    inner = []
    for x, y in best_positions:
        inner.append([float(x), float(y), float(best_angle)])
    
    center = [float(cx_final), float(cy_final)]
    side = float(S_final)
    rotation = float(best_angle)
    
    return inner, center, side, rotation
