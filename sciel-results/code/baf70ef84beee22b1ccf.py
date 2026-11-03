import math
import numpy as np
import time
import random
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115.0
N = 11
APOTHEM = math.sqrt(3) / 2.0

def enclosing_hexagon_size(points, outer_angle):
    """
    Compute the minimum side length of a regular hexagon (rotated by outer_angle)
    that encloses all points. The hexagon center is optimized to minimize size.
    
    A regular hexagon with side s and rotation theta has 6 sides. The distance
    from center to each side is s * APOTHEM. For the hexagon to contain all points,
    each point must be within distance s*APOTHEM from each side (on the interior side).
    
    We need to find the minimum s and center (cx, cy) such that all points are inside.
    """
    n = len(points)
    if n == 0:
        return 1.0, 0.0, 0.0
    
    # The 6 normal directions of the hexagon sides
    normals = []
    for k in range(6):
        ang = outer_angle + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    # For a hexagon centered at (cx, cy) with side s:
    # For each normal direction (nx, ny), we need:
    #   nx*(px - cx) + ny*(py - cy) <= s * APOTHEM  for all points
    #   -(nx*(px - cx) - ny*(py - cy)) <= s * APOTHEM  (opposite side)
    
    # This is a linear programming problem: minimize s
    # subject to: for each point i and each normal k:
    #   nx_k * (px_i - cx) + ny_k * (py_i - cy) <= s * APOTHEM
    #   -nx_k * (px_i - cx) - ny_k * (py_i - cy) <= s * APOTHEM
    
    # Variables: cx, cy, s
    # Minimize s
    # Constraints: for each (i, k):
    #   nx_k*px_i - nx_k*cx + ny_k*py_i - ny_k*cy - APOTHEM*s <= 0
    #   -nx_k*px_i + nx_k*cx - ny_k*py_i + ny_k*cy - APOTHEM*s <= 0
    
    from scipy.optimize import linprog
    
    c = np.array([0.0, 0.0, 1.0])  # minimize s
    
    A_ub = []
    b_ub = []
    
    for px, py in points:
        for nx, ny in normals:
            # nx*(px-cx) + ny*(py-cy) <= APOTHEM * s
            # => -nx*cx - ny*cy - APOTHEM*s <= -nx*px - ny*py
            A_ub.append([-nx, -ny, -APOTHEM])
            b_ub.append(-nx*px - ny*py)
            
            # -nx*(px-cx) - ny*(py-cy) <= APOTHEM * s
            # => nx*cx + ny*cy - APOTHEM*s <= nx*px + ny*py
            A_ub.append([nx, ny, -APOTHEM])
            b_ub.append(nx*px + ny*py)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(None, None), (None, None), (0.5, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            return res.x[2], res.x[0], res.x[1]
        else:
            # Fallback: compute bounding box and estimate
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            return max(w, h) / APOTHEM + 0.5, 0.0, 0.0
    except Exception:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        return max(w, h) / APOTHEM + 0.5, 0.0, 0.0


def hexagon_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = math.radians(angle_deg)
    vertices = []
    for k in range(6):
        a = angle + k * math.pi / 3.0
        vx = cx + side * math.cos(a)
        vy = cy + side * math.sin(a)
        vertices.append((vx, vy))
    return vertices


def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside the hexagon."""
    angle = math.radians(angle_deg)
    for k in range(6):
        a = angle + k * math.pi / 3.0
        nx = math.cos(a)
        ny = math.sin(a)
        dist = (px - cx) * nx + (py - cy) * ny
        if dist > side * APOTHEM + 1e-9:
            return False
    return True


def hexagon_hexagon_overlap_dist(p1, a1, p2, a2):
    """
    Compute the minimum translation distance needed to separate two unit hexagons
    at positions p1, p2 with rotations a1, a2.
    Returns 0 if they don't overlap, positive if they do.
    """
    # Get vertices of both hexagons
    v1 = hexagon_vertices(p1[0], p1[1], 1.0, a1)
    v2 = hexagon_vertices(p2[0], p2[1], 1.0, a2)
    
    # Check if any vertex of v1 is inside v2 or vice versa
    # If not, check edge-edge intersections
    
    def point_in_convex_polygon(px, py, vertices):
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i+1) % n]
            # Cross product to check if point is to the left of edge
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross < -1e-10:
                return False
        return True
    
    # Check if any vertex of v1 is inside v2
    inside = False
    for vx, vy in v1:
        if point_in_convex_polygon(vx, vy, v2):
            inside = True
            break
    
    if not inside:
        for vx, vy in v2:
            if point_in_convex_polygon(vx, vy, v1):
                inside = True
                break
    
    if not inside:
        return 0.0  # No overlap
    
    # If overlapping, estimate the overlap distance
    # Use the distance between centers as a rough measure
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist = math.sqrt(dx*dx + dy*dy)
    
    # For two unit hexagons, max distance between centers when touching is about 2*APOTHEM*2 = 2*sqrt(3)
    # But this depends on orientation. Use a conservative estimate.
    max_center_dist = 2.0 * math.sqrt(3) + 0.1
    if dist >= max_center_dist:
        return 0.0
    return max(0.0, max_center_dist - dist)


def total_overlap_penalty(params, outer_angle):
    """
    Compute total overlap penalty for all pairs of hexagons.
    params: [x1, y1, a1, x2, y2, a2, ...]
    """
    penalty = 0.0
    for i in range(N):
        x1, y1, a1 = params[3*i], params[3*i+1], params[3*i+2]
        for j in range(i+1, N):
            x2, y2, a2 = params[3*j], params[3*j+1], params[3*j+2]
            overlap = hexagon_hexagon_overlap_dist((x1, y1), a1, (x2, y2), a2)
            penalty += overlap * overlap
    return penalty


def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback: return a valid but large configuration
        inner = []
        for i in range(N):
            inner.append([0.0, 0.0, 0.0])
        return inner, [0.0, 0.0], 5.0, 0.0
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = float('inf')
    best_params = None
    
    # Try multiple random restarts
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Initialize with a rough lattice pattern, then add noise
        # Start with a compact arrangement
        base_positions = []
        # Try different initial patterns
        pattern = restart % 3
        
        if pattern == 0:
            # Hexagonal lattice
            for i in range(N):
                q = i % 4
                r = i // 4
                x = math.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                base_positions.append((x, y))
        elif pattern == 1:
            # Square-ish grid
            for i in range(N):
                q = i % 3
                r = i // 3
                x = 2.0 * q
                y = 2.0 * r
                base_positions.append((x, y))
        else:
            # Random compact
            for i in range(N):
                x = random.uniform(-3, 3)
                y = random.uniform(-3, 3)
                base_positions.append((x, y))
        
        # Add random angles
        params = []
        for i in range(N):
            x = base_positions[i][0] + random.uniform(-0.5, 0.5)
            y = base_positions[i][1] + random.uniform(-0.5, 0.5)
            a = random.uniform(0, 60)
            params.extend([x, y, a])
        
        params = np.array(params)
        
        # Simulated annealing
        T = 1.0
        T_min = 0.001
        cooling = 0.95
        max_iter = 5000
        
        current_score = float('inf')
        current_params = params.copy()
        
        for it in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            # Generate neighbor
            new_params = current_params.copy()
            idx = random.randint(0, len(new_params)-1)
            if idx % 3 == 2:  # angle
                new_params[idx] += random.gauss(0, 5)
                new_params[idx] = new_params[idx] % 60
            else:  # position
                new_params[idx] += random.gauss(0, 0.3)
            
            # Compute score
            points = []
            for i in range(N):
                points.append((new_params[3*i], new_params[3*i+1]))
            
            # Try a few outer angles
            best_outer_score = float('inf')
            best_outer_angle = 0.0
            for oa in [0, 15, 30, 45]:
                size, _, _ = enclosing_hexagon_size(points, math.radians(oa))
                overlap = total_overlap_penalty(new_params, oa)
                score = size + 100.0 * overlap
                if score < best_outer_score:
                    best_outer_score = score
                    best_outer_angle = oa
            
            new_score = best_outer_score
            
            if new_score < current_score:
                current_params = new_params
                current_score = new_score
            else:
                # Accept with probability exp(-delta/T)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_params = new_params
                    current_score = new_score
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Final evaluation
        points = []
        for i in range(N):
            points.append((current_params[3*i], current_params[3*i+1]))
        
        best_final_score = float('inf')
        best_final_angle = 0.0
        for oa in range(0, 60, 5):
            size, cx, cy = enclosing_hexagon_size(points, math.radians(oa))
            overlap = total_overlap_penalty(current_params, oa)
            score = size + 100.0 * overlap
            if score < best_final_score:
                best_final_score = score
                best_final_angle = oa
        
        if best_final_score < best_score:
            best_score = best_final_score
            best_params = current_params.copy()
            best_outer_angle = best_final_angle
    
    if best_params is None:
        # Fallback
        inner = []
        for i in range(N):
            inner.append([0.0, 0.0, 0.0])
        return inner, [0.0, 0.0], 5.0, 0.0
    
    # Extract final configuration
    points = []
    angles = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        points.append((x, y))
        angles.append(a)
    
    # Get the outer hexagon
    size, cx, cy = enclosing_hexagon_size(points, math.radians(best_outer_angle))
    
    # Verify no overlaps and all inside
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if hexagon_hexagon_overlap_dist(points[i], angles[i], points[j], angles[j]) > 0.01:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Add small perturbation to fix overlaps
        for i in range(N):
            for j in range(i+1, N):
                overlap = hexagon_hexagon_overlap_dist(points[i], angles[i], points[j], angles[j])
                if overlap > 0:
                    # Push apart
                    dx = points[j][0] - points[i][0]
                    dy = points[j][1] - points[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    shift = overlap / 2.0 + 0.01
                    points[i] = (points[i][0] - dx*shift, points[i][1] - dy*shift)
                    points[j] = (points[j][0] + dx*shift, points[j][1] + dy*shift)
        # Recompute outer hexagon
        size, cx, cy = enclosing_hexagon_size(points, math.radians(best_outer_angle))
    
    # Build output
    inner = []
    for i in range(N):
        inner.append([float(points[i][0]), float(points[i][1]), float(angles[i])])
    
    center = [float(cx), float(cy)]
    side = float(size)
    rotation = float(best_outer_angle)
    
    return inner, center, side, rotation
