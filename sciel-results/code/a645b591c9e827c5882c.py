"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    def get_hex_vertices(center, angle_deg, side=1.0):
        """Returns list of (x, y) vertices for a regular hexagon."""
        cx, cy = center
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + math.radians(60 * k)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, center, angle_deg, side=1.0):
        """Check if point (px, py) is inside the hexagon defined by center, angle, side."""
        cx, cy = center
        dx = px - cx
        dy = py - cy
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(sin_a)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        inradius = side * sqrt3 / 2
        
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = lx * nx + ly * ny
            if proj > inradius + 1e-9:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects p3-p4 (proper intersection only)."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        
        return False

    def hexagons_overlap(c1, a1, c2, a2):
        """
        Check if two hexagons overlap (interiors intersect).
        Returns True if they overlap.
        """
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.hypot(dx, dy)
        if dist > 2.0 + 1e-9:
            return False
        
        v1 = get_hex_vertices(c1, a1)
        v2 = get_hex_vertices(c2, a2)
        
        # Check if any vertex of v1 is inside v2
        for p in v1:
            if point_in_hexagon(p[0], p[1], c2, a2):
                return True
                
        # Check if any vertex of v2 is inside v1
        for p in v2:
            if point_in_hexagon(p[0], p[1], c1, a1):
                return True
                
        # Check if any edges intersect
        for i in range(6):
            p1a = v1[i]
            p1b = v1[(i + 1) % 6]
            for j in range(6):
                p2a = v2[j]
                p2b = v2[(j + 1) % 6]
                if segments_intersect(p1a, p1b, p2a, p2b):
                    return True
        return False

    def boundary_violation(c, a, s_outer):
        """
        Check if hexagon at c, angle a is inside outer hexagon with side s_outer, center (0,0), rotation 0.
        Returns max violation amount (0 if inside).
        """
        verts = get_hex_vertices(c, a)
        max_viol = 0.0
        inradius_outer = s_outer * sqrt3 / 2
        
        for p in verts:
            px, py = p
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = px * nx + py * ny
                if proj > inradius_outer:
                    viol = proj - inradius_outer
                    if viol > max_viol:
                        max_viol = viol
        return max_viol

    def min_separation(c1, a1, c2, a2):
        """
        Compute the minimum separation between two hexagons.
        Positive means separated, negative means overlapping.
        Uses support function approach.
        """
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.hypot(dx, dy)
        if dist > 2.0 + 1e-9:
            return dist - 2.0
        
        v1 = get_hex_vertices(c1, a1)
        v2 = get_hex_vertices(c2, a2)
        
        # Check if any vertex of v1 is inside v2
        for p in v1:
            if point_in_hexagon(p[0], p[1], c2, a2):
                return -1.0
                
        # Check if any vertex of v2 is inside v1
        for p in v2:
            if point_in_hexagon(p[0], p[1], c1, a1):
                return -1.0
                
        # Check if any edges intersect
        for i in range(6):
            p1a = v1[i]
            p1b = v1[(i + 1) % 6]
            for j in range(6):
                p2a = v2[j]
                p2b = v2[(j + 1) % 6]
                if segments_intersect(p1a, p1b, p2a, p2b):
                    return -1.0
        
        # Compute minimum distance between edges
        min_dist = float('inf')
        for i in range(6):
            p1a = v1[i]
            p1b = v1[(i + 1) % 6]
            for j in range(6):
                p2a = v2[j]
                p2b = v2[(j + 1) % 6]
                # Distance between two segments
                # Simplified: check vertex-to-segment distances
                for p in [p1a, p1b]:
                    # Distance from point p to segment p2a-p2b
                    ax, ay = p2a
                    bx, by = p2b
                    px, py = p
                    vx, vy = bx - ax, by - ay
                    wx, wy = px - ax, py - ay
                    vv = vx * vx + vy * vy
                    if vv < 1e-12:
                        d = math.hypot(px - ax, py - ay)
                    else:
                        t = max(0, min(1, (wx * vx + wy * vy) / vv))
                        cx_seg = ax + t * vx
                        cy_seg = ay + t * vy
                        d = math.hypot(px - cx_seg, py - cy_seg)
                    if d < min_dist:
                        min_dist = d
                for p in [p2a, p2b]:
                    ax, ay = p1a
                    bx, by = p1b
                    px, py = p
                    vx, vy = bx - ax, by - ay
                    wx, wy = px - ax, py - ay
                    vv = vx * vx + vy * vy
                    if vv < 1e-12:
                        d = math.hypot(px - ax, py - ay)
                    else:
                        t = max(0, min(1, (wx * vx + wy * vy) / vv))
                        cx_seg = ax + t * vx
                        cy_seg = ay + t * vy
                        d = math.hypot(px - cx_seg, py - cy_seg)
                    if d < min_dist:
                        min_dist = d
        
        return min_dist

    def objective(params):
        # params: 2*N positions + N angles + 1 side
        pos = []
        for i in range(N):
            pos.append([params[2*i], params[2*i+1]])
        angs = [params[2*N + i] for i in range(N)]
        s = params[3*N]
        
        if s < 2.0:
            return 1e6
            
        penalty = 0.0
        
        # Check overlaps with smooth penalty
        for i in range(N):
            for j in range(i+1, N):
                sep = min_separation(pos[i], angs[i], pos[j], angs[j])
                if sep < 0:
                    penalty += 100.0 * sep * sep
                elif sep < 0.1:
                    # Small positive separation, encourage more
                    penalty += 1.0 * (0.1 - sep) ** 2
        
        # Check boundary
        for i in range(N):
            viol = boundary_violation(pos[i], angs[i], s)
            if viol > 0:
                penalty += 1000.0 * viol * viol
        
        return s + penalty

    # Generate diverse initial configurations
    configs = []
    
    # Config 1: 3-4-4 grid
    rows = [3, 4, 4]
    y_offset = 0.0
    pos_init = []
    ang_init = []
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            pos_init.append([x, y])
            ang_init.append(0.0)
        y_offset += 1.5
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))
    
    # Config 2: 4-3-4 grid
    rows = [4, 3, 4]
    y_offset = 0.0
    pos_init = []
    ang_init = []
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            pos_init.append([x, y])
            ang_init.append(0.0)
        y_offset += 1.5
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))
    
    # Config 3: 4-4-3 grid
    rows = [4, 4, 3]
    y_offset = 0.0
    pos_init = []
    ang_init = []
    for i, count in enumerate(rows):
        x_start = -(count - 1) * sqrt3 / 2
        for j in range(count):
            x = x_start + j * sqrt3
            y = y_offset
            pos_init.append([x, y])
            ang_init.append(0.0)
        y_offset += 1.5
    xs = [p[0] for p in pos_init]
    ys = [p[1] for p in pos_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pos_init:
        p[0] -= cx
        p[1] -= cy
    configs.append((pos_init, ang_init))
    
    # Config 4: Random with some structure
    np.random.seed(42)
    pos_init = []
    ang_init = []
    for i in range(N):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        a = np.random.choice([0, 30, 60, 90, 120, 150])
        pos_init.append([x, y])
        ang_init.append(a)
    configs.append((pos_init, ang_init))
    
    # Config 5: Another random
    np.random.seed(123)
    pos_init = []
    ang_init = []
    for i in range(N):
        x = np.random.uniform(-2.5, 2.5)
        y = np.random.uniform(-2.5, 2.5)
        a = np.random.uniform(0, 180)
        pos_init.append([x, y])
        ang_init.append(a)
    configs.append((pos_init, ang_init))
    
    best_solution = None
    best_score = float('inf')
    
    bounds = []
    for i in range(N):
        bounds.append((-6, 6))
        bounds.append((-6, 6))
    for i in range(N):
        bounds.append((-180, 180))
    bounds.append((2.5, 12.0))
    
    for idx, (p_init, a_init) in enumerate(configs):
        x0 = []
        for i in range(N):
            x0.append(p_init[i][0])
            x0.append(p_init[i][1])
        for i in range(N):
            x0.append(a_init[i])
        
        max_dist = 0
        for p in p_init:
            max_dist = max(max_dist, math.hypot(p[0], p[1]))
        x0.append(max_dist + 1.5)
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=x0,
                maxiter=150,
                popsize=25,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.7,
                seed=100 + idx,
                polish=False
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x.copy()
        except Exception as e:
            pass
    
    if best_solution is None:
        return [[0, 0, 0] for _ in range(N)], (0, 0), 10.0, 0.0
    
    # Extract solution
    pos = best_solution[:2*N].reshape(N, 2)
    angs = best_solution[2*N:3*N]
    s = best_solution[3*N]
    
    # Verify and refine: try to reduce s slightly
    valid = True
    for i in range(N):
        viol = boundary_violation(pos[i], angs[i], s)
        if viol > 1e-6:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                sep = min_separation(pos[i], angs[i], pos[j], angs[j])
                if sep < -1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        return [[0, 0, 0] for _ in range(N)], (0, 0), 10.0, 0.0
    
    # Try to tighten the outer hexagon by reducing s
    for _ in range(20):
        s_new = s * 0.995
        valid_new = True
        for i in range(N):
            viol = boundary_violation(pos[i], angs[i], s_new)
            if viol > 1e-6:
                valid_new = False
                break
        if valid_new:
            s = s_new
        else:
            break
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i][0]), float(pos[i][1]), float(angs[i])])
    
    return inner, (0.0, 0.0), float(s), 0.0
# EVOLVE-BLOCK-END
