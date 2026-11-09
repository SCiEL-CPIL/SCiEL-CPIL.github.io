"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon, center (cx,cy), side s, rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def _hexagon_side_needed(points, cx, cy, angle_deg):
    """Minimum side length of regular hexagon with given center and angle to contain all points."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    max_val = 0.0
    for p in points:
        dx, dy = p[0] - cx, p[1] - cy
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        for i in range(6):
            normal_angle = np.radians(30 + 60 * i)
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            val = lx * nx + ly * ny
            if val > max_val:
                max_val = val
    return 2.0 * max_val / np.sqrt(3)

def _all_points_inside_hex(points, cx, cy, s, angle_deg):
    """Check all points inside the regular hexagon."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    apothem = s * np.sqrt(3) / 2
    for p in points:
        dx, dy = p[0] - cx, p[1] - cy
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        for i in range(6):
            normal_angle = np.radians(30 + 60 * i)
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            if lx * nx + ly * ny > apothem + 1e-9:
                return False
    return True

def _optimize_enclosing_hex(points, cx0, cy0, angle0):
    """Optimize center and angle to minimize the required side length."""
    def objective(params):
        cx, cy, angle = params
        return _hexagon_side_needed(points, cx, cy, angle)
    
    best = None
    best_val = np.inf
    for _ in range(3):
        try:
            res = minimize(objective, [cx0, cy0, angle0], method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best = res.x
        except:
            pass
    if best is None:
        return cx0, cy0, angle0, _hexagon_side_needed(points, cx0, cy0, angle0)
    return best[0], best[1], best[2], best_val

def _generate_positions_pattern1():
    """Hex cluster: center + ring1(6) + 4 from ring2."""
    coords = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
              (2,0), (2,-1), (-2,0), (-2,1)]
    sqrt3 = np.sqrt(3)
    positions = []
    for q, r in coords:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        positions.append([x, y, 0.0])
    return positions

def _generate_positions_pattern2():
    """Different 11-hex arrangement."""
    coords = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
              (2,0), (2,-1), (2,-2), (-2,0)]
    sqrt3 = np.sqrt(3)
    positions = []
    for q, r in coords:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        positions.append([x, y, 0.0])
    return positions

def _check_no_overlap(positions, eps=1e-6):
    """Check that no two hexagons overlap (they can touch)."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, a1 = positions[i]
            x2, y2, a2 = positions[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            # For two unit hexagons, minimum center distance for no overlap is 2*apothem = sqrt(3)
            # Actually for flat-top hexagons side by side, center distance = 2*apothem = sqrt(3) ≈ 1.732
            # But this depends on orientation. For same orientation, min dist = sqrt(3)
            # For safety, check actual vertex overlap
            if dist < np.sqrt(3) - eps:
                # Potential overlap, do detailed check
                v1 = _hex_vertices(x1, y1, 1.0, a1)
                v2 = _hex_vertices(x2, y2, 1.0, a2)
                # Check if any vertex of v1 is inside v2 or vice versa
                # Simplified: if centers are too close, assume overlap
                return False
    return True

def construct():
    global _START_TIME
    if time.time() - _START_TIME > _DEADLINE:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_result = None
    best_side = np.inf
    
    patterns = [_generate_positions_pattern1(), _generate_positions_pattern2()]
    
    for positions in patterns:
        if not _check_no_overlap(positions):
            continue
        
        all_points = []
        for pos in positions:
            x, y, ang = pos
            verts = _hex_vertices(x, y, 1.0, ang)
            all_points.extend(verts.tolist())
        all_points = np.array(all_points)
        
        cx0 = np.mean(all_points[:, 0])
        cy0 = np.mean(all_points[:, 1])
        
        for angle0 in [0, 30, 60, 90, 120, 150]:
            cx, cy, ang, side = _optimize_enclosing_hex(all_points, cx0, cy0, angle0)
            if side < best_side:
                if _all_points_inside_hex(all_points, cx, cy, side, ang):
                    best_side = side
                    best_result = (positions, (cx, cy), side, ang)
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

