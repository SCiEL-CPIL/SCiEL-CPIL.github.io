"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """Use LP to find optimal center and inradius for given outer hexagon angle.
    Returns (cx, cy, side_length)."""
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    
    # Variables: cx, cy, t (inradius)
    # Minimize t
    c = [0.0, 0.0, 1.0]
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        # For each vertex: (vx - cx)*nx + (vy - cy)*ny <= t
        # => -nx*cx - ny*cy - t <= -nx*vx - ny*vy
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
        
        # For each vertex: -(vx - cx)*nx - (vy - cy)*ny <= t
        # => nx*cx + ny*cy - t <= nx*vx + ny*vy
        for vx, vy in all_vertices:
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    
    # t >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except:
        return None

def _find_best_outer_angle(inner_hexes, time_limit=10.0):
    """Find the best outer angle and corresponding center/side for given inner hexagons.
    Returns (cx, cy, side, angle_deg)."""
    start_time = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse search: 0 to 30 degrees (symmetry)
    coarse_angles = np.arange(0, 30.01, 0.5)
    for angle_deg in coarse_angles:
        if time.time() > start_time + time_limit:
            break
        res = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Fine search around the best coarse angle
    if best_angle > 0:
        fine_start = max(0.0, best_angle - 0.5)
        fine_end = min(30.0, best_angle + 0.5)
        fine_angles = np.arange(fine_start, fine_end + 0.01, 0.01)
        for angle_deg in fine_angles:
            if time.time() > start_time + time_limit:
                break
            res = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)
    
    return best_center[0], best_center[1], best_side, best_angle

def _check_disjoint(inner_hexes, tol=1e-6):
    """Check if all inner hexagons are disjoint (no overlapping interiors).
    Returns True if valid, False otherwise."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner_hexes[i]
            x2, y2, a2 = inner_hexes[j]
            v1 = _get_hex_vertices(x1, y1, a1)
            v2 = _get_hex_vertices(x2, y2, a2)
            
            # Check if any vertex of hex1 is inside hex2 (or vice versa)
            # For convex polygons, if no vertex is inside and no edges cross, they are disjoint
            # Simplified check: if distance between centers is less than 2*inradius, they might overlap
            # But for rotated hexagons, we need a more robust check.
            # For now, assume lattice placement is valid. If we optimize positions, we need this.
            # A quick check: if the minimum distance between any two vertices is < tol, they might touch/overlap
            # Actually, the safest check for "disjoint interiors" is that the intersection is empty.
            # For unit hexagons on a triangular lattice with spacing sqrt(3), they touch at edges.
            # Let's just return True for the lattice case, and for optimized cases, we rely on the optimizer
            # not creating overlaps if we add constraints. But the current optimizer doesn't have overlap constraints.
            # So we must ensure the initial guess is good and the optimizer doesn't break it.
            # For a simple check: if centers are closer than 2, they definitely overlap.
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < 2.0 - tol:
                return False
    return True

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points in a hexagonal cluster arrangement
    # Central hexagon + ring of 6 + 4 more in second ring
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidate points in a spiral/grid around the origin
    # We want the 11 closest points to the origin in the triangular lattice
    # Lattice points are (i, j) -> i*u + j*v
    # Let's generate points with |i|, |j| <= 3 and sort by distance
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    # Sort by distance
    candidates.sort(key=lambda t: t[0])
    
    # Take the first N unique points
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    # Strategy: Try two main configurations for inner hexagon angles
    # 1. All aligned with lattice (30 deg)
    # 2. All aligned with lattice (0 deg)
    # And for each, optimize the outer hexagon angle.
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Configuration 1: Inner angle 30 deg
    if time.time() < _DEADLINE - 20.0:
        inner_30 = [[pos[0], pos[1], 30.0] for pos in positions]
        cx, cy, side, angle = _find_best_outer_angle(inner_30, time_limit=15.0)
        if side < best_side:
            best_side = side
            best_inner = inner_30
            best_center = (cx, cy)
            best_outer_angle = angle
    
    # Configuration 2: Inner angle 0 deg
    if time.time() < _DEADLINE - 20.0:
        inner_0 = [[pos[0], pos[1], 0.0] for pos in positions]
        cx, cy, side, angle = _find_best_outer_angle(inner_0, time_limit=15.0)
        if side < best_side:
            best_side = side
            best_inner = inner_0
            best_center = (cx, cy)
            best_outer_angle = angle
    
    # If time permits, try a local optimization of the inner positions
    # This is risky without overlap constraints, so we'll skip it for now
    # unless we have a very robust check.
    
    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
# EVOLVE-BLOCK-END
