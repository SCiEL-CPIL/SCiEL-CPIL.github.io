"""Combined program: LP-based outer hexagon with continuous angle search + inner polish."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """Use LP to find optimal center for given outer hexagon angle.
    Returns (cx, cy, side_length) or None on failure."""
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    n_pts = len(all_vertices)
    
    # Variables: cx, cy, t (inradius)
    # Minimize t
    c = np.array([0.0, 0.0, 1.0])
    
    # Build constraint matrix vectorized
    # For each of 6 face normals (and their negations), for each vertex:
    #   (vx - cx)*nx + (vy - cy)*ny <= t  =>  -nx*cx - ny*cy - t <= -nx*vx - ny*vy
    #   -(vx - cx)*nx - (vy - cy)*ny <= t  =>  nx*cx + ny*cy - t <= nx*vx + ny*vy
    
    normals = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    verts_arr = np.array(all_vertices)  # (n_pts, 2)
    n_constraints = 12 * n_pts
    
    A_ub = np.zeros((n_constraints, 3))
    b_ub = np.zeros(n_constraints)
    
    row = 0
    for nx, ny in normals:
        # First set: -nx*cx - ny*cy - t <= -nx*vx - ny*vy
        A_ub[row:row+n_pts, 0] = -nx
        A_ub[row:row+n_pts, 1] = -ny
        A_ub[row:row+n_pts, 2] = -1.0
        b_ub[row:row+n_pts] = -(nx * verts_arr[:, 0] + ny * verts_arr[:, 1])
        row += n_pts
        
        # Second set: nx*cx + ny*cy - t <= nx*vx + ny*vy
        A_ub[row:row+n_pts, 0] = nx
        A_ub[row:row+n_pts, 1] = ny
        A_ub[row:row+n_pts, 2] = -1.0
        b_ub[row:row+n_pts] = nx * verts_arr[:, 0] + ny * verts_arr[:, 1]
        row += n_pts
    
    # t >= 0  =>  -t <= 0
    A_ub = np.vstack([A_ub, np.array([[0.0, 0.0, -1.0]])])
    b_ub = np.append(b_ub, 0.0)
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                        bounds=[(None, None), (None, None), (0, None)],
                        method='highs')
        if result.success:
            cx, cy, t = result.x
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side
    except:
        pass
    
    return None

def _min_enclosing_hex_side(inner_hexes, coarse_step=1.0, fine_step=0.01, fine_range=0.5):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Uses continuous angle search. Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse search: 0 to 30 degrees (due to 60-degree symmetry)
    for angle_deg in np.arange(0, 30.0 + coarse_step, coarse_step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = angle_deg
    
    # Fine search around best coarse angle
    if best_side < float('inf'):
        lo = max(0.0, best_angle - fine_range)
        hi = min(30.0, best_angle + fine_range)
        for angle_deg in np.arange(lo, hi + fine_step, fine_step):
            if time.time() > _DEADLINE - 10.0:
                break
            res = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = angle_deg
    
    return best_center[0], best_center[1], best_side, best_angle

def _check_disjoint(inner_hexes):
    """Check if all inner hexagons have disjoint interiors."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner_hexes[i]
            x2, y2, a2 = inner_hexes[j]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # Conservative check for overlapping
            if dist < math.sqrt(3) - 0.001:
                return False
    return True

def _nelder_mead_polish(inner_hexes, fixed_outer_angle=0.0, max_iter=200, tol=1e-6):
    """Polish inner hexagon positions using Nelder-Mead to minimize enclosing hexagon side.
    Uses a fixed outer angle for speed (single LP per objective eval)."""
    sqrt3 = math.sqrt(3)
    
    def objective(params):
        # params: flattened [x1, y1, a1, x2, y2, a2, ...]
        new_inner = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            new_inner.append([x, y, a])
        
        # Check disjointness - if violated, return large penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = new_inner[i][0] - new_inner[j][0]
                dy = new_inner[i][1] - new_inner[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < sqrt3 - 0.005:
                    return 1000.0 + (sqrt3 - dist) * 100
        
        # Use fixed outer angle for speed
        res = _min_enclosing_hex_side_lp(new_inner, fixed_outer_angle)
        if res is None:
            return 1000.0
        _, _, side = res
        return side
    
    # Initial params
    x0 = []
    for h in inner_hexes:
        x0.extend([h[0], h[1], h[2]])
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
        
        if result.success and result.fun < 1000.0:
            polished = []
            for i in range(N):
                polished.append([result.x[i*3], result.x[i*3+1], result.x[i*3+2]])
            return polished
    except:
        pass
    
    return inner_hexes

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points in a hexagonal cluster arrangement
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidate points in a spiral/grid around the origin
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
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try inner angles 0 and 30
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner = [[pos[0], pos[1], inner_angle] for pos in positions]
        
        # Compute initial minimum enclosing hexagon
        cx, cy, side, angle = _min_enclosing_hex_side(inner)
        
        if side < best_side:
            best_side = side
            best_inner = [h[:] for h in inner]
            best_center = (cx, cy)
            best_outer_angle = angle
    
    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Try Nelder-Mead polish if time permits (using fixed outer angle for speed)
    if time.time() < _DEADLINE - 40.0:
        polished = _nelder_mead_polish(best_inner, fixed_outer_angle=best_outer_angle,
                                       max_iter=200, tol=1e-6)
        if polished is not best_inner:
            # Re-evaluate with full angle search
            pcx, pcy, pside, pangle = _min_enclosing_hex_side(polished,
                                                              coarse_step=1.0, fine_step=0.01, fine_range=0.5)
            if pside < best_side - 1e-6:
                best_side = pside
                best_inner = polished
                best_center = (pcx, pcy)
                best_outer_angle = pangle
    
    # Final high-precision LP polish for outer hexagon
    if time.time() < _DEADLINE - 15.0:
        # Very tight angle search around the best angle
        fine_lo = max(0.0, best_outer_angle - 1.0)
        fine_hi = min(30.0, best_outer_angle + 1.0)
        for test_angle in np.arange(fine_lo, fine_hi + 0.001, 0.001):
            if time.time() > _DEADLINE - 15.0:
                break
            res = _min_enclosing_hex_side_lp(best_inner, test_angle)
            if res:
                tcx, tcy, tside = res
                if tside < best_side - 1e-10:
                    best_side = tside
                    best_center = (tcx, tcy)
                    best_outer_angle = test_angle
    
    return best_inner, (best_center[0], best_center[1]), best_side, best_outer_angle
