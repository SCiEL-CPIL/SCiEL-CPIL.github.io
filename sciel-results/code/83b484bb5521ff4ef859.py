import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _solve_lp(verts, outer_angle_deg):
    """Solve LP to find min enclosing hexagon. Returns (cx, cy, side) or None."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    nx = np.cos(angles)
    ny = np.sin(angles)
    
    M = len(verts)
    # Pre-compute dot products
    # verts shape (M,2), normals shape (6,)
    vdots = verts[:, 0] * nx[np.newaxis, :] + verts[:, 1] * ny[np.newaxis, :]  # (M,6)
    
    # Build A and b efficiently
    # For each normal j: -nx_j*cx - ny_j*cy - R <= -vdots[i,j]  for all i
    #                    nx_j*cx + ny_j*cy - R <=  vdots[i,j]  for all i
    # Plus: -R <= 0  (i.e., R >= 0)
    
    # Total rows: 2*M*6 + 1
    total_rows = 2 * M * 6 + 1
    A = np.zeros((total_rows, 3))
    b = np.zeros(total_rows)
    
    row = 0
    for j in range(6):
        nj_x, nj_y = nx[j], ny[j]
        # -nx_j*cx - ny_j*cy - R <= -vdots[:,j]
        A[row:row+M, 0] = -nj_x
        A[row:row+M, 1] = -nj_y
        A[row:row+M, 2] = -1.0
        b[row:row+M] = -vdots[:, j]
        row += M
        # nx_j*cx + ny_j*cy - R <= vdots[:,j]
        A[row:row+M, 0] = nj_x
        A[row:row+M, 1] = nj_y
        A[row:row+M, 2] = -1.0
        b[row:row+M] = vdots[:, j]
        row += M
    
    A[row, 2] = -1.0
    b[row] = 0.0
    
    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs', options={'presolve': True,
                                               'dual_feasibility_tolerance': 1e-12,
                                               'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None

def _build_hex_lattice(rings, scale=1.0000001):
    """Build hexagonal lattice positions centered at origin."""
    sqrt3 = _SQRT3 * scale
    u = (sqrt3, 0.0)
    v = (sqrt3 / 2, 1.5 * scale)
    
    positions = [(0.0, 0.0)]  # center
    
    for ring in range(1, rings + 1):
        # Hexagonal ring: 6*ring positions
        # Start at (ring, 0) and go around
        # Using axial coordinates
        for k in range(6 * ring):
            # Direction changes every ring steps
            seg = k // ring
            pos_in_seg = k % ring
            if seg == 0:
                x = ring + pos_in_seg
                y = -pos_in_seg
            elif seg == 1:
                x = ring - pos_in_seg
                y = -ring
            elif seg == 2:
                x = -ring
                y = -ring + pos_in_seg
            elif seg == 3:
                x = -ring - pos_in_seg
                y = pos_in_seg
            elif seg == 4:
                x = -ring + pos_in_seg
                y = ring
            else:
                x = ring
                y = ring - pos_in_seg
            px = x * u[0] + y * v[0]
            py = x * u[1] + y * v[1]
            positions.append((px, py))
    
    return positions

def _build_staggered_rows(n_rows, per_row, scale=1.0000001):
    """Build staggered rows of hexagons."""
    sqrt3 = _SQRT3 * scale
    positions = []
    for r in range(n_rows):
        y = r * 1.5 * scale
        offset = (sqrt3 / 2) if r % 2 == 1 else 0.0
        for c in range(per_row):
            x = c * sqrt3 + offset
            positions.append((x, y))
    return positions

def _build_positions():
    """Generate multiple candidate layouts for 11 hexagons."""
    layouts = []
    
    # Layout 0: 3-4-3-1 staggered (classic)
    # Actually let's do proper hex packing
    sqrt3 = _SQRT3 * 1.0000001
    s = 1.0000001
    
    # Layout 0: Center + 6 ring + 4 more (partial ring 2)
    pos0 = _build_hex_lattice(1, s)  # 7 hexagons
    # Add 4 from ring 2
    ring2 = _build_hex_lattice(2, s)[7:]
    # Pick 4 closest to center
    ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
    pos0.extend(ring2_sorted[:4])
    layouts.append(pos0[:N])
    
    # Layout 1: 3-4-3-1 rows (offset)
    pos1 = []
    for row, count, offset in [(0, 3, 0), (1, 4, sqrt3/2), (2, 3, 0), (3, 1, sqrt3/2)]:
        y = row * 1.5 * s
        for c in range(count):
            x = c * sqrt3 + offset
            pos1.append((x, y))
    layouts.append(pos1[:N])
    
    # Layout 2: 2-4-4-1 rows
    pos2 = []
    for row, count, offset in [(0, 2, sqrt3/4), (1, 4, 0), (2, 4, sqrt3/2), (3, 1, sqrt3/4)]:
        y = row * 1.5 * s
        for c in range(count):
            x = c * sqrt3 + offset
            pos2.append((x, y))
    layouts.append(pos2[:N])
    
    # Layout 3: 4-3-4 rows (centered)
    pos3 = []
    for row, count, offset in [(0, 4, 0), (1, 3, sqrt3/2), (2, 4, 0)]:
        y = row * 1.5 * s
        for c in range(count):
            x = c * sqrt3 + offset
        pos3.append((x, y))
    # Wait, that's wrong. Let me redo.
    pos3 = []
    for row, count, offset in [(0, 4, 0), (1, 3, sqrt3/2), (2, 4, 0)]:
        y = row * 1.5 * s
        for c in range(count):
            x = c * sqrt3 + offset
            pos3.append((x, y))
    layouts.append(pos3[:N])
    
    # Layout 4: 2-3-3-2-1 (more compact)
    pos4 = []
    for row, count, offset in [(0, 2, sqrt3/2), (1, 3, 0), (2, 3, sqrt3/2), (3, 2, 0), (4, 1, sqrt3/2)]:
        y = row * 1.5 * s
        for c in range(count):
            x = c * sqrt3 + offset
            pos4.append((x, y))
    layouts.append(pos4[:N])
    
    # Layout 5: 3-3-3-2 rows
    pos5 = []
    for row, count, offset in [(0, 3, 0), (1, 3, sqrt3/2), (2, 3, 0), (3, 2, sqrt3/2)]:
        y = row * 1.5 * s
        for c in range(count):
            x = c * sqrt3 + offset
            pos5.append((x, y))
    layouts.append(pos5[:N])
    
    # Layout 6: 1-4-4-2 rows
    pos6 = []
    for row, count, offset in [(0, 1, sqrt3), (1, 4, 0), (2, 4, sqrt3/2), (3, 2, sqrt3)]:
        y = row * 1.5 * s
        for c in range(count):
            x = c * sqrt3 + offset
            pos6.append((x, y))
    layouts.append(pos6[:N])
    
    # Layout 7: Diamond/rhombus arrangement
    pos7 = []
    # 4-3-4 but shifted
    for row, count, offset in [(0, 4, 0), (1, 3, sqrt3/2), (2, 4, 0)]:
        y = row * 1.5 * s
        for c in range(count):
            x = c * sqrt3 + offset
            pos7.append((x, y))
    layouts.append(pos7[:N])
    
    return layouts

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    layouts = _build_positions()
    inner_angles = [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 7.5, -7.5, 22.5, -22.5]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 12.0:
            break
        
        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 12.0:
                break
            
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
            
            # Coarse grid search over outer angle [0, 30)
            coarse_best = (float('inf'), 0.0, None)
            for outer_angle_deg in np.arange(0, 30.0, 1.0):
                if time.time() > _DEADLINE - 12.0:
                    break
                res = _solve_lp(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < coarse_best[0]:
                        coarse_best = (side, outer_angle_deg, (cx, cy))
            
            if coarse_best[0] >= best_side:
                continue
            
            # Medium grid: 0.1 degree steps around best
            best_med = coarse_best
            lo = max(0.0, coarse_best[1] - 1.0)
            hi = min(30.0, coarse_best[1] + 1.0)
            for outer_angle_deg in np.arange(lo, hi + 0.05, 0.1):
                if time.time() > _DEADLINE - 12.0:
                    break
                res = _solve_lp(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_med[0]:
                        best_med = (side, outer_angle_deg, (cx, cy))
            
            # Fine grid: 0.01 degree steps
            best_fine = best_med
            lo = max(0.0, best_med[1] - 0.1)
            hi = min(30.0, best_med[1] + 0.1)
            for outer_angle_deg in np.arange(lo, hi + 0.005, 0.01):
                if time.time() > _DEADLINE - 12.0:
                    break
                res = _solve_lp(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_fine[0]:
                        best_fine = (side, outer_angle_deg, (cx, cy))
            
            # Ultra-fine: 0.001 degree steps
            best_ufine = best_fine
            lo = max(0.0, best_fine[1] - 0.01)
            hi = min(30.0, best_fine[1] + 0.01)
            for outer_angle_deg in np.arange(lo, hi + 0.0005, 0.001):
                if time.time() > _DEADLINE - 12.0:
                    break
                res = _solve_lp(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_ufine[0]:
                        best_ufine = (side, outer_angle_deg, (cx, cy))
            
            if best_ufine[0] < best_side:
                best_side = best_ufine[0]
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_ufine[2]
                best_outer_angle = best_ufine[1]
                best_inner_angle = inner_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final precision polish
    if time.time() < _DEADLINE - 5.0:
        all_vertices_final = np.vstack([_hex_vertices(px, py, best_inner_angle) 
                                        for px, py in [(p[0], p[1]) for p in best_inner]])
        
        # Very fine grid around best angle
        lo = max(0.0, best_outer_angle - 0.001)
        hi = min(30.0, best_outer_angle + 0.001)
        for outer_angle_deg in np.arange(lo, hi + 0.00005, 0.0001):
            if time.time() > _DEADLINE - 5.0:
                break
            res = _solve_lp(all_vertices_final, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_outer_angle = outer_angle_deg
    
    return best_inner, best_center, best_side, best_outer_angle
