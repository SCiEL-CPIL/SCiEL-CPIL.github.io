import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3
_PI = math.pi

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * _PI / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _hex_vertices_all(positions, angle_deg):
    """Get all vertices for multiple hexagons."""
    verts = []
    for px, py in positions:
        verts.append(_hex_vertices(px, py, angle_deg))
    return np.vstack(verts)

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (flat-top offset by outer_angle_deg)
    that contains all points in verts. Uses LP.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * _PI / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _min_enclosing_hex_side_fast(verts, outer_angle_deg):
    """Fast version: just compute the required side length for a given center (centroid)."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * _PI / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _compute_side_for_positions(positions, inner_angle, outer_angle):
    """Compute the enclosing hex side for given positions and angles."""
    verts = _hex_vertices_all(positions, inner_angle)
    _, _, side = _min_enclosing_hex_side_fast(verts, outer_angle)
    return side

def _build_hexagonal_cluster(center, scale=1.0):
    """
    Build 11 hexagons in a hexagonal cluster:
    center + 6 in ring 1 + 4 in ring 2 (symmetric selection)
    """
    s3 = _SQRT3 * scale
    positions = []
    # Center
    positions.append((center[0], center[1]))
    # Ring 1: 6 hexagons
    for i in range(6):
        angle = i * _PI / 3
        x = center[0] + s3 * math.cos(angle)
        y = center[1] + 1.5 * scale * math.sin(angle)
        positions.append((x, y))
    # Ring 2: 4 hexagons at alternating positions
    ring2_angles = [0, _PI/3, 2*_PI/3, _PI]
    for i in range(4):
        angle = ring2_angles[i]
        x = center[0] + 2*s3 * math.cos(angle)
        y = center[1] + 3.0 * scale * math.sin(angle)
        positions.append((x, y))
    return positions

def _build_row_layout(n_rows=3, offset=0.0):
    """Build a 3-row layout with 4-3-4 or similar pattern."""
    s3 = _SQRT3
    scale = 1.0
    positions = []
    # Row 1: 4 hexagons
    y1 = -1.5 * scale
    for k in range(-1, 3):
        x = k * s3 * scale + s3/2 * scale + offset
        positions.append((x, y1))
    # Row 2: 3 hexagons
    y2 = 0.0
    for k in range(-1, 2):
        x = k * s3 * scale + offset
        positions.append((x, y2))
    # Row 3: 4 hexagons
    y3 = 1.5 * scale
    for k in range(-1, 3):
        x = k * s3 * scale + s3/2 * scale + offset
        positions.append((x, y3))
    return positions

def _build_compact_11():
    """Compact 11-hexagon arrangement."""
    s3 = _SQRT3
    scale = 1.0
    positions = []
    # Center
    positions.append((0, 0))
    # 6 neighbors
    for i in range(6):
        angle = i * _PI / 3
        x = s3 * scale * math.cos(angle)
        y = 1.5 * scale * math.sin(angle)
        positions.append((x, y))
    # 4 more at ring 2
    for i in range(4):
        angle = i * _PI / 2 + _PI/4
        x = 2*s3 * scale * math.cos(angle)
        y = 3.0 * scale * math.sin(angle)
        positions.append((x, y))
    return positions

def _optimize_positions(initial_positions, inner_angle, outer_angle, time_limit=10.0):
    """
    Optimize positions to minimize enclosing hex side length.
    Uses L-BFGS-B with the LP-based objective.
    """
    start_time = time.time()
    n = len(initial_positions)
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return float('inf')
        positions = [(x[2*i], x[2*i+1]) for i in range(n)]
        verts = _hex_vertices_all(positions, inner_angle)
        _, _, side = _min_enclosing_hex_side_fast(verts, outer_angle)
        return side
    
    x0 = np.array([initial_positions[i][0] for i in range(n)] + 
                  [initial_positions[i][1] for i in range(n)])
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < float('inf'):
            positions = [(res.x[2*i], res.x[2*i+1]) for i in range(n)]
            return positions
    except Exception:
        pass
    
    return initial_positions

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 30.0
    
    # Try multiple inner angles and outer angles
    inner_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    outer_angles = np.arange(0, 30.0, 1.0)
    
    # Initial layouts
    layouts = []
    
    # Layout 0: Hexagonal cluster
    layouts.append(_build_hexagonal_cluster((0,0)))
    
    # Layout 1: Row layout
    layouts.append(_build_row_layout())
    
    # Layout 2: Compact 11
    layouts.append(_build_compact_11())
    
    # Layout 3: 4-4-3 pattern
    s3 = _SQRT3
    scale = 1.0
    pos = []
    for k in range(-1, 3):
        pos.append((k*s3*scale + s3/2*scale, -1.5*scale))
    for k in range(-2, 2):
        pos.append((k*s3*scale, 0.0))
    for k in range(-1, 2):
        pos.append((k*s3*scale + s3/2*scale, 1.5*scale))
    layouts.append(pos)
    
    # Layout 4: 3-5-3 pattern
    pos = []
    for k in range(-1, 2):
        pos.append((k*s3*scale + s3/2*scale, -1.5*scale))
    for k in range(-2, 3):
        pos.append((k*s3*scale, 0.0))
    for k in range(-1, 2):
        pos.append((k*s3*scale + s3/2*scale, 1.5*scale))
    layouts.append(pos)
    
    # Layout 5: 5-6 pattern (2 rows)
    pos = []
    for k in range(-2, 3):
        pos.append((k*s3*scale, -0.75*scale))
    for k in range(-3, 2):
        pos.append((k*s3*scale + s3/2*scale, 0.75*scale))
    layouts.append(pos)
    
    # Layout 6: 6-5 pattern (2 rows)
    pos = []
    for k in range(-3, 2):
        pos.append((k*s3*scale + s3/2*scale, -0.75*scale))
    for k in range(-2, 3):
        pos.append((k*s3*scale, 0.75*scale))
    layouts.append(pos)
    
    # Layout 7: Diamond/rotated cluster
    pos = []
    pos.append((0, 0))
    for i in range(6):
        angle = i * _PI / 3 + _PI/6
        x = s3 * scale * math.cos(angle)
        y = 1.5 * scale * math.sin(angle)
        pos.append((x, y))
    for i in range(4):
        angle = i * _PI / 2
        x = 2*s3 * scale * math.cos(angle)
        y = 3.0 * scale * math.sin(angle)
        pos.append((x, y))
    layouts.append(pos)
    
    # Layout 8: Offset row
    pos = []
    for k in range(-1, 3):
        pos.append((k*s3*scale + s3/4*scale, -1.5*scale))
    for k in range(-2, 2):
        pos.append((k*s3*scale + s3/4*scale, 0.0))
    for k in range(-1, 2):
        pos.append((k*s3*scale + s3/4*scale, 1.5*scale))
    layouts.append(pos)
    
    # Layout 9: Staggered
    pos = []
    for k in range(-1, 2):
        pos.append((k*s3*scale, -1.5*scale))
    for k in range(-1, 2):
        pos.append((k*s3*scale + s3/2*scale, 0.0))
    for k in range(-2, 3):
        pos.append((k*s3*scale, 1.5*scale))
    layouts.append(pos)
    
    # Layout 10: 2-5-4
    pos = []
    for k in range(0, 2):
        pos.append((k*s3*scale - s3/2*scale, -3.0*scale))
    for k in range(-2, 3):
        pos.append((k*s3*scale, -1.5*scale))
    for k in range(-2, 2):
        pos.append((k*s3*scale + s3/2*scale, 0.0))
    layouts.append(pos)
    
    # Layout 11: 4-5-2
    pos = []
    for k in range(-2, 2):
        pos.append((k*s3*scale, -1.5*scale))
    for k in range(-2, 3):
        pos.append((k*s3*scale + s3/2*scale, 0.0))
    for k in range(0, 2):
        pos.append((k*s3*scale - s3/2*scale, 1.5*scale))
    layouts.append(pos)
    
    # Try each layout with different angles
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 15.0:
                break
            
            best_local_side = float('inf')
            best_local_center = (0.0, 0.0)
            best_local_outer_angle = 0.0
            
            for outer_angle in outer_angles:
                if time.time() > _DEADLINE - 15.0:
                    break
                
                side = _compute_side_for_positions(positions, inner_angle, outer_angle)
                if side < best_local_side:
                    best_local_side = side
                    _, _, _ = _min_enclosing_hex_side_lp(_hex_vertices_all(positions, inner_angle), outer_angle)
                    best_local_center = _min_enclosing_hex_side_lp(_hex_vertices_all(positions, inner_angle), outer_angle)[:2]
                    best_local_outer_angle = outer_angle
            
            if best_local_side < best_side:
                best_side = best_local_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_local_center
                best_outer_angle = best_local_outer_angle
                best_inner_angle = inner_angle
        
        # Try to optimize the best configuration for this layout
        if time.time() < _DEADLINE - 20.0:
            # Find best angles for this layout
            best_layout_side = float('inf')
            best_layout_inner_angle = 30.0
            best_layout_outer_angle = 0.0
            
            for inner_angle in inner_angles:
                for outer_angle in outer_angles:
                    side = _compute_side_for_positions(positions, inner_angle, outer_angle)
                    if side < best_layout_side:
                        best_layout_side = side
                        best_layout_inner_angle = inner_angle
                        best_layout_outer_angle = outer_angle
            
            # Optimize positions
            optimized = _optimize_positions(positions, best_layout_inner_angle, best_layout_outer_angle, time_limit=8.0)
            side = _compute_side_for_positions(optimized, best_layout_inner_angle, best_layout_outer_angle)
            
            if side < best_side:
                best_side = side
                best_inner = [[pos[0], pos[1], best_layout_inner_angle] for pos in optimized]
                _, _, _ = _min_enclosing_hex_side_lp(_hex_vertices_all(optimized, best_layout_inner_angle), best_layout_outer_angle)
                best_center = _min_enclosing_hex_side_lp(_hex_vertices_all(optimized, best_layout_inner_angle), best_layout_outer_angle)[:2]
                best_outer_angle = best_layout_outer_angle
                best_inner_angle = best_layout_inner_angle
    
    # Refine with finer angle search
    if best_inner is not None and time.time() < _DEADLINE - 25.0:
        best_positions = [(p[0], p[1]) for p in best_inner]
        
        # Finer outer angle search
        angle_start = max(0, best_outer_angle - 1.0)
        angle_end = min(30.0, best_outer_angle + 1.0)
        
        for outer_angle in np.arange(angle_start, angle_end, 0.05):
            if time.time() > _DEADLINE - 15.0:
                break
            
            side = _compute_side_for_positions(best_positions, best_inner_angle, outer_angle)
            if side < best_side:
                best_side = side
                _, _, _ = _min_enclosing_hex_side_lp(_hex_vertices_all(best_positions, best_inner_angle), outer_angle)
                best_center = _min_enclosing_hex_side_lp(_hex_vertices_all(best_positions, best_inner_angle), outer_angle)[:2]
                best_outer_angle = outer_angle
        
        # Try optimizing positions with the refined angles
        if time.time() < _DEADLINE - 20.0:
            optimized = _optimize_positions(best_positions, best_inner_angle, best_outer_angle, time_limit=10.0)
            side = _compute_side_for_positions(optimized, best_inner_angle, best_outer_angle)
            
            if side < best_side:
                best_side = side
                best_inner = [[pos[0], pos[1], best_inner_angle] for pos in optimized]
                _, _, _ = _min_enclosing_hex_side_lp(_hex_vertices_all(optimized, best_inner_angle), best_outer_angle)
                best_center = _min_enclosing_hex_side_lp(_hex_vertices_all(optimized, best_inner_angle), best_outer_angle)[:2]
    
    # Final high-precision polish
    if best_inner is not None and time.time() < _DEADLINE - 10.0:
        best_positions = [(p[0], p[1]) for p in best_inner]
        
        # Very fine angle search
        angle_start = max(0, best_outer_angle - 0.1)
        angle_end = min(30.0, best_outer_angle + 0.1)
        
        for outer_angle in np.arange(angle_start, angle_end, 0.005):
            if time.time() > _DEADLINE - 5.0:
                break
            
            side = _compute_side_for_positions(best_positions, best_inner_angle, outer_angle)
            if side < best_side:
                best_side = side
                _, _, _ = _min_enclosing_hex_side_lp(_hex_vertices_all(best_positions, best_inner_angle), outer_angle)
                best_center = _min_enclosing_hex_side_lp(_hex_vertices_all(best_positions, best_inner_angle), outer_angle)[:2]
                best_outer_angle = outer_angle
        
        # Final position optimization
        if time.time() < _DEADLINE - 5.0:
            optimized = _optimize_positions(best_positions, best_inner_angle, best_outer_angle, time_limit=5.0)
            side = _compute_side_for_positions(optimized, best_inner_angle, best_outer_angle)
            
            if side < best_side:
                best_side = side
                best_inner = [[pos[0], pos[1], best_inner_angle] for pos in optimized]
                _, _, _ = _min_enclosing_hex_side_lp(_hex_vertices_all(optimized, best_inner_angle), best_outer_angle)
                best_center = _min_enclosing_hex_side_lp(_hex_vertices_all(optimized, best_inner_angle), best_outer_angle)[:2]
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
