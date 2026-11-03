import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3.0)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def _check_disjoint(inner_hexes, tol=1e-7):
    """Check all pairs of inner hexagons are disjoint using SAT."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            if _hexagon_overlap(inner_hexes[i], inner_hexes[j], tol):
                return False
    return True

def _hexagon_overlap(h1, h2, tol=1e-7):
    """SAT for two convex hexagons."""
    def get_axes(poly):
        axes = []
        for i in range(6):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % 6]
            dx, dy = x2-x1, y2-y1
            length = math.sqrt(dx*dx + dy*dy)
            if length > 1e-12:
                axes.append((-dy/length, dx/length))
        return axes
    
    axes = get_axes(h1) + get_axes(h2)
    for ax, ay in axes:
        proj1 = [vx*ax + vy*ay for vx, vy in h1]
        proj2 = [vx*ax + vy*ay for vx, vy in h2]
        if max(proj1) < min(proj2) + tol or max(proj2) < min(proj1) + tol:
            return False
    return True

def _axial_to_cartesian(axial_list):
    """Convert axial hex coords to cartesian (unit hex spacing)."""
    pts = []
    for q, r in axial_list:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def _get_normals(beta):
    """6 unit outward normals of a regular hexagon with rotation beta."""
    return np.array([
        (math.cos(beta + math.pi/6 + k*math.pi/3),
         math.sin(beta + math.pi/6 + k*math.pi/3))
        for k in range(6)
    ])

def _compute_D(pts, center, normals):
    """D = max |(p-c).n| over all points and normals."""
    centered = pts - np.array(center)
    projs = centered @ normals.T
    return float(np.max(np.abs(projs)))

def _calc_S_for_beta(pts, beta):
    """LP over center for fixed beta. Returns (S, center)."""
    normals = _get_normals(beta)
    c_obj = np.array([0.0, 0.0, 1.0])
    n_pts = len(pts)
    
    A_ub_blocks = []
    b_ub_blocks = []
    for nx, ny in normals:
        proj = pts[:, 0]*nx + pts[:, 1]*ny
        A1 = np.tile([nx, ny, -1.0], (n_pts, 1))
        A2 = np.tile([-nx, -ny, -1.0], (n_pts, 1))
        A_ub_blocks.extend([A1, A2])
        b_ub_blocks.extend([proj, -proj])
    
    A_ub = np.vstack(A_ub_blocks)
    b_ub = np.concatenate(b_ub_blocks)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = float(res.x[2])
            cx, cy = float(res.x[0]), float(res.x[1])
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy)
    except Exception:
        pass
    return float('inf'), None

def _optimize_shape(pts):
    """Optimize (center, rotation) for a fixed point set."""
    best_S = float('inf')
    best_beta = 0.0
    best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
    
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return best_S, best_beta, best_center
    
    # Phase 1: Grid search over beta
    n_grid = 181
    for i in range(n_grid):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        beta = i * (math.pi/6) / (n_grid - 1)
        S, center = _calc_S_for_beta(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            if center is not None:
                best_center = center
    
    # Phase 2: Bounded scalar refine
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            lo = max(0.0, best_beta - math.pi/60)
            hi = min(math.pi/6, best_beta + math.pi/60)
            res = minimize_scalar(lambda b: _calc_S_for_beta(pts, b)[0],
                                  bounds=(lo, hi), method='bounded',
                                  options={'maxiter': 2000, 'xatol': 1e-15})
            if res.fun < best_S:
                best_S = float(res.fun)
                best_beta = float(res.x)
                _, best_center = _calc_S_for_beta(pts, best_beta)
    except Exception:
        pass
    
    # Phase 3: Nelder-Mead polish in 3D (cx, cy, beta)
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def obj_3d(params):
                cx, cy, b = params
                b = ((b % (math.pi/6)) + math.pi/6) % (math.pi/6)
                normals = _get_normals(b)
                D = _compute_D(pts, (cx, cy), normals)
                return (D + APOTHEM) / APOTHEM
            
            starts = [
                [best_center[0], best_center[1], best_beta],
                [best_center[0], best_center[1], best_beta + 0.01],
                [best_center[0], best_center[1], best_beta - 0.01],
                [best_center[0]+0.01, best_center[1], best_beta],
                [best_center[0], best_center[1]+0.01, best_beta],
            ]
            for init in starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    res = minimize(obj_3d, init, method='Nelder-Mead',
                                   options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-16})
                    if res.fun < best_S:
                        best_S = float(res.fun)
                        best_center = (float(res.x[0]), float(res.x[1]))
                        best_beta = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
                except Exception:
                    pass
    except Exception:
        pass
    
    return best_S, best_beta, best_center

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Generate diverse 11-hexagon polyhex candidates
    candidates = []
    seen = set()
    
    def add_candidate(axial):
        # Normalize: sort and shift to canonical form
        sorted_pts = sorted(axial)
        # Shift so min q, min r, min (q+r) are at origin
        min_q = min(p[0] for p in sorted_pts)
        min_r = min(p[1] for p in sorted_pts)
        min_s = min(p[0]+p[1] for p in sorted_pts)
        # Use a simple canonical form: sort the set
        key = tuple(sorted((q - min_q, r - min_r) for q, r in sorted_pts))
        if key not in seen:
            seen.add(key)
            candidates.append(sorted_pts)
    
    # --- Hexagonal ring based candidates (remove 3 from 14-hex ring) ---
    # 14-hex ring: positions at distance 2 from center in hex grid
    # Actually, let's use a 3x3x3 hex block minus corners, or other shapes
    
    # Compact 3x3 hex block (7 hexes) + extensions
    base3x3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Add 4 more hexes in various positions
    extensions_pool = [
        (2,0), (-2,0), (0,2), (0,-2),
        (2,1), (2,-1), (-2,2), (-2,-2),
        (1,2), (1,-2), (-1,3), (-1,-3),
        (3,0), (-3,0), (0,3), (0,-3),
        (2,2), (-2,-2), (3,1), (-3,-1),
        (1,1), (-1,-1), (2,-2), (-2,2),
    ]
    
    # Try all combinations of 4 extensions from pool
    from itertools import combinations
    ext_combos = list(combinations(extensions_pool, 4))
    for combo in ext_combos:
        axial = base3x3 + list(combo)
        # Check no duplicates
        if len(set(axial)) == 11:
            add_candidate(axial)
    
    # --- Other structural families ---
    
    # 2x3 rectangle + 5
    rect2x3 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]
    for combo in combinations(extensions_pool, 5):
        axial = rect2x3 + list(combo)
        if len(set(axial)) == 11:
            add_candidate(axial)
    
    # 1x5 line + 6
    line5 = [(0,0),(1,0),(2,0),(3,0),(4,0)]
    for combo in combinations(extensions_pool, 6):
        axial = line5 + list(combo)
        if len(set(axial)) == 11:
            add_candidate(axial)
    
    # Diamond/rhombus shapes
    diamond1 = [(0,0),(1,0),(0,1),(1,1),(2,0),(2,1),(1,2),(0,2),(-1,1),(-1,2),(0,3)]
    add_candidate(diamond1)
    
    diamond2 = [(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,1),(1,2),(2,2),(0,2),(1,3)]
    add_candidate(diamond2)
    
    # T-shapes
    t1 = [(0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(1,-1),(2,-1),(3,-1)]
    add_candidate(t1)
    
    # Cross/plus shapes
    cross1 = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2),(1,1),(-1,-1)]
    add_candidate(cross1)
    
    cross2 = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1),(0,2),(0,-2)]
    add_candidate(cross2)
    
    # Zigzag
    zigzag1 = [(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3),(4,3),(0,1),(1,2),(2,3)]
    add_candidate(zigzag1)
    
    # L-shapes
    l1 = [(0,0),(1,0),(2,0),(0,1),(0,2),(0,3),(0,4),(1,1),(1,2),(1,3),(1,4)]
    add_candidate(l1)
    
    l2 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(0,2),(0,3),(1,1),(2,1),(3,1)]
    add_candidate(l2)
    
    # Compact clusters
    compact1 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    add_candidate(compact1)
    
    compact2 = [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)]
    add_candidate(compact2)
    
    compact3 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(1,3),(2,3)]
    add_candidate(compact3)
    
    compact4 = [(0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(2,2),(3,2),(4,2)]
    add_candidate(compact4)
    
    # Hexagonal 11 (center + ring of 6 + 4 more)
    hex11 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    add_candidate(hex11)
    
    hex11b = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(-2,2)]
    add_candidate(hex11b)
    
    # Limit candidates to manageable number
    max_candidates = 200
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    
    # --- Main optimization loop ---
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    
    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = _axial_to_cartesian(axial)
        
        # Quick bounding box filter
        x_range = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        y_range = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        rough_S = max(x_range, y_range * 2.0 / SQRT3) / (2.0 * APOTHEM) + 1.0
        if best_global_S < float('inf') and rough_S > best_global_S * 1.1:
            continue
        
        S, beta, center = _optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
    
    # --- Fallback ---
    if best_global_center is None or best_global_S == float('inf'):
        axial_fb = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
        best_global_pts = _axial_to_cartesian(axial_fb)
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        normals = _get_normals(best_global_beta)
        max_proj = 0.0
        for nx, ny in normals:
            projs = best_global_pts[:, 0]*nx + best_global_pts[:, 1]*ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
    
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Final: recompute exact S
    normals = _get_normals(best_global_beta)
    max_proj = 0.0
    for nx, ny in normals:
        projs = (best_global_pts[:, 0] - cx)*nx + (best_global_pts[:, 1] - cy)*ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    
    S_final = (max_proj + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
