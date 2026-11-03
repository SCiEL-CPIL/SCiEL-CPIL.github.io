import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Generate all connected hexagonal polyominoes of size 11 ---
    def gen_hex_polyominoes(n):
        """Generate all connected hexagonal polyominoes of size n using axial coordinates."""
        if n == 1:
            return [frozenset([(0, 0)])]
        
        # Recursively build by adding a hexagon adjacent to existing ones
        # Use a canonical form to avoid duplicates
        seen = set()
        results = []
        
        # Get all polyominoes of size n-1
        prev = gen_hex_polyominoes(n - 1)
        
        # Directions in axial coordinates
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        for poly in prev:
            # Find all possible positions to add a new hexagon
            candidates = set()
            for (q, r) in poly:
                for dq, dr in dirs:
                    candidates.add((q + dq, r + dr))
            
            for (q, r) in candidates:
                new_poly = poly | {(q, r)}
                
                # Canonicalize: translate so min q is 0, then min r is 0
                min_q = min(q for q, r in new_poly)
                min_r = min(r for q, r in new_poly)
                canonical = frozenset((q - min_q, r - min_r) for q, r in new_poly)
                
                # Also try all 6 rotations and 6 reflections to get the true canonical form
                # For now, just use the translation canonical form as a first filter
                if canonical not in seen:
                    seen.add(canonical)
                    results.append(canonical)
        
        return results

    def canonicalize_hex(poly):
        """Get the canonical form of a hexagonal polyomino by trying all symmetries."""
        # Axial to cube: x = q, z = r, y = -x-z
        def axial_to_cube(q, r):
            return (q, -q - r, r)
        
        def cube_to_axial(x, y, z):
            return (x, z)
        
        def rotate_cube(x, y, z):
            # 60 degree rotation: (x,y,z) -> (y,z,x)
            return (y, z, x)
        
        def reflect_cube(x, y, z):
            # Reflection: swap two coordinates, e.g., (x,y,z) -> (y,x,z)
            return (y, x, z)
        
        # Try all 12 symmetries of the hexagon (6 rotations * 2 reflections)
        best = None
        cube_set = set(axial_to_cube(q, r) for q, r in poly)
        
        for _ in range(6):
            # Try this rotation
            translated = []
            min_x = min(x for x, y, z in cube_set)
            min_y = min(y for x, y, z in cube_set)
            min_z = min(z for x, y, z in cube_set)
            for x, y, z in cube_set:
                translated.append((x - min_x, y - min_y, z - min_z))
            canonical = tuple(sorted(translated))
            if best is None or canonical < best:
                best = canonical
            
            # Apply rotation
            cube_set = set(rotate_cube(x, y, z) for x, y, z in cube_set)
        
        # Now try reflections
        for _ in range(6):
            translated = []
            min_x = min(x for x, y, z in cube_set)
            min_y = min(y for x, y, z in cube_set)
            min_z = min(z for x, y, z in cube_set)
            for x, y, z in cube_set:
                translated.append((x - min_x, y - min_y, z - min_z))
            canonical = tuple(sorted(translated))
            if best is None or canonical < best:
                best = canonical
            
            # Apply reflection
            cube_set = set(reflect_cube(x, y, z) for x, y, z in cube_set)
        
        return best

    def gen_hex_polyominoes_canonical(n):
        """Generate all distinct connected hexagonal polyominoes of size n."""
        if n == 1:
            return [frozenset([(0, 0)])]
        
        seen = set()
        results = []
        
        prev = gen_hex_polyominoes_canonical(n - 1)
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        for poly in prev:
            candidates = set()
            for (q, r) in poly:
                for dq, dr in dirs:
                    candidates.add((q + dq, r + dr))
            
            for (q, r) in candidates:
                new_poly = poly | {(q, r)}
                canon = canonicalize_hex(new_poly)
                if canon not in seen:
                    seen.add(canon)
                    results.append(new_poly)
        
        return results

    # Generate all 11-hexagon polyominoes
    all_polys = gen_hex_polyominoes_canonical(N)
    
    # Filter to only "compact" ones (small bounding box) to limit search
    def polyomino_compactness(poly):
        """Measure how compact a polyomino is."""
        qs = [q for q, r in poly]
        rs = [r for q, r in poly]
        width = max(qs) - min(qs) + 1
        height = max(rs) - min(rs) + 1
        return width * height
    
    # Sort by compactness and take the most compact ones
    all_polys_sorted = sorted(all_polys, key=polyomino_compactness)
    
    # Take top 500 most compact polyominoes
    top_polys = all_polys_sorted[:500]
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []

        for nx, ny in normals:
            for px, py in pts:
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
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S_vectorized(pts, cx, cy, beta):
        """Vectorized exact computation of S for given center and beta."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        max_proj = np.max(np.abs(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse grid search over beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Final high-precision polish using vectorized exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return exact_S_vectorized(pts, cx, cy, beta)

                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Evaluate all top polyominoes
    for i, poly in enumerate(top_polys):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(list(poly))
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback to a simple configuration
        axial_fallback = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]
        best_global_pts = axial_to_cartesian(axial_fallback)
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
