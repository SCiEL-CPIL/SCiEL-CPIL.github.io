import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from collections import deque

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

# Axial directions: (q, r)
AXIAL_DIRS = [
    (1, 0), (1, -1), (0, -1),
    (-1, 0), (-1, 1), (0, 1)
]

def normalize_shape(cells):
    """
    Normalize a set of axial coordinates to a canonical form.
    Translate so that min q is 0 and min r is 0.
    """
    if not cells:
        return frozenset()
    
    min_q = min(q for q, r in cells)
    min_r = min(r for q, r in cells)
    
    normalized = frozenset((q - min_q, r - min_r) for q, r in cells)
    return normalized

def generate_polyhexes(size):
    """
    Generate all free polyhexes of the given size using BFS.
    Returns a set of frozensets of axial coordinates.
    """
    if size == 1:
        return {frozenset([(0, 0)])}
    
    shapes = {frozenset([(0, 0)])}
    
    for s in range(2, size + 1):
        new_shapes = set()
        for shape in shapes:
            # For each cell in the current shape, try adding a neighbor
            # To avoid duplicates, we only add to the "frontier" in a specific way?
            # Standard BFS: add any neighbor that is not already in the shape.
            # Then normalize and add to new_shapes.
            cells = list(shape)
            for cell in cells:
                q, r = cell
                for dq, dr in AXIAL_DIRS:
                    nq, nr = q + dq, r + dr
                    if (nq, nr) not in shape:
                        new_shape = shape | {(nq, nr)}
                        # Normalize to reduce duplicates in the set
                        norm_shape = normalize_shape(new_shape)
                        new_shapes.add(norm_shape)
        
        shapes = new_shapes
        
        # Optional: Early exit if time is running out
        if time.time() - START_TIME > TIME_LIMIT - 30.0:
            break
            
    return shapes

def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def calc_S_exact(pts, beta):
    """
    Calculate the exact minimal side length S for a given rotation beta
    using Linear Programming.
    """
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
            # (px - cx)nx + (py - cy)ny <= D  => -nx*cx - ny*cy - D <= -px*nx - py*ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            
            # -((px - cx)nx + (py - cy)ny) <= D => nx*cx + ny*cy - D <= px*nx + py*ny
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

def calc_S_fast_approx(pts, beta):
    """
    Fast approximation using half-widths.
    D_approx = 0.5 * max_k (max_proj - min_proj)
    """
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_width = 0
    for nx, ny in normals:
        projs = pts[:, 0] * nx + pts[:, 1] * ny
        width = np.max(projs) - np.min(projs)
        if width > max_width:
            max_width = width
            
    D = max_width / 2.0
    S = (D + APOTHEM) / APOTHEM
    return S

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # 1. Generate all free 11-hexes
    # To save time, we might limit the generation if it takes too long, 
    # but 11-hexes are small enough.
    try:
        all_shapes = generate_polyhexes(N)
    except Exception:
        # Fallback to a few hardcoded ones if generation fails or is too slow
        all_shapes = set()
        # Hardcode a few good ones
        all_shapes.add(normalize_shape([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)]))
        all_shapes.add(normalize_shape([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)]))

    if not all_shapes:
        # Absolute fallback
        pts = axial_to_cartesian([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)])
        return [[float(x), float(y), 0.0] for x, y in pts], [0.0, 0.0], 10.0, 0.0

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    
    # 2. Fast Screening
    # We will test each shape with 6 rotations.
    # For each, we do a coarse scan of beta to find a good candidate.
    candidates_for_exact = []
    
    # Pre-compute beta steps for coarse scan
    beta_steps = np.linspace(0, math.pi / 6, 36, endpoint=False)
    
    for shape in all_shapes:
        if time.time() - START_TIME > TIME_LIMIT - 30.0:
            break
            
        cells = list(shape)
        pts_base = axial_to_cartesian(cells)
        
        # Try 6 rotations of the shape itself
        for rot_idx in range(6):
            if time.time() - START_TIME > TIME_LIMIT - 30.0:
                break
                
            # Rotate points
            if rot_idx == 0:
                pts = pts_base
            else:
                angle = rot_idx * math.pi / 3
                c = math.cos(angle)
                s = math.sin(angle)
                rot_mat = np.array([[c, -s], [s, c]])
                pts = pts_base @ rot_mat.T
                
            # Coarse scan for best beta
            best_S_approx = float('inf')
            best_beta_approx = 0.0
            
            for beta in beta_steps:
                S_approx = calc_S_fast_approx(pts, beta)
                if S_approx < best_S_approx:
                    best_S_approx = S_approx
                    best_beta_approx = beta
            
            # Store candidate
            candidates_for_exact.append((best_S_approx, best_beta_approx, pts))

    # 3. Exact Optimization for Top Candidates
    # Sort by approximate S and take top 50
    candidates_for_exact.sort(key=lambda x: x[0])
    
    top_candidates = candidates_for_exact[:50]
    
    for S_approx, beta_init, pts in top_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Refine beta using minimize_scalar around the best approx beta
        # We need to handle the periodicity. beta is in [0, pi/6).
        # Let's just do a finer grid search around beta_init and then polish.
        
        best_S = float('inf')
        best_beta = beta_init
        best_center = None
        
        # Fine grid search
        fine_betas = np.linspace(max(0, beta_init - math.pi/72), min(math.pi/6, beta_init + math.pi/72), 20)
        
        for beta in fine_betas:
            S, center, D = calc_S_exact(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Polish with Nelder-Mead
        if best_center is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            def objective(params):
                cx, cy, beta = params
                if beta < 0 or beta >= math.pi / 6:
                    return 1e10
                S, _, _ = calc_S_exact(pts, beta)
                # Wait, calc_S_exact optimizes center too. 
                # If we use Nelder-Mead on (cx, cy, beta), we should NOT optimize center inside calc_S_exact.
                # Let's create a version that takes cx, cy, beta and returns S directly.
                
                normals = []
                for k in range(6):
                    ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                    normals.append((math.cos(ang), math.sin(ang)))
                
                max_proj = 0
                for nx, ny in normals:
                    for px, py in pts:
                        proj = (px - cx) * nx + (py - cy) * ny
                        max_proj = max(max_proj, proj, -proj)
                
                S = (max_proj + APOTHEM) / APOTHEM
                return S
                
            initial = [best_center[0], best_center[1], best_beta]
            try:
                res = minimize(objective, initial, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_S:
                    best_S = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_beta = res.x[2]
            except Exception:
                pass

        if best_S < best_global_S:
            best_global_S = best_S
            best_global_pts = pts
            best_global_beta = best_beta
            best_global_center = best_center

    # Fallback if no valid result found
    if best_global_center is None or best_global_S == float('inf'):
        # Use the first candidate
        pts = axial_to_cartesian(list(all_shapes)[0])
        best_global_pts = pts
        best_global_beta = 0.0
        S, center, D = calc_S_exact(pts, 0.0)
        if center:
            best_global_center = center
            best_global_S = S
        else:
            # Manual calculation
            best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
            max_proj = 0
            for k in range(6):
                ang = math.pi / 6.0 + k * math.pi / 3.0
                nx, ny = math.cos(ang), math.sin(ang)
                projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            best_global_S = (max_proj + APOTHEM) / APOTHEM

    # Prepare output
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale to ensure disjointness (very slight expansion to avoid floating point errors)
    SCALE = 1.0 + 1e-8
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recalculate S for scaled points to report accurate side length
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

if __name__ == "__main__":
    # Test
    start = time.time()
    res = construct()
    print(f"Time taken: {time.time() - start:.2f}s")
    print(f"Side length: {res[2]:.6f}")
