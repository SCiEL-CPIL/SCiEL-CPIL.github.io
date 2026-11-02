import math
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _time_left():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _generate_candidates():
    """
    Generate candidate configurations of 11 unit hexagons.
    Returns a list of np.array of shape (11, 2).
    """
    candidates = []
    sqrt3 = math.sqrt(3)
    
    # Helper to convert axial to cartesian
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
    
    # 1. Hardcoded good configurations from current program
    hardcoded = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)], # This has 12, skip if >11
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)], # 12, skip
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)], # 12, skip
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)], # 12, skip
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)], # 12, skip
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)]
    ]
    
    # Filter and convert
    for h in hardcoded:
        if len(h) == 11:
            candidates.append(axial_to_cartesian(h))
            
    # 2. Generate more systematic candidates
    # Start with central hexagon of 7, add 4 more
    central_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Neighbors to add
    # We want to add 4 cells that keep the shape compact.
    # Possible additions: (2,0), (-2,0), (0,2), (0,-2), (2,-1), (-2,1), etc.
    # Let's try all combinations of 4 from a set of "outer ring" candidates
    outer_ring = [
        (2,0), (-2,0), (0,2), (0,-2), 
        (2,-1), (-2,1), (1,2), (-1,-2), 
        (2,1), (-2,-1), (1,-2), (-1,2)
    ]
    
    # This is too many combinations (12 choose 4 = 495).
    # We can just pick a few good ones manually or use a heuristic.
    # Heuristic: Add cells that minimize the bounding box area.
    
    # Let's just add a few specific "star" extensions
    ext1 = central_7 + [(2,0), (-2,0), (0,2), (0,-2)]
    ext2 = central_7 + [(2,0), (-2,0), (1,1), (-1,-1)]
    ext3 = central_7 + [(2,-1), (-2,1), (1,2), (-1,-2)]
    ext4 = central_7 + [(2,0), (0,2), (-2,1), (0,-2)]
    ext5 = central_7 + [(1,0), (0,1), (-1,1), (-1,0)] # Wait, these are in central_7.
    
    # Correct extensions (must be disjoint from central_7)
    # Central 7 covers: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Available neighbors:
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,2), (2,1)
    
    # Try a few specific compact shapes
    shapes = [
        central_7 + [(2,0), (-2,0), (0,2), (0,-2)],
        central_7 + [(2,0), (0,2), (-2,1), (0,-2)],
        central_7 + [(2,-1), (-2,1), (1,2), (-1,-2)],
        central_7 + [(2,0), (-2,0), (1,2), (-1,-2)],
        central_7 + [(2,0), (-2,0), (0,2), (1,-2)],
        central_7 + [(2,0), (-2,0), (0,2), (-1,2)],
        central_7 + [(2,0), (0,2), (-2,0), (0,-2)], # Duplicate of first
        central_7 + [(1,1), (-1,-1), (2,0), (-2,0)], # (1,1) is not in central 7? 
        # Central 7: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # (1,1) is a neighbor of (1,0) and (0,1). It is NOT in central 7.
        # So this is valid.
    ]
    
    for s in shapes:
        if len(s) == 11:
            # Check for duplicates (axial coords)
            if len(set(s)) == 11:
                candidates.append(axial_to_cartesian(s))

    # Remove duplicates based on set of rounded coords
    unique_candidates = []
    seen = set()
    for c in candidates:
        # Round to 2 decimals to handle float errors
        key = tuple(sorted([tuple(np.round(p, 2)) for p in c]))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    return unique_candidates

def _smooth_objective(params, pts, temperature=0.1):
    """
    Smooth surrogate for the minimum enclosing hexagon side length.
    params: [cx, cy, beta]
    """
    cx, cy, beta = params
    # Normalize beta to [0, pi/6]
    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
    
    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    pts_centered = pts - np.array([cx, cy])
    projs = pts_centered @ normals.T  # (n_pts, 6)
    all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
    
    max_val = np.max(all_projs)
    if max_val > 100:
        return 1e10
    
    shifted = (all_projs - max_val) / temperature
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
    D = temperature * log_sum
    S = (D + APOTHEM) / APOTHEM
    return S

def _exact_objective(params, pts):
    """
    Exact objective function using max projection.
    """
    cx, cy, beta = params
    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
    
    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    pts_centered = pts - np.array([cx, cy])
    projs = np.abs(pts_centered @ normals.T)
    D = np.max(projs)
    S = (D + APOTHEM) / APOTHEM
    return S

def _optimize_configuration(pts):
    """
    Optimize the enclosing hexagon for a given point configuration.
    Returns (best_S, best_center, best_beta).
    """
    if _time_left() < 1.0:
        return float('inf'), None, None
    
    # Initial guess: centroid
    cx0 = np.mean(pts[:, 0])
    cy0 = np.mean(pts[:, 1])
    beta0 = 0.0
    
    # Stage 1: Smooth optimization
    best_S = float('inf')
    best_params = None
    
    # Try a few starting betas
    start_betas = [0.0, math.pi/12, math.pi/6, math.pi/24, 3*math.pi/24]
    
    for sb in start_betas:
        if _time_left() < 0.5:
            break
        init = [cx0, cy0, sb]
        try:
            res = minimize(_smooth_objective, init, args=(pts, 0.05), method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        # Fallback to grid search if optimization fails
        best_S = float('inf')
        for i in range(13):
            beta = i * (math.pi / 6) / 12
            S = _exact_objective([cx0, cy0, beta], pts)
            if S < best_S:
                best_S = S
                best_params = [cx0, cy0, beta]
        
    # Stage 2: Exact optimization with Nelder-Mead
    if best_params is not None and _time_left() > 0.5:
        try:
            res = minimize(_exact_objective, best_params, args=(pts,), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
            
    cx, cy, beta = best_params
    return best_S, (cx, cy), beta

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = _generate_candidates()
    
    # Sort by bounding box area to prioritize compact ones
    def bbox_area(pts):
        return (np.max(pts[:, 0]) - np.min(pts[:, 0])) * (np.max(pts[:, 1]) - np.min(pts[:, 1]))
    
    candidates.sort(key=bbox_area)
    
    best_global_S = float('inf')
    best_global_pts = None
    best_global_center = None
    best_global_beta = 0.0
    
    for pts in candidates:
        if _time_left() < 2.0:
            break
            
        S, center, beta = _optimize_configuration(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_center = center
            best_global_beta = beta
            
    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = candidates[0]
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
    
    # Final verification and safety scaling
    # We scale the points slightly inward to ensure they fit
    # But the task asks for the side length of the OUTER hexagon.
    # The score is the side length.
    # We should return the side length that exactly contains the points.
    
    # Recalculate exact S for the best configuration
    S_final = _exact_objective([cx, cy, best_global_beta], best_global_pts)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

