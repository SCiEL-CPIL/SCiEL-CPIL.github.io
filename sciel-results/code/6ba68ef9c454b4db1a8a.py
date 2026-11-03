import math
import numpy as np
import time
from scipy.optimize import minimize_scalar
from itertools import combinations

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []

    # Core 7-point hexagon
    core = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Ring 2 points (hex distance 2 from origin)
    ring2 = [
        (2, 0), (2, -1), (1, -2), (0, -2), (-1, -1), (-2, 0),
        (-2, 1), (-1, 2), (0, 2), (1, 1), (2, 1), (2, 2) # Wait, check these
    ]
    # Let's verify ring 2 points
    # Hex dist (q,r) is (|q| + |r| + |q+r|) / 2
    # (2,0) -> 2. (2,-1) -> (2+1+1)/2=2. (1,-2) -> (1+2+1)/2=2. (0,-2) -> 2. (-1,-1) -> 2. (-2,0) -> 2.
    # (-2,1) -> (2+1+1)/2=2. (-1,2) -> (1+2+1)/2=2. (0,2) -> 2. (1,1) -> (1+1+2)/2=2. (2,1) -> (2+1+3)/2=3? No.
    # (2,1): |2|+|1|+|3| = 6/2=3. So (2,1) is ring 3.
    # Let's list ring 2 correctly.
    # q from -2 to 2
    # If q=2, r can be -2, -1, 0?
    # dist(2, -2) = (2+2+0)/2=2.
    # dist(2, -1) = (2+1+1)/2=2.
    # dist(2, 0) = 2.
    # dist(2, 1) = 3.
    # So for q=2, r in {-2, -1, 0}.
    # For q=1, r in {-2, -1, 0, 1}?
    # dist(1, -2) = 2. dist(1, -1) = 1 (ring 1). dist(1, 0) = 1. dist(1, 1) = 2.
    # So for q=1, r in {-2, 1}.
    # For q=0, r in {-2, 2}.
    # For q=-1, r in {-1, 2}?
    # dist(-1, -1) = 2. dist(-1, 2) = 2.
    # For q=-2, r in {0, 1, 2}?
    # dist(-2, 0) = 2. dist(-2, 1) = 2. dist(-2, 2) = 2.
    
    ring2_correct = [
        (2, 0), (2, -1), (2, -2),
        (1, -2), (1, 1),
        (0, -2), (0, 2),
        (-1, -1), (-1, 2),
        (-2, 0), (-2, 1), (-2, 2)
    ]
    # Count: 3 + 2 + 2 + 2 + 3 = 12. Correct.

    # Generate all combinations of 4 points from ring2
    for combo in combinations(ring2_correct, 4):
        candidates.append(core + list(combo))

    # Add some other candidates for diversity
    # Line of 11
    candidates.append([(i, 0) for i in range(11)])
    # T-shape
    candidates.append([(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0), (10,0)]) # Just line again, let's make a real T
    # Real T
    t_shape = [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0), (10,0)]
    # Let's just use a few hardcoded ones as fallbacks
    axial_line = [(i, 0) for i in range(11)]
    candidates.append(axial_line)
    
    axial_diag = [(i, i) for i in range(11)]
    candidates.append(axial_diag)

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

        # Center the points at origin for symmetry
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        pts_centered = pts - np.array([cx0, cy0])

        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))

        # Compute max projection in each normal direction
        max_proj = 0.0
        for nx, ny in normals:
            projs = pts_centered[:, 0] * nx + pts_centered[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))

        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S, (cx0, cy0), D

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse grid search over beta
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback to first candidate
        best_global_pts = axial_to_cartesian(candidates[0])
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

    # Use a very small safety margin to ensure no overlap
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
