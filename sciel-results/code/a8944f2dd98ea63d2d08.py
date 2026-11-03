import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0

# Precompute unit normals for a hexagon with a flat side along +x
# Vertices at angles 0, 60, ..., 300 degrees
# Edge normals at 30, 90, 150, 210, 270, 330 degrees
BASE_NORMALS = np.array([
    [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
    for k in range(6)
])

def _time_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY_MARGIN)

def _calc_S(pts, cx, cy, beta):
    """Compute side length S of enclosing hexagon given centers, center, and rotation."""
    # Rotate normals by beta
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)
    rot = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
    normals = rot @ BASE_NORMALS.T  # 2x6
    pts_c = pts - np.array([cx, cy])
    projs = pts_c @ normals  # (N, 6)
    D = float(np.max(np.abs(projs)))
    return (D + APOTHEM) / APOTHEM

def _overlap_penalty(pts, w=50.0):
    """Smooth penalty for overlapping hexagons (all with same orientation)."""
    n = len(pts)
    if n < 2:
        return 0.0
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
    # Project onto the 6 normals
    projs = np.abs(diffs @ BASE_NORMALS.T)  # (N, N, 6)
    min_projs = np.min(projs, axis=2)  # (N, N)
    iu = np.triu_indices(n, k=1)
    violations = np.maximum(0.0, SQRT3 - min_projs[iu])
    return w * float(np.sum(violations))

def _objective(params):
    n = N
    pts = params[:2*n].reshape(n, 2)
    cx = params[2*n]
    cy = params[2*n+1]
    beta = params[2*n+2]
    # Reduce beta to [0, pi/6)
    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
    S = _calc_S(pts, cx, cy, beta)
    pen = _overlap_penalty(pts)
    return S + pen

def _hex_grid_positions(rows_cols):
    """Generate positions on a hex grid. rows_cols is a list of (row, [cols...])."""
    pts = []
    for r, cols in rows_cols:
        for c in cols:
            x = c * SQRT3 + (r % 2) * (SQRT3 / 2.0)
            y = r * 1.5
            pts.append((x, y))
    return np.array(pts)

def _generate_lattice_starts():
    """Generate diverse lattice-based starting configurations."""
    starts = []
    
    # 1x11 strip
    starts.append(_hex_grid_positions([(0, list(range(11)))]))
    # 2x6 (one extra)
    starts.append(_hex_grid_positions([(0, list(range(6))), (1, list(range(5))), (2, [0])]))
    # 3x4
    starts.append(_hex_grid_positions([(0, list(range(4))), (1, list(range(4))), (2, list(range(3)))]))
    # 4x3
    starts.append(_hex_grid_positions([(0, list(range(3))), (1, list(range(3))), (2, list(range(3))), (3, list(range(2)))]))
    # 5x3
    starts.append(_hex_grid_positions([(0, list(range(3))), (1, list(range(3))), (2, list(range(3))), (3, list(range(2))), (4, [0])]))
    # 11x1 vertical
    starts.append(_hex_grid_positions([(r, [0]) for r in range(11)]))
    # 2x11 (one missing)
    starts.append(_hex_grid_positions([(0, list(range(11))), (1, list(range(10)))]))
    # 3x11 (two missing)
    starts.append(_hex_grid_positions([(0, list(range(11))), (1, list(range(11))), (2, [0, 1, 2, 3, 4, 5])]))
    # Diamond-ish
    starts.append(_hex_grid_positions([(0, [0, 1, 2]), (1, [0, 1, 2, 3]), (2, [0, 1, 2, 3, 4]), (3, [0, 1, 2, 3]), (4, [0, 1, 2])]))
    # Compact 3-4-3-1
    starts.append(_hex_grid_positions([(0, [0, 1, 2]), (1, [0, 1, 2, 3]), (2, [0, 1, 2]), (3, [0])]))
    # 4-3-4
    starts.append(_hex_grid_positions([(0, [0, 1, 2, 3]), (1, [0, 1, 2]), (2, [0, 1, 2, 3])]))
    # 5-3-3
    starts.append(_hex_grid_positions([(0, list(range(5))), (1, [0, 1, 2]), (2, [0, 1, 2])]))
    # 3-5-3
    starts.append(_hex_grid_positions([(0, [0, 1, 2]), (1, list(range(5))), (2, [0, 1, 2])]))
    # 4-4-3
    starts.append(_hex_grid_positions([(0, list(range(4))), (1, list(range(4))), (2, [0, 1, 2])]))
    # 2-4-5
    starts.append(_hex_grid_positions([(0, [0, 1]), (1, list(range(4))), (2, list(range(5)))]))
    # 1-5-5
    starts.append(_hex_grid_positions([(0, [0]), (1, list(range(5))), (2, list(range(5)))]))
    # 4-5-2
    starts.append(_hex_grid_positions([(0, list(range(4))), (1, list(range(5))), (2, [0, 1])]))
    # 3-3-3-2
    starts.append(_hex_grid_positions([(0, [0, 1, 2]), (1, [0, 1, 2]), (2, [0, 1, 2]), (3, [0, 1])]))
    # 2-3-3-3
    starts.append(_hex_grid_positions([(0, [0, 1]), (1, [0, 1, 2]), (2, [0, 1, 2]), (3, [0, 1, 2])]))
    # 5-4-2
    starts.append(_hex_grid_positions([(0, list(range(5))), (1, list(range(4))), (2, [0, 1])]))
    # 3-4-4
    starts.append(_hex_grid_positions([(0, [0, 1, 2]), (1, list(range(4))), (2, list(range(4)))]))
    # 2-5-4
    starts.append(_hex_grid_positions([(0, [0, 1]), (1, list(range(5))), (2, list(range(4)))]))
    # 1-6-4
    starts.append(_hex_grid_positions([(0, [0]), (1, list(range(6))), (2, list(range(4)))]))
    # 3-3-5
    starts.append(_hex_grid_positions([(0, [0, 1, 2]), (1, [0, 1, 2]), (2, list(range(5)))]))
    # 4-3-4
    starts.append(_hex_grid_positions([(0, list(range(4))), (1, [0, 1, 2]), (2, list(range(4)))]))
    # 5-2-4
    starts.append(_hex_grid_positions([(0, list(range(5))), (1, [0, 1]), (2, list(range(4)))]))
    # 2-4-5 shifted
    starts.append(_hex_grid_positions([(0, [0, 1]), (1, list(range(4))), (2, list(range(5)))]))
    # 3-5-3 shifted
    starts.append(_hex_grid_positions([(0, [0, 1, 2]), (1, list(range(5))), (2, [0, 1, 2])]))
    # 4-4-3 different
    starts.append(_hex_grid_positions([(0, list(range(4))), (1, list(range(4))), (2, [0, 1, 2])]))
    # 5-3-3 shifted
    starts.append(_hex_grid_positions([(0, list(range(5))), (1, [0, 1, 2]), (2, [0, 1, 2])]))
    # 3-4-4 different
    starts.append(_hex_grid_positions([(0, [0, 1, 2]), (1, list(range(4))), (2, list(range(4)))]))
    # 2-5-4 shifted
    starts.append(_hex_grid_positions([(0, [0, 1]), (1, list(range(5))), (2, list(range(4)))]))
    # 1-6-4 shifted
    starts.append(_hex_grid_positions([(0, [0]), (1, list(range(6))), (2, list(range(4)))]))
    # 3-3-5 shifted
    starts.append(_hex_grid_positions([(0, [0, 1, 2]), (1, [0, 1, 2]), (2, list(range(5)))]))
    
    # Add random perturbations to some starts for diversity
    rng = np.random.default_rng(42)
    for i in range(min(10, len(starts))):
        perturbed = starts[i].copy()
        perturbed += rng.normal(0, 0.3, size=perturbed.shape)
        starts.append(perturbed)
    
    return starts

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    starts = _generate_lattice_starts()
    
    best_S = float('inf')
    best_params = None
    
    # Quick score to sort starts
    def quick_score(pts):
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        return _calc_S(pts, cx, cy, 0.0)
    
    scored = sorted(starts, key=quick_score)
    
    for start_pts in scored:
        if not _time_ok():
            break
        
        cx = float(np.mean(start_pts[:, 0]))
        cy = float(np.mean(start_pts[:, 1]))
        
        # Try multiple beta initial values
        for beta_init in [0.0, math.pi/12, math.pi/6]:
            if not _time_ok():
                break
            
            init_params = np.concatenate([start_pts.ravel(), [cx, cy, beta_init]])
            
            try:
                res = minimize(_objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                
                if res.fun < best_S:
                    n = N
                    candidate_pts = res.x[:2*n].reshape(n, 2)
                    candidate_cx = res.x[2*n]
                    candidate_cy = res.x[2*n+1]
                    candidate_beta = ((res.x[2*n+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    
                    # Check if valid (no overlap)
                    pen = _overlap_penalty(candidate_pts, w=1.0)
                    if pen < 1e-6:
                        S_exact = _calc_S(candidate_pts, candidate_cx, candidate_cy, candidate_beta)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_params = (candidate_pts, candidate_cx, candidate_cy, candidate_beta)
            except Exception:
                pass
    
    # If no valid solution found, use the best lattice start
    if best_params is None:
        start_pts = scored[0]
        cx = float(np.mean(start_pts[:, 0]))
        cy = float(np.mean(start_pts[:, 1]))
        beta = 0.0
        best_S = _calc_S(start_pts, cx, cy, beta)
        best_params = (start_pts, cx, cy, beta)
    
    pts, cx, cy, beta = best_params
    angle_deg = math.degrees(beta)
    
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
