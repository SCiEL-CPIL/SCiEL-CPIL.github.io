import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 3.0

# Axial coordinate helpers
def axial_to_cartesian(q, r):
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return (x, y)

def generate_candidates():
    """Generate diverse sets of 11 non-overlapping hexagon centers in axial coordinates."""
    candidates = []
    
    # Configuration 1: Central hexagon + 6 neighbors + 4 more in a ring
    # Center (0,0), 6 neighbors, then 4 more at distance 2
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (2, -2)
    ]
    candidates.append(axial_1)
    
    # Configuration 2: 3x3 block + 2 extra
    axial_2 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (3, 1), (-1, 1)
    ]
    candidates.append(axial_2)
    
    # Configuration 3: 4 in a row + 3 above + 3 below + 1 extra
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, -1), (1, -1), (2, -1),
        (4, 0)
    ]
    candidates.append(axial_3)
    
    # Configuration 4: Cross shape
    axial_4 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_4)
    
    # Configuration 5: 5 in a row + 3 above + 3 below
    axial_5 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1),
        (1, -1), (2, -1), (3, -1)
    ]
    candidates.append(axial_5)
    
    # Configuration 6: 2 rows of 4 + 3 in a staggered row
    axial_6 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_6)
    
    # Configuration 7: Compact cluster
    axial_7 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_7)
    
    # Configuration 8: 3-4-3-1 pattern
    axial_8 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_8)
    
    # Configuration 9: 2-3-4-2 pattern
    axial_9 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_9)
    
    # Configuration 10: Diagonal arrangement
    axial_10 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
        (3, 2)
    ]
    candidates.append(axial_10)
    
    # Configuration 11: 1-2-3-2-1-2 pattern
    axial_11 = [
        (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4),
        (2, 1)
    ]
    candidates.append(axial_11)
    
    # Configuration 12: 4-3-4 pattern
    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_12)
    
    # Configuration 13: 2-4-3-2 pattern
    axial_13 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_13)
    
    # Configuration 14: 3-5-3 pattern
    axial_14 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_14)
    
    return candidates

def compute_support_function(pts, angle):
    """
    Compute the support function of the set of points in the given direction.
    The support function h(u) = max_i (p_i · u) where u is a unit vector.
    """
    ux = math.cos(angle)
    uy = math.sin(angle)
    projections = pts[:, 0] * ux + pts[:, 1] * uy
    return np.max(projections)

def compute_width(pts, angle):
    """
    Compute the width of the point set in the given direction.
    Width = max projection - min projection = h(u) + h(-u)
    """
    ux = math.cos(angle)
    uy = math.sin(angle)
    projections = pts[:, 0] * ux + pts[:, 1] * uy
    return np.max(projections) - np.min(projections)

def compute_min_enclosing_hexagon_side(pts, rotation):
    """
    Compute the side length of the minimal enclosing regular hexagon.
    
    The regular hexagon has 6 sides. The normals to the sides are at angles:
    rotation + pi/6 + k*pi/3 for k=0..5
    
    For each pair of opposite sides, the distance between them must be at least
    the width of the point set in that direction.
    
    The apothem (distance from center to side) is s * sqrt(3) / 2 where s is side length.
    The distance between opposite sides is 2 * apothem = s * sqrt(3).
    
    So for each direction k, we need:
    width_k <= s * sqrt(3)
    s >= width_k / sqrt(3)
    
    The minimal s is max_k (width_k / sqrt(3)).
    
    But we also need to center the hexagon properly. The center of the hexagon
    should be at the centroid of the projections for each direction pair.
    """
    widths = []
    for k in range(6):
        angle = rotation + math.pi / 6.0 + k * math.pi / 3.0
        w = compute_width(pts, angle)
        widths.append(w)
    
    max_width = max(widths)
    side_length = max_width / math.sqrt(3)
    
    return side_length

def compute_center_and_side(pts, rotation):
    """
    Compute the optimal center and side length for the enclosing hexagon.
    
    The center should be chosen to minimize the maximum distance to the sides.
    For each pair of opposite sides, the center's projection should be at the
    midpoint of the min and max projections.
    """
    # For each of the 6 directions, compute the midpoint of the projection range
    midpoints = []
    widths = []
    for k in range(6):
        angle = rotation + math.pi / 6.0 + k * math.pi / 3.0
        ux = math.cos(angle)
        uy = math.sin(angle)
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        min_p = np.min(projections)
        max_p = np.max(projections)
        midpoints.append((min_p + max_p) / 2.0)
        widths.append(max_p - min_p)
    
    # The center (cx, cy) should satisfy:
    # cx * ux_k + cy * uy_k = midpoint_k for all k
    # This is an overdetermined system, so we solve in least squares sense
    A = np.zeros((6, 2))
    b = np.zeros(6)
    for k in range(6):
        angle = rotation + math.pi / 6.0 + k * math.pi / 3.0
        A[k, 0] = math.cos(angle)
        A[k, 1] = math.sin(angle)
        b[k] = midpoints[k]
    
    # Least squares solution
    try:
        center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = center
    except:
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
    
    max_width = max(widths)
    side_length = max_width / math.sqrt(3)
    
    return (cx, cy), side_length

def optimize_rotation(pts):
    """
    Optimize the rotation angle to minimize the enclosing hexagon side length.
    """
    def objective(theta):
        return compute_min_enclosing_hexagon_side(pts, theta)
    
    # Coarse grid search
    best_theta = 0.0
    best_val = float('inf')
    for i in range(60):
        theta = i * math.pi / 6 / 60
        val = objective(theta)
        if val < best_val:
            best_val = val
            best_theta = theta
    
    # Refine with bounded minimization
    try:
        res = minimize_scalar(objective, bounds=(0, math.pi / 6), method='bounded',
                             options={'maxiter': 1000, 'xatol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_theta = res.x
    except:
        pass
    
    # Further refinement with Nelder-Mead
    try:
        res = minimize(objective, best_theta, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15})
        if res.fun < best_val:
            best_val = res.fun
            best_theta = res.x[0]
    except:
        pass
    
    return best_theta, best_val

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    candidates = generate_candidates()
    
    best_global_side = float('inf')
    best_global_theta = 0.0
    best_global_center = None
    best_global_pts = None
    
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Convert axial to cartesian
        pts = np.array([axial_to_cartesian(q, r) for q, r in axial])
        
        # Optimize rotation
        theta, side = optimize_rotation(pts)
        
        if side < best_global_side:
            best_global_side = side
            best_global_theta = theta
            best_global_pts = pts
            # Compute center
            best_global_center, _ = compute_center_and_side(pts, theta)
    
    if best_global_center is None or best_global_side == float('inf'):
        # Fallback
        axial = candidates[0]
        best_global_pts = np.array([axial_to_cartesian(q, r) for q, r in axial])
        best_global_theta = 0.0
        best_global_center, best_global_side = compute_center_and_side(best_global_pts, best_global_theta)
    
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_theta)
    
    # Apply a tiny scale to ensure no overlap
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recompute side with scaled points
    _, side_scaled = compute_center_and_side(pts_scaled, best_global_theta)
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(side_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
