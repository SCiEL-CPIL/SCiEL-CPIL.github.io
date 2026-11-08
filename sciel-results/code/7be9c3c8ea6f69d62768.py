import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 145.0
MIN_DIST = np.sqrt(3.0) # Still used for initial heuristic, but SAT is the truth

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy)."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def min_outer_side_for_points(points, outer_center, outer_angle_deg):
    """
    Compute minimum side length s of a regular hexagon centered at outer_center
    with rotation outer_angle_deg that contains all given points.
    """
    cx, cy = outer_center
    angle_rad = np.radians(outer_angle_deg)
    normal_angles = angle_rad + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    rel_points = points - np.array([cx, cy])
    dots = rel_points @ normals.T
    max_dot = np.max(dots, axis=0)
    r_needed = np.max(max_dot)
    if r_needed < 0:
        r_needed = 0.0
    s = 2.0 / np.sqrt(3.0) * r_needed
    return s

def compute_min_side(inner, outer_center, outer_angle_deg):
    """Compute minimum outer side length for given inner configuration."""
    all_points = []
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        all_points.append(v)
    all_points = np.vstack(all_points)
    return min_outer_side_for_points(all_points, outer_center, outer_angle_deg)

def get_axes(polygon):
    """Get the normal axes for a convex polygon."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1.0
    return normals / norms[:, np.newaxis]

def project(polygon, axis):
    """Project polygon onto axis."""
    dots = polygon @ axis
    return np.min(dots), np.max(dots)

def polygons_overlap(poly1, poly2):
    """Check if two convex polygons overlap using Separating Axis Theorem."""
    axes = np.vstack([get_axes(poly1), get_axes(poly2)])
    for axis in axes:
        min1, max1 = project(poly1, axis)
        min2, max2 = project(poly2, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True

def check_disjointness(inner):
    """Check if all inner hexagons are disjoint."""
    polys = [hexagon_vertices(x, y, 1.0, a) for x, y, a in inner]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap(polys[i], polys[j]):
                return False
    return True

def objective(params):
    """
    Objective: minimize outer side length + penalty for overlaps.
    params: 24 values [x1,y1,...,x12,y12]
    Assumes all inner hexagons have angle 0.
    """
    pos = params.reshape(N, 2)
    
    # Calculate outer side length
    all_points = []
    for i in range(N):
        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0)
        all_points.append(v)
    all_points = np.vstack(all_points)
    
    s = min_outer_side_for_points(all_points, (0.0, 0.0), 0.0)
    
    # Penalty for overlaps using SAT
    penalty = 0.0
    polys = [hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, 0.0) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if polygons_overlap(polys[i], polys[j]):
                # Calculate a penalty based on how much they overlap?
                # SAT doesn't give depth easily.
                # Use a fixed large penalty or distance-based.
                # For optimization, a smooth penalty is better.
                # Fallback to distance penalty for smoothness in objective, 
                # but use SAT for final validation.
                d = np.linalg.norm(pos[i] - pos[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2 * 100.0
                else:
                    penalty += 10.0 # They overlap but d is large? Unlikely for parallel.
    
    return s + penalty

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    def try_config(inner, center, angle):
        nonlocal best_score, best_inner, best_center, best_side, best_angle
        s = compute_min_side(inner, center, angle)
        if check_disjointness(inner):
            if s < best_score:
                best_score = s
                best_inner = [list(x) for x in inner]
                best_center = center
                best_side = s
                best_angle = angle

    sqrt3 = np.sqrt(3)
    
    # Initial Configurations
    
    # 1. Hexagonal Grid (Tighter)
    # Centers at distance sqrt3
    positions = []
    # Generate a hex grid of side 3 (19 hexes) and take 12
    coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            coords.append((q, r, x, y))
            
    coords.sort(key=lambda c: c[2]**2 + c[3]**2)
    selected = coords[:12]
    pos3 = [(x, y) for q, r, x, y in selected]
    cx3 = np.mean([p[0] for p in pos3])
    cy3 = np.mean([p[1] for p in pos3])
    inner_ring = [[p[0] - cx3, p[1] - cy3, 0.0] for p in pos3]
    try_config(inner_ring, (0.0, 0.0), 0.0)

    # Optimization
    if best_inner is not None:
        init_pos = np.array([[x, y] for x, y, a in best_inner])
        
        n_restarts = 10
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            if r == 0:
                x0 = init_pos.flatten()
            else:
                scale = 0.5 * np.sqrt(r)
                x0 = init_pos.flatten() + scale * np.random.randn(2*N)
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                pos_opt = result.x.reshape(N, 2)
                inner_opt = [[float(pos_opt[i, 0]), float(pos_opt[i, 1]), 0.0] for i in range(N)]
                
                if check_disjointness(inner_opt):
                    s_opt = compute_min_side(inner_opt, (0.0, 0.0), 0.0)
                    if s_opt < best_score:
                        best_score = s_opt
                        best_inner = inner_opt
                        best_center = (0.0, 0.0)
                        best_side = s_opt
                        best_angle = 0.0
                        
                        # Try outer rotation
                        for outer_ang in [0, 30]:
                            s_rot = compute_min_side(inner_opt, (0.0, 0.0), float(outer_ang))
                            if s_rot < best_score:
                                best_score = s_rot
                                best_side = s_rot
                                best_angle = float(outer_ang)
                                
            except Exception:
                pass
    
    # Add safety margin
    best_side += 0.05
    
    return best_inner, best_center, best_side, best_angle
