"""Optimized hexagon packing: 11 unit hexagons using discrete lattice search and scipy optimization for the container."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _min_outer_side(hexes, cx, cy, angle_deg):
    """
    Calculate the minimum side length of a regular hexagon centered at (cx, cy)
    with rotation angle_deg that contains all hexagons in `hexes`.
    `hexes` is a list of (x, y, side, angle).
    """
    # Precompute vertices for all inner hexagons
    all_verts = []
    for hx, hy, hs, ha in hexes:
        all_verts.extend(_hex_vertices(hx, hy, hs, ha))
    
    if not all_verts:
        return 0.0
        
    # Convert to numpy array for speed
    pts = np.array(all_verts)
    
    # Normal vectors for the 6 sides of the outer hexagon
    # The apothem direction is 30 + 60k
    a_rad = math.radians(angle_deg)
    cos_a = math.cos(a_rad)
    sin_a = math.sin(a_rad)
    
    max_proj = 0.0
    # We can vectorize this
    # Directions of the 6 normals
    thetas = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack((np.cos(thetas), np.sin(thetas)))
    
    # Relative positions
    rel_pts = pts - np.array([cx, cy])
    
    # Project onto each normal
    # rel_pts is (N, 2), normals is (6, 2)
    # Projections: (N, 6)
    projs = rel_pts @ normals.T
    
    # The required apothem is the max projection
    apothem = np.max(projs)
    
    # Side length = 2 * apothem / sqrt(3)
    return apothem * 2.0 / math.sqrt(3)

def _optimize_container(hexes, init_cx, init_cy, init_angle):
    """
    Optimize the outer hexagon (center and angle) to minimize side length.
    Uses scipy.optimize.minimize.
    """
    def objective(params):
        cx, cy, angle = params
        return _min_outer_side(hexes, cx, cy, angle)
    
    # Initial guess
    x0 = [init_cx, init_cy, init_angle]
    
    # Nelder-Mead is good for small dimensional non-smooth problems
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000, 'maxfev': 1000})
    
    return res.x[0], res.x[1], res.x[2], res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    sqrt3 = math.sqrt(3)

    # Axial coordinates for flat-top hexagons
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Center + Ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (12 hexagons)
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try all combinations of 4 from ring2
    # C(12, 4) = 495
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Try a few initial angles for the container
        # The symmetry of the cluster might suggest 0 or 30 degrees
        initial_candidates = [
            (cx_mean, cy_mean, 0.0),
            (cx_mean, cy_mean, 30.0),
        ]
        
        for ocx, ocy, oang in initial_candidates:
            # Quick check to avoid expensive optimization if clearly bad
            # But Nelder-Mead is fast enough for 495 * 2 cases?
            # 495 * 2 * ~1000 evaluations * ~0.01ms/eval = 10 seconds.
            # This is acceptable.
            
            opt_cx, opt_cy, opt_angle, opt_side = _optimize_container(
                inner_hexes, ocx, ocy, oang
            )
            
            if opt_side < best_side:
                best_side = opt_side
                best_positions = full_xy
                best_center = (opt_cx, opt_cy)
                best_angle = opt_angle

    if best_positions is None:
        # Fallback
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        cx_mean = sum(x for x, y in best_positions) / 7.0
        cy_mean = sum(y for x, y in best_positions) / 7.0
        best_cx, best_cy, best_angle, best_side = _optimize_container(
            inner_hexes, cx_mean, cy_mean, 0.0
        )
        best_center = (best_cx, best_cy)

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
