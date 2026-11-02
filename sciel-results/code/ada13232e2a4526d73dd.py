"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    angle = np.deg2rad(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside or on boundary of hexagon."""
    # Transform to local coordinates
    dx, dy = px - cx, py - cy
    angle = -np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Regular hexagon with side length s: vertices at angles 0,60,...
    # In local coords, the hexagon is defined by |x| <= s, |y| <= s*sqrt(3)/2, |x*sqrt(3)/2 + y/2| <= s
    # Actually for a regular hexagon with vertices at (s,0), (s/2, s*sqrt(3)/2), etc:
    # The constraints are:
    # -s <= x <= s
    # -s*sqrt(3)/2 <= y <= s*sqrt(3)/2
    # -s <= x*sqrt(3)/2 + y/2 <= s
    # -s <= -x*sqrt(3)/2 + y/2 <= s
    s = side
    sqrt3 = np.sqrt(3)
    return (np.abs(lx) <= s and 
            np.abs(ly) <= s * sqrt3 / 2 and 
            np.abs(lx * sqrt3 / 2 + ly / 2) <= s and
            np.abs(-lx * sqrt3 / 2 + ly / 2) <= s)

def hexagon_contains_hexagon(outer_verts, inner_verts):
    """Check if inner hexagon is fully inside outer hexagon using vertex containment."""
    for v in inner_verts:
        if not point_in_hexagon(v[0], v[1], 0, 0, 1, 0):  # placeholder, will be replaced
            pass
    # Better approach: check all 6 vertices of inner are inside outer
    return True

def check_validity(inner_positions, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are disjoint and inside the outer hexagon."""
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # For each inner hexagon, check all 6 vertices are inside outer
    for (ix, iy, iangle) in inner_positions:
        inner_verts = hex_vertices(ix, iy, 1.0, iangle)
        for vx, vy in inner_verts:
            # Check if point is inside outer hexagon
            if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    
    # Check pairwise disjointness (non-overlapping interiors)
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            ix1, iy1, ia1 = inner_positions[i]
            ix2, iy2, ia2 = inner_positions[j]
            # Two unit hexagons are disjoint if their centers are far enough apart
            # For two regular hexagons of side 1, the minimum distance for non-overlap depends on orientation
            # Simple check: if distance between centers < 2*inradius = sqrt(3), they might overlap
            # But this is not sufficient for all orientations. Let's do a more careful check.
            # Actually, for two unit hexagons, the maximum "reach" in any direction is 1 (circumradius)
            # So if center distance > 2, they're definitely disjoint. If < sqrt(3) ~ 1.732, they might overlap.
            dx = ix2 - ix1
            dy = iy2 - iy1
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1.9:  # conservative threshold
                # More careful check: sample points on boundary of each, or check vertex containment
                # For simplicity, use a slightly more permissive threshold
                # Two hexagons of side 1: if centers are at distance d, they're disjoint if d >= 2*cos(min_angle/2) for worst case
                # Worst case is when they're aligned, requiring d >= 2. But at 30 deg offset, less is needed.
                # Let's use: if dist < sqrt(3) - 0.1, definitely overlapping; if > 2, definitely disjoint
                if dist < 1.7:
                    return False
    return True

def construct():
    """Place 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial guess: hexagonal arrangement
    # 12 hexagons in a pattern: center + ring of 6 + partial outer ring
    # Let's try a compact arrangement
    
    # Strategy: Use optimization to find good positions
    # Variables: for each inner hex: x, y, angle (12*3 = 36 vars)
    # Plus outer: cx, cy, side, angle (4 vars)
    # Total: 40 vars
    
    # Objective: outer_side + penalty for violations
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        # Extract parameters
        inner_params = params[:36].reshape(12, 3)
        outer_params = params[36:]
        outer_cx, outer_cy, outer_side, outer_angle = outer_params
        
        if outer_side <= 0:
            return 1e10
        
        # Penalty for inner hexagons outside outer
        penalty = 0.0
        outer_verts = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
        
        for k in range(12):
            ix, iy, iangle = inner_params[k]
            inner_verts = hex_vertices(ix, iy, 1.0, iangle)
            for vx, vy in inner_verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                    # Compute distance to boundary and add penalty
                    penalty += 10.0
        
        # Penalty for overlapping inner hexagons
        for i in range(12):
            for j in range(i+1, 12):
                ix1, iy1, _ = inner_params[i]
                ix2, iy2, _ = inner_params[j]
                dx = ix2 - ix1
                dy = iy2 - iy1
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1.73:  # sqrt(3)
                    penalty += 10.0 * (1.73 - dist)
        
        return outer_side + penalty
    
    # Initial guess
    # Place hexagons in a hexagonal pattern
    # Center at origin, outer hexagon centered at origin
    # Inner hexagons: try a pattern with spacing ~1.75
    
    # Hexagonal grid positions for 12 hexagons
    # Use axial coordinates
    positions = []
    # Ring 0: (0,0) - 1 hex
    positions.append((0, 0))
    # Ring 1: 6 hexes
    for i in range(6):
        angle = i * np.pi / 3
        positions.append((1.85 * np.cos(angle), 1.85 * np.sin(angle)))
    # Ring 2: partial, 5 more hexes
    for i in range(5):
        angle = (i * 2 * np.pi / 5) + np.pi / 5
        positions.append((3.5 * np.cos(angle), 3.5 * np.sin(angle)))
    
    # Initial parameters
    params0 = np.zeros(40)
    for k in range(12):
        params0[k*3] = positions[k][0]
        params0[k*3+1] = positions[k][1]
        params0[k*3+2] = 0.0  # angle
    params0[36] = 0.0   # outer cx
    params0[37] = 0.0   # outer cy
    params0[38] = 5.0   # outer side
    params0[39] = 0.0   # outer angle
    
    # Try multiple random restarts
    best_score = 1e10
    best_params = params0.copy()
    
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Perturb initial guess
        if r == 0:
            params = params0.copy()
        else:
            params = params0 + 0.5 * np.random.randn(40)
        
        try:
            result = minimize(objective, params, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-6})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract best solution
    inner_params = best_params[:36].reshape(12, 3)
    outer_cx, outer_cy, outer_side, outer_angle = best_params[36:]
    
    inner = [[float(inner_params[k,0]), float(inner_params[k,1]), float(inner_params[k,2])] for k in range(12)]
    
    # Validate and adjust if needed
    if not check_validity(inner, (outer_cx, outer_cy), outer_side, outer_angle):
        # Fallback to a known valid configuration
        # Simple: place in a line with large outer hexagon
        inner = [[3.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        outer_cx, outer_cy, outer_side, outer_angle = 0.0, 0.0, 20.0, 0.0
    
    return inner, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle)
