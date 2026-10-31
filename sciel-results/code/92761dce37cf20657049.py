"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    # Vertices of unit hexagon: at angles 0, 60, 120, 180, 240, 300 degrees from center
    # For a regular hexagon with side length s, the circumradius = s
    angles = angle_rad + np.pi/6 + np.arange(6) * np.pi/3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
    # Transform to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    # Rotate point by -angle to align hexagon
    ca, sa = np.cos(-angle_rad), np.sin(-angle_rad)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # In local coords, hexagon with circumradius = side has vertices at angles 30, 90, 150, 210, 270, 330
    # The hexagon is the intersection of 6 half-planes
    # For a hexagon with circumradius R=side, the apothem = R * cos(pi/6) = R * sqrt(3)/2
    # The 6 face normals are at angles 0, 60, 120, 180, 240, 300 (perpendicular to edges)
    # Actually, let's use the standard form: |x| <= apothem, |x*cos60 + y*sin60| <= apothem, |x*cos60 - y*sin60| <= apothem
    apothem = side * np.sqrt(3) / 2
    # The three pairs of parallel sides give constraints:
    # |lx| <= apothem
    # |lx * cos(60) + ly * sin(60)| <= apothem  => |lx*0.5 + ly*sqrt(3)/2| <= apothem
    # |lx * cos(60) - ly * sin(60)| <= apothem  => |lx*0.5 - ly*sqrt(3)/2| <= apothem
    c1 = abs(lx)
    c2 = abs(lx * 0.5 + ly * np.sqrt(3)/2)
    c3 = abs(lx * 0.5 - ly * np.sqrt(3)/2)
    return max(c1, c2, c3) <= apothem

def _min_dist_to_hex_boundary(px, py, cx, cy, side, angle_deg):
    """Minimum distance from point to boundary of hexagon. Positive if inside, negative if outside."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    ca, sa = np.cos(-angle_rad), np.sin(-angle_rad)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    apothem = side * np.sqrt(3) / 2
    # Distance to each of the 6 sides (negative if outside)
    d1 = apothem - abs(lx)
    d2 = apothem - abs(lx * 0.5 + ly * np.sqrt(3)/2)
    d3 = apothem - abs(lx * 0.5 - ly * np.sqrt(3)/2)
    return min(d1, d2, d3)

def _hex_hex_min_center_dist(angle1_deg, angle2_deg):
    """Minimum center-to-center distance for two unit hexagons with given absolute rotations to not overlap."""
    # The Minkowski sum of two hexagons determines the exclusion zone
    # For simplicity, use a conservative estimate based on relative angle
    rel_angle = abs(angle1_deg - angle2_deg) % 60
    rel_angle = min(rel_angle, 60 - rel_angle)
    # When aligned (rel=0), min dist = 2
    # When 30 deg apart, min dist = sqrt(3)
    # Linear interpolation as approximation
    t = rel_angle / 30.0
    return 2.0 * (1 - t) + np.sqrt(3) * t

def _objective(params, deadline):
    """Compute objective: outer side + penalties. Returns (value, outer_side, valid)."""
    if time.time() > deadline:
        return (float('inf'), 0, False)
    
    n = N
    # params layout: [x0,y0,a0, x1,y1,a1, ..., x_{n-1},y_{n-1},a_{n-1}, cx, cy, side, outer_angle]
    # Total: 3n + 4 parameters
    inner = params[:3*n].reshape(n, 3)
    cx = params[3*n]
    cy = params[3*n+1]
    side = params[3*n+2]
    outer_angle = params[3*n+3]
    
    # Ensure side is positive
    if side < 0.5:
        return (float('inf'), side, False)
    
    # Compute minimum distance from each inner hexagon's center to outer hexagon boundary
    # We need all vertices of each inner hexagon to be inside the outer hexagon
    # Use a smooth penalty: for each inner hex vertex, compute signed distance to outer boundary
    # Sum of soft-max of violations
    
    violations = []
    for i in range(n):
        ix, iy, ia = inner[i]
        # Get vertices of inner hexagon
        verts = _hexagon_vertices(ix, iy, 1.0, ia)
        for v in verts:
            d = _min_dist_to_hex_boundary(v[0], v[1], cx, cy, side, outer_angle)
            violations.append(-d)  # positive if outside
    
    # Smooth max of violations using log-sum-exp
    violations = np.array(violations)
    if len(violations) > 0:
        max_v = np.max(violations)
        if max_v > 0:
            log_sum_exp = max_v + np.log(np.sum(np.exp(violations - max_v)))
        else:
            log_sum_exp = 0.0
    else:
        log_sum_exp = 0.0
    
    # Overlap penalties: for each pair, check if centers are too close
    overlap_violations = []
    for i in range(n):
        for j in range(i+1, n):
            ix_i, iy_i, ia_i = inner[i]
            ix_j, iy_j, ia_j = inner[j]
            dist = np.sqrt((ix_i - ix_j)**2 + (iy_i - iy_j)**2)
            min_dist = _hex_hex_min_center_dist(ia_i, ia_j)
            overlap_violations.append(min_dist - dist)
    
    overlap_violations = np.array(overlap_violations)
    if len(overlap_violations) > 0:
        max_ov = np.max(overlap_violations)
        if max_ov > 0:
            log_sum_exp_ov = max_ov + np.log(np.sum(np.exp(overlap_violations - max_ov)))
        else:
            log_sum_exp_ov = 0.0
    else:
        log_sum_exp_ov = 0.0
    
    # Objective: side + weight * (boundary_violation + overlap_violation)
    weight = 10.0
    obj = side + weight * (log_sum_exp + log_sum_exp_ov)
    
    return (obj, side, True)

def _make_initial_guess():
    """Create a reasonable initial arrangement of 11 hexagons in a hexagonal cluster."""
    # Hexagonal cluster: center + 6 around + 4 more
    # Positions in a hex grid with spacing 2 (for non-overlapping unit hexagons)
    # Use axial coordinates for hex grid
    hex_positions = []
    # Center
    hex_positions.append((0, 0))
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        hex_positions.append((x, y))
    # Ring 2: add 4 more to make 11 total (we have 7, need 4 more)
    # Place them at some ring 2 positions
    for k in range(4):
        angle = k * np.pi / 3 + np.pi/6
        x = 2.0 * np.sqrt(3) * np.cos(angle)
        y = 2.0 * np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
    
    n = len(hex_positions)
    assert n == N, f"Expected {N}, got {n}"
    
    # Estimate outer hexagon size
    # Find bounding hexagon
    max_r = 0
    for x, y in hex_positions:
        r = np.sqrt(x*x + y*y)
        max_r = max(max_r, r)
    # Outer hexagon needs to contain all inner hexagons
    # Rough estimate: side ≈ max_r + 1 (circumradius of unit hex)
    side_est = max_r + 1.5
    
    # Build parameter vector
    params = []
    for x, y in hex_positions:
        params.extend([x, y, 0.0])
    params.extend([0.0, 0.0, side_est, 0.0])
    return np.array(params)

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Start with initial guess
    best_params = _make_initial_guess()
    best_obj, best_side, best_valid = _objective(best_params, deadline)
    
    # Multi-start Nelder-Mead optimization
    np.random.seed(42)
    n_starts = 8
    n_per_start = 0
    
    for start in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        
        if start == 0:
            x0 = best_params.copy()
        else:
            # Perturb initial guess
            x0 = _make_initial_guess().copy()
            # Add random perturbation
            perturbation = np.random.randn(len(x0)) * 0.5
            x0 = x0 + perturbation
            # Ensure side is positive
            x0[3*N+2] = max(x0[3*N+2], 1.0)
        
        # Nelder-Mead
        def obj_func(p):
            if time.time() > deadline:
                return float('inf')
            val, _, _ = _objective(p, deadline)
            return val
        
        try:
            result = minimize(obj_func, x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
        except Exception:
            continue
        
        n_per_start += 1
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
            best_side, _, best_valid = _objective(best_params, deadline)
    
    # Extract best solution
    inner = best_params[:3*N].reshape(N, 3)
    cx = best_params[3*N]
    cy = best_params[3*N+1]
    side = best_params[3*N+2]
    outer_angle = best_params[3*N+3]
    
    # Ensure side is reasonable
    if side < 1.0:
        side = 1.0
    
    # Format inner as list of [x, y, angle_degrees]
    inner_list = [[float(inner[i][0]), float(inner[i][1]), float(inner[i][2])] for i in range(N)]
    
    return inner_list, (float(cx), float(cy)), float(side), float(outer_angle)
# EVOLVE-BLOCK-END
