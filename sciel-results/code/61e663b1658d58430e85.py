import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon using half-plane tests."""
    angle = np.radians(angle_deg)
    # The 6 inward-facing normals for a regular hexagon
    for i in range(6):
        theta = angle + i * np.pi / 3
        # Normal pointing inward from edge i
        nx = -np.sin(theta)
        ny = np.cos(theta)
        # Edge midpoint is at distance (sqrt(3)/2)*side from center along direction theta
        ex = cx + (np.sqrt(3)/2) * side * np.cos(theta)
        ey = cy + (np.sqrt(3)/2) * side * np.sin(theta)
        # Distance from point to edge plane
        d = (px - ex) * nx + (py - ey) * ny
        if d > 1e-10:
            return False
    return True

def max_distance_to_container(px, py, angle_deg, side, cx, cy):
    """
    Return the maximum distance by which a unit hexagon centered at (px,py) with rotation angle_deg
    extends beyond the container hexagon centered at (cx,cy) with side length 'side'.
    Returns 0 if fully inside, positive if extending out.
    """
    # Container hexagon vertices
    container_verts = hex_vertices(cx, cy, side, 0.0)
    # Inner hexagon vertices
    inner_verts = hex_vertices(px, py, 1.0, angle_deg)
    
    max_violation = 0.0
    for vx, vy in inner_verts:
        # Check if vertex is outside container
        if not point_in_hexagon(vx, vy, cx, cy, side, 0.0):
            # Find distance to container boundary
            # Compute distance to each edge of container
            min_dist = float('inf')
            for i in range(6):
                theta = i * np.pi / 3
                nx = -np.sin(theta)
                ny = np.cos(theta)
                ex = cx + (np.sqrt(3)/2) * side * np.cos(theta)
                ey = cy + (np.sqrt(3)/2) * side * np.sin(theta)
                d = (vx - ex) * nx + (vy - ey) * ny
                if d > 0:
                    min_dist = min(min_dist, d)
            max_violation = max(max_violation, min_dist)
    return max_violation

def hexagon_overlap_violation(cx1, cy1, a1, cx2, cy2, a2):
    """
    Return a measure of overlap between two unit hexagons.
    Uses sampling of vertices and midpoints.
    """
    verts1 = hex_vertices(cx1, cy1, 1.0, a1)
    verts2 = hex_vertices(cx2, cy2, 1.0, a2)
    
    # Check if any vertex of hex1 is inside hex2 or vice versa
    violation = 0.0
    for vx, vy in verts1:
        if point_in_hexagon(vx, vy, cx2, cy2, 1.0, a2):
            violation = max(violation, 0.1)
    for vx, vy in verts2:
        if point_in_hexagon(vx, vy, cx1, cy1, 1.0, a1):
            violation = max(violation, 0.1)
    
    # Also check center-to-center distance as a rough proxy
    dx = cx1 - cx2
    dy = cy1 - cy2
    dist = np.sqrt(dx*dx + dy*dy)
    if dist < 2.0:
        violation = max(violation, 2.0 - dist)
    
    return violation

def objective(params, side, cx, cy):
    """
    Objective: minimize side length subject to constraints.
    We use a penalty approach: minimize side + penalty * (max violations)
    """
    n = len(params) // 3
    total_violation = 0.0
    
    # Check each inner hexagon against container
    for i in range(n):
        px = params[3*i]
        py = params[3*i+1]
        ang = params[3*i+2]
        v = max_distance_to_container(px, py, ang, side, cx, cy)
        total_violation += v
    
    # Check pairwise overlaps
    for i in range(n):
        for j in range(i+1, n):
            px1 = params[3*i]
            py1 = params[3*i+1]
            a1 = params[3*i+2]
            px2 = params[3*j]
            py2 = params[3*j+1]
            a2 = params[3*j+2]
            v = hexagon_overlap_violation(px1, py1, a1, px2, py2, a2)
            total_violation += v
    
    return side + 10.0 * total_violation

def construct():
    start_time = time.time()
    
    # Initial guess: hexagonal lattice arrangement
    # Place hexagons in a 3-4-3 or similar pattern
    # For 11 hexagons, try a hexagonal cluster pattern
    
    # Start with a reasonable initial configuration
    # Center hexagon at origin, others around it
    initial_params = []
    
    # A common tight packing for hexagons is a hexagonal lattice
    # Let's try placing them in rows
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon (or adjust)
    
    # Actually, let's use a more systematic initial guess
    # Place in a grid-like pattern with hexagonal offset
    
    spacing = 2.0  # minimum center distance
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice positions for 11 hexagons
    # Using axial coordinates (q, r) -> (x, y)
    # x = spacing * (q + r/2)
    # y = spacing * sqrt3/2 * r
    
    positions = []
    # Center at (0,0)
    positions.append((0, 0))
    # 6 neighbors at distance 2
    for i in range(6):
        angle = i * np.pi / 3
        positions.append((2.0 * np.cos(angle), 2.0 * np.sin(angle)))
    # 4 more at distance 2*sqrt(3) in certain directions
    for i in range(0, 6, 3):
        angle = i * np.pi / 3 + np.pi/6
        positions.append((2.0*sqrt3 * np.cos(angle), 2.0*sqrt3 * np.sin(angle)))
    
    # Take first 11
    positions = positions[:11]
    
    for px, py in positions:
        initial_params.extend([px, py, 0.0])
    
    initial_params = np.array(initial_params)
    
    best_side = 15.0
    best_params = initial_params.copy()
    
    # Try different initial side lengths and optimize
    side_candidates = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    for side in side_candidates:
        if time.time() - start_time > DEADLINE - 30:
            break
            
        # Optimize positions for this side length
        result = minimize(
            objective,
            initial_params,
            args=(side, 0.0, 0.0),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Check if solution is valid
        final_violation = 0.0
        n = len(result.x) // 3
        for i in range(n):
            px = result.x[3*i]
            py = result.x[3*i+1]
            ang = result.x[3*i+2]
            v = max_distance_to_container(px, py, ang, side, 0.0, 0.0)
            final_violation += v
        for i in range(n):
            for j in range(i+1, n):
                v = hexagon_overlap_violation(
                    result.x[3*i], result.x[3*i+1], result.x[3*i+2],
                    result.x[3*j], result.x[3*j+1], result.x[3*j+2]
                )
                final_violation += v
        
        if final_violation < 1e-6:
            if side < best_side:
                best_side = side
                best_params = result.x.copy()
    
    # If we found a valid solution, try to tighten it further
    if best_side < 15.0:
        # Binary search for minimum valid side
        lo, hi = best_side * 0.9, best_side * 1.1
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 10:
                break
            mid = (lo + hi) / 2
            result = minimize(
                objective,
                best_params,
                args=(mid, 0.0, 0.0),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            
            final_violation = 0.0
            n = len(result.x) // 3
            for i in range(n):
                px = result.x[3*i]
                py = result.x[3*i+1]
                ang = result.x[3*i+2]
                v = max_distance_to_container(px, py, ang, mid, 0.0, 0.0)
                final_violation += v
            for i in range(n):
                for j in range(i+1, n):
                    v = hexagon_overlap_violation(
                        result.x[3*i], result.x[3*i+1], result.x[3*i+2],
                        result.x[3*j], result.x[3*j+1], result.x[3*j+2]
                    )
                    final_violation += v
            
            if final_violation < 1e-6:
                hi = mid
                best_params = result.x.copy()
            else:
                lo = mid
        
        best_side = hi
    
    # Build inner hexagon list
    inner = []
    n = len(best_params) // 3
    for i in range(n):
        px = best_params[3*i]
        py = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([px, py, ang])
    
    return inner, (0.0, 0.0), best_side, 0.0
