"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Place 12 unit hexagons on a hexagonal lattice (flat-top orientation)
    # Hexagonal lattice spacing for flat-top hexagons:
    # Horizontal spacing: 3/2, Vertical spacing: sqrt(3)
    # Odd rows offset by 3/4
    
    hex_positions = []
    # Use a 4x3 grid arrangement on hex lattice
    # Rows: 3 rows, Cols: 4 cols
    for row in range(3):
        for col in range(4):
            x = col * 1.5 + (row % 2) * 0.75
            y = row * np.sqrt(3)
            hex_positions.append((x, y))
    
    # Center the arrangement
    xs = [p[0] for p in hex_positions]
    ys = [p[1] for p in hex_positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    hex_positions = [(x - cx, y - cy) for x, y in hex_positions]
    
    # Now find minimal enclosing regular hexagon
    # A regular hexagon with center (hx, hy), side length s, rotation theta
    # contains point (px, py) if the point is within all 6 half-planes
    # The 6 vertices are at angles theta + k*60 degrees, distance s from center
    # The 6 sides are at distance s * sqrt(3)/2 from center (the apothem)
    
    def hexagon_contains_point(px, py, hx, hy, s, theta):
        # The hexagon has 6 sides. Each side is a line at distance apothem = s*sqrt(3)/2
        # from center, with normal direction at angle theta + k*60 + 30 degrees
        # (the normals are at the midpoints of the sides)
        apothem = s * np.sqrt(3) / 2
        for k in range(6):
            normal_angle = theta + np.radians(k * 60 + 30)
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            # Distance from point to the side line (signed)
            dist = (px - hx) * nx + (py - hy) * ny
            if dist > apothem + 1e-9:
                return False
        return True
    
    def violation(hx_hy_s_theta, hex_pts):
        hx, hy, s, theta = hx_hy_s_theta
        if s <= 0:
            return 1e6
        max_violation = 0
        for px, py in hex_pts:
            apothem = s * np.sqrt(3) / 2
            for k in range(6):
                normal_angle = theta + np.radians(k * 60 + 30)
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                dist = (px - hx) * nx + (py - hy) * ny
                viol = dist - apothem
                if viol > max_violation:
                    max_violation = viol
        return max_violation
    
    def objective(params):
        hx, hy, s, theta = params
        if s <= 0:
            return 1e6
        # We want to minimize s such that all points are inside
        # Use violation as constraint
        max_viol = 0
        for px, py in hex_positions:
            apothem = s * np.sqrt(3) / 2
            for k in range(6):
                normal_angle = theta + np.radians(k * 60 + 30)
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                dist = (px - hx) * nx + (py - hy) * ny
                viol = dist - apothem
                if viol > max_viol:
                    max_viol = viol
        if max_viol > 0:
            return s + 10 * max_viol
        return s
    
    # Try multiple initial guesses for theta
    best_result = None
    best_s = float('inf')
    
    for theta_init in [0, 30, 60, 90, 120, 150]:
        theta_init_rad = np.radians(theta_init)
        # Initial guess for center
        hx0, hy0 = 0.0, 0.0
        # Initial guess for side length - estimate from bounding box
        xs = [p[0] for p in hex_positions]
        ys = [p[1] for p in hex_positions]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        s0 = max(width, height) / np.sqrt(3) + 1.0
        
        x0 = [hx0, hy0, s0, theta_init_rad]
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < best_s:
            hx, hy, s, theta = res.x
            if s > 0 and violation(res.x, hex_positions) < 1e-6:
                if s < best_s:
                    best_s = s
                    best_result = (hx, hy, s, theta)
    
    if best_result is None:
        # Fallback: use a large hexagon
        return [(x, y, 0.0) for x, y in hex_positions], (0.0, 0.0), 10.0, 0.0
    
    hx, hy, s, theta = best_result
    theta_deg = np.degrees(theta) % 360
    
    inner = [(x, y, 0.0) for x, y in hex_positions]
    return inner, (hx, hy), s, theta_deg
# EVOLVE-BLOCK-END
