"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    
    # Place 11 unit hexagons in a compact hexagonal arrangement using axial coordinates
    # Flat-top hexagons: x = 1.5*q, y = sqrt(3)*(r + q/2)
    hex_positions = [
        (0, 0),      # center
        (1, 0), (1, -1), (0, -1),  # 3 neighbors
        (-1, 0), (-1, 1), (0, 1),  # 3 more neighbors
        (2, -1),     # extend
        (2, -2),     # extend
        (-2, 1),     # extend
        (-2, 2),     # extend
    ]
    
    positions = []
    for q, r in hex_positions:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        positions.append((x, y))
    
    inner = [[x, y, 0.0] for x, y in positions]
    
    def hexagon_side_for_points(points, cx, cy, theta):
        """Find minimum side length s such that all points are inside hexagon."""
        max_val = 0.0
        for px, py in points:
            dx = px - cx
            dy = py - cy
            for k in range(6):
                angle = math.radians(theta + 30 + k * 60)
                nx = math.cos(angle)
                ny = math.sin(angle)
                val = dx * nx + dy * ny
                if val > max_val:
                    max_val = val
        if max_val <= 0:
            return 0.0
        return 2.0 * max_val / sqrt3
    
    def optimize_center(points, theta):
        """Find optimal center for given rotation."""
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        for _ in range(20):
            cx_new = 0.0
            cy_new = 0.0
            for k in range(6):
                angle = math.radians(theta + 30 + k * 60)
                nx = math.cos(angle)
                ny = math.sin(angle)
                max_proj = -float('inf')
                max_px, max_py = 0.0, 0.0
                for px, py in points:
                    proj = (px - cx) * nx + (py - cy) * ny
                    if proj > max_proj:
                        max_proj = proj
                        max_px, max_py = px, py
                cx_new += max_px
                cy_new += max_py
            cx_new /= 6.0
            cy_new /= 6.0
            cx = 0.5 * cx + 0.5 * cx_new
            cy = 0.5 * cy + 0.5 * cy_new
        return cx, cy
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_theta = 0.0
    
    # Try rotations
    for theta_deg in range(0, 180, 3):
        if time.time() > deadline:
            break
        theta = theta_deg
        cx, cy = optimize_center(positions, theta)
        side = hexagon_side_for_points(positions, cx, cy, theta)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_theta = theta
    
    # Fine-tune around best
    for theta_offset in range(-3, 4, 1):
        if time.time() > deadline:
            break
        theta = best_theta + theta_offset
        cx, cy = optimize_center(positions, theta)
        side = hexagon_side_for_points(positions, cx, cy, theta)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_theta = theta
    
    return inner, best_center, best_side, best_theta
# EVOLVE-BLOCK-END
