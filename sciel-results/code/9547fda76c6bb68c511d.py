"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place 11 unit hexagons in a compact honeycomb arrangement
    sqrt3 = np.sqrt(3.0)
    
    # Honeycomb arrangement: rows of 3, 3, 3, 2
    inner = []
    rows = [
        (3, 0, 0.0),
        (3, 1, sqrt3/2),
        (3, 2, 0.0),
        (2, 3, sqrt3/2),
    ]
    
    for count, row_idx, x_off in rows:
        y = -row_idx * 1.5
        for i in range(count):
            x = x_off + (i - (count - 1) / 2.0) * sqrt3
            inner.append([x, y, 0.0])
    
    # Get all vertices of all inner hexagons
    all_points = []
    for cx, cy, ang in inner:
        theta = np.radians(ang)
        for k in range(6):
            phi = theta + np.radians(60 * k)
            vx = cx + np.cos(phi)
            vy = cy + np.sin(phi)
            all_points.append((vx, vy))
    
    # Function to compute minimum side length for given center and outer hex angle
    def min_side(ox, oy, angle_deg):
        theta = np.radians(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        max_proj = 0.0
        for px, py in all_points:
            dx = px - ox
            dy = py - oy
            cx = dx * cos_t + dy * sin_t
            cy = -dx * sin_t + dy * cos_t
            for k in range(6):
                phi = np.radians(60 * k)
                proj = cx * np.cos(phi) + cy * np.sin(phi)
                if proj > max_proj:
                    max_proj = proj
        if max_proj <= 0:
            return 0.0
        return max_proj / (sqrt3 / 2.0)
    
    # Optimize: try several outer hexagon angles and find best center
    best_S = float('inf')
    best_ox = 0.0
    best_oy = 0.0
    best_angle = 0.0
    
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_c0 = (min(xs) + max(xs)) / 2.0
    y_c0 = (min(ys) + max(ys)) / 2.0
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    
    for angle_deg in [0.0, 30.0]:
        x_c = x_c0
        y_c = y_c0
        for it in range(3):
            step = spread / (10.0 * (it + 1))
            found = False
            for i in range(-15, 16):
                for j in range(-15, 16):
                    ox = x_c + i * step
                    oy = y_c + j * step
                    S = min_side(ox, oy, angle_deg)
                    if S < best_S:
                        best_S = S
                        best_ox = ox
                        best_oy = oy
                        best_angle = angle_deg
                        x_c = ox
                        y_c = oy
                        found = True
            if not found:
                break
    
    return inner, (best_ox, best_oy), best_S, best_angle
# EVOLVE-BLOCK-END
