"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    # Hexagonal lattice for edge-sharing unit hexagons
    # Axial to Cartesian (flat-top): x = 1.5*q, y = sqrt(3)*(r + q/2)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = math.sqrt(3) * (r + 0.5 * q)
        return x, y
    
    # 11 hexagons: center + 6 ring1 + 4 ring2 (symmetric)
    positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (1, -2), (-1, -1), (-2, 0),
    ]
    
    centers = [axial_to_xy(q, r) for q, r in positions_axial]
    
    # Compute all vertices of all hexagons
    all_points = []
    for cx, cy in centers:
        for k in range(6):
            vx = cx + math.cos(k * math.pi / 3)
            vy = cy + math.sin(k * math.pi / 3)
            all_points.append((vx, vy))
    
    # Centroid of all vertices
    cx_avg = sum(p[0] for p in all_points) / len(all_points)
    cy_avg = sum(p[1] for p in all_points) / len(all_points)
    
    def min_enclosing_hex_side(points, rotation=0.0):
        cos_r = math.cos(-rotation)
        sin_r = math.sin(-rotation)
        max_val = 0.0
        for x, y in points:
            rx = x * cos_r - y * sin_r
            ry = x * sin_r + y * cos_r
            for i in range(6):
                theta = i * math.pi / 3
                val = rx * math.cos(theta) + ry * math.sin(theta)
                if val > max_val:
                    max_val = val
        return max_val
    
    best_side = float('inf')
    best_rotation = 0.0
    
    for rot_deg in range(0, 61, 5):
        rot = math.radians(rot_deg)
        translated = [(x - cx_avg, y - cy_avg) for x, y in all_points]
        side = min_enclosing_hex_side(translated, rotation=rot)
        if side < best_side:
            best_side = side
            best_rotation = rot
    
    outer_center = (cx_avg, cy_avg)
    outer_side = best_side
    outer_angle = math.degrees(best_rotation)
    
    inner = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END

