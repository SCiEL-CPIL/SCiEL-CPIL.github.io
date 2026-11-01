"""Pack 11 unit hexagons inside a minimal regular hexagon."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = math.sqrt(3)
    
    # Generate hexagonal lattice points in axial coordinates
    centers_all = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                centers_all.append((x, y))
    
    # Sort by distance from origin and take the 11 closest
    centers_all.sort(key=lambda p: p[0]**2 + p[1]**2)
    top_11 = centers_all[:11]
    
    def bounding_hex_side(points, theta):
        max_val = 0.0
        for px, py in points:
            for k in range(6):
                angle = math.radians(theta + 30 + k * 60)
                nx = math.cos(angle)
                ny = math.sin(angle)
                val = abs(nx * px + ny * py)
                if val > max_val:
                    max_val = val
        return 2.0 / sqrt3 * max_val
    
    best_s = float('inf')
    best_theta = 0
    for theta in range(0, 60):
        s = bounding_hex_side(top_11, theta)
        if s < best_s:
            best_s = s
            best_theta = theta
    
    for theta in [best_theta + d / 100.0 for d in range(-100, 101)]:
        s = bounding_hex_side(top_11, theta)
        if s < best_s:
            best_s = s
            best_theta = theta
    
    inner = [[x, y, 0.0] for x, y in top_11]
    return inner, (0.0, 0.0), best_s, best_theta
# EVOLVE-BLOCK-END
