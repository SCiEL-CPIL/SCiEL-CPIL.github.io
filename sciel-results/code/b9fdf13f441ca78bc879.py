import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3.0)
    inradius = sqrt3 / 2.0
    spacing = sqrt3
    
    def axial_to_cartesian(q, r):
        x = spacing * (q + r / 2.0)
        y = spacing * (sqrt3 / 2.0) * r
        return x, y
    
    candidates = []
    max_radius = 10
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            dist = (abs(q) + abs(r) + abs(q + r)) // 2
            x, y = axial_to_cartesian(q, r)
            candidates.append((dist, q, r, x, y))
    
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    selected = candidates[:N]
    inner_positions = [(x, y) for (_, _, _, x, y) in selected]
    
    xs = [p[0] for p in inner_positions]
    ys = [p[1] for p in inner_positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    
    # For flat-top hexagon, normals at 30 + k*60 degrees
    theta_angles = [(30 + k * 60) * np.pi / 180.0 for k in range(6)]
    
    max_required_R = 0.0
    for theta in theta_angles:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        max_proj = 0.0
        for (px, py) in inner_positions:
            proj = (px - cx) * cos_t + (py - cy) * sin_t
            if proj > max_proj:
                max_proj = proj
        required_R = max_proj + inradius
        if required_R > max_required_R:
            max_required_R = required_R
    
    R = max_required_R
    side_length = 2.0 * R / sqrt3
    
    inner = [[x, y, 0.0] for (x, y) in inner_positions]
    
    return inner, (cx, cy), side_length, 0.0
# EVOLVE-BLOCK-END
