import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 1. Define the cluster centers
    centers = [(0.0, 0.0)]
    
    # Inner ring (6 hexagons)
    d1 = np.sqrt(3)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        centers.append((d1 * np.cos(angle), d1 * np.sin(angle)))
        
    # Outer ring (5 hexagons)
    d2 = 2 * np.sqrt(3)
    for i in range(5):
        angle = np.deg2rad(60 * i)
        centers.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        
    # 2. Define vertices of a unit flat-top hexagon centered at origin
    hex_vertices = []
    for i in range(6):
        angle = np.deg2rad(60 * i)
        hex_vertices.append((np.cos(angle), np.sin(angle)))
        
    # 3. Generate all 72 points
    points = []
    for cx, cy in centers:
        for vx, vy in hex_vertices:
            points.append((cx + vx, cy + vy))
            
    points = np.array(points)
    
    # 4. Optimization function
    def get_side_length(alpha_rad):
        base_angles = [np.deg2rad(30), np.deg2rad(90), np.deg2rad(150)]
        max_proj = 0.0
        for base_angle in base_angles:
            n_angle = base_angle + alpha_rad
            n = np.array([np.cos(n_angle), np.sin(n_angle)])
            projs = np.abs(points @ n)
            max_proj = max(max_proj, np.max(projs))
        return (2.0 / np.sqrt(3)) * max_proj

    # 5. Coarse search
    best_alpha = 0.0
    best_S = float('inf')
    # 100 steps over 2*pi
    step = 2 * np.pi / 100
    for i in range(100):
        alpha = i * step
        S = get_side_length(alpha)
        if S < best_S:
            best_S = S
            best_alpha = alpha
            
    # 6. Fine search (Ternary)
    lo = best_alpha - step
    hi = best_alpha + step
    
    for _ in range(100):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = get_side_length(m1)
        f2 = get_side_length(m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
            
    final_alpha = (lo + hi) / 2.0
    final_S = get_side_length(final_alpha)
    
    # 7. Construct output
    cos_a = np.cos(final_alpha)
    sin_a = np.sin(final_alpha)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    inner = []
    for cx, cy in centers:
        c_arr = np.array([cx, cy])
        c_rot = R @ c_arr
        angle_deg = np.rad2deg(final_alpha)
        inner.append([float(c_rot[0]), float(c_rot[1]), float(angle_deg)])
        
    return inner, [0.0, 0.0], float(final_S), 0.0
# EVOLVE-BLOCK-END
