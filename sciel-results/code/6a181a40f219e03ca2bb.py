import numpy as np
from scipy.optimize import minimize
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define hexagon vertices for a unit hexagon at origin, angle 0
    # Flat-top hexagon vertices relative to center:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    def get_hex_vertices(cx, cy):
        # Returns 6 vertices of a unit hexagon centered at (cx, cy) with angle 0
        # This orientation has a flat top and bottom? 
        # Standard "flat top" has vertices at 0, 60, 120...
        # Let's use the standard mathematical orientation:
        # Vertex 0: (1, 0)
        # Vertex 1: (0.5, sqrt(3)/2)
        # ...
        verts = []
        for k in range(6):
            angle = k * np.pi / 3
            x = cx + np.cos(angle)
            y = cy + np.sin(angle)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, cx, cy):
        # Check if point (px, py) is inside unit hexagon centered at (cx, cy)
        # Translate to center
        dx = px - cx
        dy = py - cy
        # Hexagon is intersection of 6 half-planes
        # Normals at 0, 60, 120, 180, 240, 300
        # Inequality: x*cos(theta) + y*sin(theta) <= sqrt(3)/2
        for k in range(6):
            angle = k * np.pi / 3
            proj = dx * np.cos(angle) + dy * np.sin(angle)
            if proj > np.sqrt(3) / 2 + 1e-9:
                return False
        return True

    def calculate_min_S(positions):
        # positions: list of (x, y)
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            verts = get_hex_vertices(cx, cy)
            for vx, vy in verts:
                # Project onto 6 normal directions of the outer hexagon
                # Outer hexagon is centered at (0,0), angle 0
                for k in range(6):
                    angle = k * np.pi / 3
                    proj = vx * np.cos(angle) + vy * np.sin(angle)
                    if proj > max_proj:
                        max_proj = proj
        # Apothem of outer hexagon must be >= max_proj
        # Apothem = S * sqrt(3) / 2
        # So S >= max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / np.sqrt(3)

    def overlap_penalty(positions):
        penalty = 0.0
        for i in range(N):
            cx1, cy1 = positions[i]
            verts1 = get_hex_vertices(cx1, cy1)
            for j in range(i + 1, N):
                cx2, cy2 = positions[j]
                # Quick check: if distance > 2, no overlap
                dist = np.hypot(cx1 - cx2, cy1 - cy2)
                if dist > 2.0:
                    continue
                # Check if any vertex of 1 is inside 2
                for vx, vy in verts1:
                    if point_in_hexagon(vx, vy, cx2, cy2):
                        penalty += 100.0
                        break
                # Check if any vertex of 2 is inside 1
                verts2 = get_hex_vertices(cx2, cy2)
                for vx, vy in verts2:
                    if point_in_hexagon(vx, vy, cx1, cy1):
                        penalty += 100.0
                        break
        return penalty

    def objective(params):
        # params: flattened array of x1, y1, ..., x12, y12
        positions = [(params[i*2], params[i*2+1]) for i in range(N)]
        s_min = calculate_min_S(positions)
        pen = overlap_penalty(positions)
        # Weight penalty heavily to ensure validity
        return s_min + 10.0 * pen

    # Initialization: Compact grid
    # 3 rows of 4, or similar
    # Let's try a 4x3 grid in hex coordinates
    init_params = []
    for r in range(3):
        for c in range(4):
            x = np.sqrt(3) * (c + 0.5 * (r % 2)) - 1.5 * np.sqrt(3) # Center it roughly
            y = 1.5 * r - 1.5
            init_params.append(x)
            init_params.append(y)
            
    # Optimize
    # Use Powell method as it doesn't require gradients and is robust for non-smooth functions (like max)
    # Nelder-Mead might be faster
    result = minimize(objective, init_params, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_params = result.x
    best_positions = [(best_params[i*2], best_params[i*2+1]) for i in range(N)]
    
    # Check if solution is valid (no overlap)
    if overlap_penalty(best_positions) > 0:
        # Try Powell if Nelder-Mead failed to find valid
        result2 = minimize(objective, init_params, method='Powell', 
                      options={'maxiter': 500, 'ftol': 1e-4})
        if result2.fun < result.fun:
            best_params = result2.x
            best_positions = [(best_params[i*2], best_params[i*2+1]) for i in range(N)]
            
    # If still invalid, fallback to a known valid configuration (e.g. linear)
    # But we hope the optimizer found a valid one.
    # Let's do a final check and if invalid, use a safe but suboptimal layout
    if overlap_penalty(best_positions) > 0:
         # Fallback: Linear arrangement
         best_positions = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
         # Calculate S for this fallback
         s_val = calculate_min_S(best_positions)
    else:
         s_val = calculate_min_S(best_positions)

    inner = [[x, y, 0.0] for x, y in best_positions]
    return inner, (0.0, 0.0), s_val, 0.0

# EVOLVE-BLOCK-END
