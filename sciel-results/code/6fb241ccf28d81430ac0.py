import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    s3 = np.sqrt(3)
    
    # Define compact cluster of 11 hexagons using axial coordinates
    # Pattern: 3-5-3 rows
    axial_coords = []
    for r in [-1, 0, 1]:
        if r == -1:
            qs = [-1, 0, 1]
        elif r == 0:
            qs = [-2, -1, 0, 1, 2]
        else:
            qs = [-1, 0, 1]
        for q in qs:
            axial_coords.append((q, r))
    
    # Convert to Cartesian (flat-top hexagons)
    inner = []
    for q, r in axial_coords:
        x = s3 * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for x, y, _ in inner:
        for i in range(6):
            angle = i * 60 * np.pi / 180
            vx = x + np.cos(angle)
            vy = y + np.sin(angle)
            all_vertices.append([vx, vy])
    all_vertices = np.array(all_vertices)
    
    # Function to compute minimum side length for given center
    def min_side_length(cx, cy, vertices):
        dx = vertices[:, 0] - cx
        dy = vertices[:, 1] - cy
        # 6 linear forms
        v1 = dx
        v2 = dy / s3
        v3 = dx + dy / s3
        v4 = -dx
        v5 = -dy / s3
        v6 = dx - dy / s3
        return np.max(np.maximum(np.maximum(np.maximum(v1, v2), np.maximum(v3, v4)), np.maximum(v5, v6)))
    
    # Initial center: centroid
    cx0 = np.mean(all_vertices[:, 0])
    cy0 = np.mean(all_vertices[:, 1])
    
    best_L = min_side_length(cx0, cy0, all_vertices)
    best_cx, best_cy = cx0, cy0
    
    # Grid search for better center
    for dx in np.linspace(-1.0, 1.0, 21):
        for dy in np.linspace(-1.0, 1.0, 21):
            L = min_side_length(cx0 + dx, cy0 + dy, all_vertices)
            if L < best_L:
                best_L = L
                best_cx, best_cy = cx0 + dx, cy0 + dy
    
    # Fine-tune
    for _ in range(3):
        step = 0.1
        for dx in np.linspace(-step, step, 11):
            for dy in np.linspace(-step, step, 11):
                L = min_side_length(best_cx + dx, best_cy + dy, all_vertices)
                if L < best_L:
                    best_L = L
                    best_cx, best_cy = best_cx + dx, best_cy + dy
        step *= 0.5
    
    return inner, (best_cx, best_cy), best_L, 0.0
# EVOLVE-BLOCK-END

