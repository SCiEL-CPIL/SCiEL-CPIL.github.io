import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define compact cluster of 11 unit hexagons using axial coordinates
    centers_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (-1, 2)
    ]
    
    # Convert axial to cartesian (flat-top hexagons)
    centers = []
    for q, r in centers_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2)
        centers.append((x, y))
    
    # Generate all vertices
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = np.radians(60 * k)
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append((vx, vy))
    all_vertices = np.array(all_vertices)
    
    def min_side_length(phi_deg):
        phi_rad = np.radians(phi_deg)
        widths = []
        for k in range(6):
            angle = phi_rad + np.radians(30 + 60 * k)
            nx = np.cos(angle)
            ny = np.sin(angle)
            proj = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            widths.append(np.max(proj) - np.min(proj))
        return max(widths) / np.sqrt(3)
    
    # Ternary search over [0, 60)
    lo, hi = 0.0, 60.0
    for _ in range(200):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if min_side_length(m1) < min_side_length(m2):
            hi = m2
        else:
            lo = m1
    optimal_phi = (lo + hi) / 2
    optimal_side = min_side_length(optimal_phi)
    
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    cx_center = float(np.mean(all_vertices[:, 0]))
    cy_center = float(np.mean(all_vertices[:, 1]))
    
    return inner, (cx_center, cy_center), float(optimal_side), float(optimal_phi)
# EVOLVE-BLOCK-END
