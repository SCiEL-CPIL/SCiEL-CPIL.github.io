import numpy as np
from scipy.spatial import ConvexHull
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    d = math.sqrt(3)
    
    # Compact 11-hex cluster in hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-2, 1), (-1, -1)
    ]
    
    centers = []
    for q, r in axial_coords:
        x = d * (q + r / 2.0)
        y = d * (r * math.sqrt(3) / 2.0)
        centers.append((x, y))
    
    # All vertices
    all_points = []
    for cx, cy in centers:
        for k in range(6):
            angle = math.radians(60 * k)
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            all_points.append([vx, vy])
    
    all_points = np.array(all_points)
    centroid = np.mean(all_points, axis=0)
    all_points_centered = all_points - centroid
    
    hull = ConvexHull(all_points_centered)
    hull_points = all_points_centered[hull.vertices]
    
    def min_side_length(theta):
        max_proj = 0.0
        for px, py in hull_points:
            for k in range(6):
                angle = theta + math.radians(30 + 60 * k)
                nx = math.cos(angle)
                ny = math.sin(angle)
                proj = px * nx + py * ny
                if proj > max_proj:
                    max_proj = proj
        return (2.0 / math.sqrt(3)) * max_proj
    
    best_theta = 0
    best_L = float('inf')
    
    for i in range(3600):
        theta = math.radians(i * 60.0 / 3600)
        L = min_side_length(theta)
        if L < best_L:
            best_L = L
            best_theta = theta
    
    inner = []
    for cx, cy in centers:
        inner.append([cx - centroid[0], cy - centroid[1], 0.0])
    
    return inner, (0.0, 0.0), best_L, math.degrees(best_theta)
# EVOLVE-BLOCK-END
