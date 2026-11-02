import numpy as np
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 3x4 grid of hexagons in honeycomb lattice
    # Basis vectors for pointy-top unit hexagons
    # Horizontal distance: sqrt(3)
    # Vertical distance: 1.5
    # x-offset for next row: sqrt(3)/2
    
    u = np.array([math.sqrt(3), 0.0])
    v = np.array([math.sqrt(3)/2, 1.5])
    
    centers = []
    for i in range(4):
        for j in range(3):
            c = i * u + j * v
            centers.append(c)
            
    centers = np.array(centers)
    
    # Center the cluster at the origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Generate all vertices of the inner hexagons
    # Inner hexagons are pointy-top, so vertices are at angles 30, 90, 150, 210, 270, 330
    # Angles in radians: (30 + 60*k) * pi / 180
    vertex_angles = np.array([(30 + 60*k) * math.pi / 180 for k in range(6)])
    vertex_offsets = np.stack([np.cos(vertex_angles), np.sin(vertex_angles)], axis=1)
    
    all_vertices = []
    for c in centers:
        verts = c + vertex_offsets
        all_vertices.extend(verts)
        
    all_vertices = np.array(all_vertices)
    
    # Function to calculate the required side length for a given outer hexagon rotation
    def get_side(alpha_deg):
        alpha_rad = alpha_deg * math.pi / 180
        # Normals for pointy-top outer hexagon are at 0, 60, 120, 180, 240, 300
        # Rotated by alpha: alpha + 60*k
        normal_angles = np.array([(alpha_deg + 60*k) * math.pi / 180 for k in range(6)])
        normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
        
        # Project all vertices onto all normals
        # dots[i, j] is projection of vertex i onto normal j
        dots = all_vertices @ normals.T
        
        # The max projection determines the apothem
        max_proj = np.max(dots)
        
        # Apothem = S * sqrt(3) / 2
        # So S = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)
        
    # Optimize alpha
    # The function is periodic with period 60 degrees
    # We can search in [0, 60)
    # Use a fine grid search first, then refine if needed
    alphas = np.linspace(0, 60, 600, endpoint=False)
    sides = [get_side(a) for a in alphas]
    
    min_side = min(sides)
    best_alpha = alphas[sides.index(min_side)]
    
    # Refine around best_alpha
    # Simple local search
    best_alpha_local = best_alpha
    best_side_local = min_side
    step = 0.01
    for _ in range(100):
        for delta in [-step, step]:
            a = best_alpha_local + delta
            if 0 <= a < 60:
                s = get_side(a)
                if s < best_side_local:
                    best_side_local = s
                    best_alpha_local = a
        step /= 2.0
        
    final_side = best_side_local
    final_angle = best_alpha_local
    
    # Construct inner list
    inner = []
    for c in centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), float(final_angle)
# EVOLVE-BLOCK-END
