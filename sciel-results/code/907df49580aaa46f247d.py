import numpy as np

N = 11

def hex_to_cart(q, r):
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def construct():
    # Compact honeycomb arrangement for 11 hexagons
    axial = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
             (2,-1), (2,-2), (-2,1), (-2,2)]
    
    centers = np.array([hex_to_cart(q, r) for q, r in axial])
    
    # All inner hexagons axis-aligned (pointy-top)
    base_angles = np.radians([90, 150, 210, 270, 330, 30])
    all_vertices = []
    for cx, cy in centers:
        verts = np.column_stack([cx + np.cos(base_angles), cy + np.sin(base_angles)])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Find minimal axis-aligned pointy-top regular hexagon containing all vertices
    c = np.mean(all_vertices, axis=0)
    normals = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in [0, 60, 120, 180, 240, 300]])
    
    shifted = all_vertices - c
    dots = shifted @ normals.T
    max_dot = np.max(dots, axis=0)
    s = np.max(max_dot) / (np.sqrt(3) / 2)
    s *= 1.001  # safety margin
    
    inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
    return inner, (float(c[0]), float(c[1])), float(s), 0.0
