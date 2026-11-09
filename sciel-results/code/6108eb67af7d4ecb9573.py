import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define centers of 11 unit hexagons
    centers = []
    
    # 1 center
    centers.append((0.0, 0.0))
    
    # 6 neighbors (Ring 1)
    ring1_axial = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for q, r in ring1_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        centers.append((x, y))
        
    # 4 more (Ring 2 subset)
    # Using (2,0), (0,2), (-2,0), (0,-2) for symmetry
    ring2_axial = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    for q, r in ring2_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * q)
        centers.append((x, y))
        
    # Prepare inner list: (x, y, angle)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Generate vertices for all inner hexagons
    rel_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    all_verts = []
    for cx, cy in centers:
        verts = rel_verts + np.array([cx, cy])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_L = float('inf')
    best_alpha = 0.0
    
    # Search alpha
    for alpha_deg in np.linspace(0, 60, 121):
        alpha = np.radians(alpha_deg)
        max_proj = 0
        for k in range(6):
            theta = alpha + k * np.pi / 3
            n = np.array([np.cos(theta), np.sin(theta)])
            projs = all_verts @ n
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        L = 2 * max_proj / np.sqrt(3)
        if L < best_L:
            best_L = L
            best_alpha = alpha_deg
            
    return inner, (0.0, 0.0), best_L, best_alpha
# EVOLVE-BLOCK-END
