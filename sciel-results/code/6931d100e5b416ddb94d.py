import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define candidate configurations in axial coordinates (q, r)
    # Flat-top hexagon grid
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    candidates = []
    
    # Candidate 1: 4x3 Block
    # q in [-1, 2], r in [-1, 1]
    c1 = [(q, r) for q in range(-1, 3) for r in range(-1, 2)]
    candidates.append(c1)
    
    # Candidate 2: 3x4 Block
    # q in [-1, 1], r in [-1, 2]
    c2 = [(q, r) for q in range(-1, 2) for r in range(-1, 3)]
    candidates.append(c2)
    
    # Candidate 3: Hexagonal Cluster (1 center, 6 ring, 5 of second ring)
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1) -- Pick 5 to make 12
    # Let's try a symmetric subset of Ring 2
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2_full = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
    # We need 5 more.
    # To keep it compact, pick adjacent ones.
    # Let's just use the 4x3 block as it's dense.
    
    best_L = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    
    sqrt3 = np.sqrt(3)
    
    for config in candidates:
        # Convert to Cartesian
        centers = []
        for q, r in config:
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            centers.append([x, y])
        
        centers = np.array(centers)
        
        # Center the configuration
        mean_center = np.mean(centers, axis=0)
        centers = centers - mean_center
        
        # Calculate required L for flat-top container
        # Constraints for point (x,y) to be in hexagon side L:
        # 1. |x| <= L
        # 2. |y| <= L * sqrt(3) / 2
        # 3. |sqrt(3)*x - y| <= L * sqrt(3)
        
        max_L = 0.0
        
        # Check all vertices of all inner hexagons
        # Inner hexagon vertices relative to center (flat-top, side 1):
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        vertices_rel = np.array([
            [1.0, 0.0],
            [0.5, sqrt3/2.0],
            [-0.5, sqrt3/2.0],
            [-1.0, 0.0],
            [-0.5, -sqrt3/2.0],
            [0.5, -sqrt3/2.0]
        ])
        
        for cx, cy in centers:
            vertices = vertices_rel + np.array([cx, cy])
            for vx, vy in vertices:
                L1 = abs(vx)
                L2 = abs(vy) * 2.0 / sqrt3
                L3 = abs(sqrt3 * vx - vy) / sqrt3
                max_L = max(max_L, L1, L2, L3)
        
        if max_L < best_L:
            best_L = max_L
            best_inner = [[cx, cy, 0.0] for cx, cy in centers]
            best_center = (0.0, 0.0) # Container is centered at origin
            
    return best_inner, best_center, best_L, 0.0
# EVOLVE-BLOCK-END
