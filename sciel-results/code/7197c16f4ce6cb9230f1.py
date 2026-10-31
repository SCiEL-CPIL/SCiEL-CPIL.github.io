import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for 11 hexagons in a hexagonal lattice pattern
    # Flat-top orientation for lattice
    # Neighbor distance d = sqrt(3)
    d = np.sqrt(3)
    
    centers = [(0.0, 0.0)]
    
    # Layer 1: 6 hexagons
    for k in range(6):
        angle = np.deg2rad(30 + 60 * k)
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        centers.append((x, y))
        
    # Layer 2: 4 hexagons at "side" positions
    # Sides are between Layer 1 neighbors
    # Vectors to Layer 1:
    # v0 = (1.5, sqrt(3)/2)
    # v1 = (0, sqrt(3))
    # v2 = (-1.5, sqrt(3)/2)
    # v3 = (-1.5, -sqrt(3)/2)
    # v4 = (0, -sqrt(3))
    # v5 = (1.5, -sqrt(3)/2)
    
    # Side positions are sums of adjacent Layer 1 vectors
    # S0 = v0 + v1 = (1.5, 1.5*sqrt(3))
    # S1 = v1 + v2 = (-1.5, 1.5*sqrt(3))
    # S2 = v2 + v3 = (-3, 0)
    # S3 = v3 + v4 = (-1.5, -1.5*sqrt(3))
    # S4 = v4 + v5 = (1.5, -1.5*sqrt(3))
    # S5 = v5 + v0 = (3, 0)
    
    # We want to minimize the bounding hexagon.
    # S0, S1, S4, S5 form a rectangle symmetric about axes?
    # S0: (1.5, 2.598)
    # S1: (-1.5, 2.598)
    # S4: (1.5, -2.598)
    # S5: (-1.5, -2.598)
    # These 4 are symmetric.
    side_centers = [
        (1.5, 1.5 * np.sqrt(3)),
        (-1.5, 1.5 * np.sqrt(3)),
        (-1.5, -1.5 * np.sqrt(3)),
        (1.5, -1.5 * np.sqrt(3))
    ]
    
    for c in side_centers:
        centers.append(c)
        
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    # Calculate minimum S for flat-top outer hexagon
    # Flat-top normals are at 30, 90, 150, 210, 270, 330 degrees
    # Support function of unit flat-top hexagon:
    # Max support is 1 (at 0, 60, ... degrees)
    # Min support is sqrt(3)/2 (at 30, 90, ... degrees)
    
    def support(angle_deg):
        # Angle of normal
        a = np.deg2rad(angle_deg)
        # For flat-top hex, vertices at 0, 60...
        # The support function is periodic.
        # Reduce angle to [0, 60)
        a_mod = a % (np.pi / 3)
        # In the first sector [0, 30], support decreases from 1 to sqrt(3)/2
        # In [30, 60], it increases? No, symmetry.
        # Actually, for flat-top:
        # Normal 0: support 1
        # Normal 30: support sqrt(3)/2
        # Normal 60: support 1
        # So support = cos(a_mod - 30deg) * ? No.
        # Let's just use the max over the 6 vertices.
        # Vertices at 0, 60, 120, 180, 240, 300
        max_s = 0
        for k in range(6):
            v_angle = np.deg2rad(60 * k)
            v = np.array([np.cos(v_angle), np.sin(v_angle)])
            u = np.array([np.cos(a), np.sin(a)])
            proj = np.dot(v, u)
            if proj > max_s:
                max_s = proj
        return max_s

    max_h = 0
    for normal_deg in [30, 90, 150, 210, 270, 330]:
        u = np.array([np.cos(np.deg2rad(normal_deg)), np.sin(np.deg2rad(normal_deg))])
        h_support = support(normal_deg)
        
        max_proj = 0
        for c in centers:
            proj = np.dot(np.array(c), u)
            total = proj + h_support
            if total > max_proj:
                max_proj = total
        if max_proj > max_h:
            max_h = max_proj
            
    # S = 2 * H / sqrt(3)
    S = 2 * max_h / np.sqrt(3)
    
    # Add small epsilon for safety
    S += 0.01
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
