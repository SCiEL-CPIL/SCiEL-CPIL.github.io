import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate 3-5-3 cluster of unit hexagons
    # Using flat-top orientation lattice vectors
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
    # We will generate centers and then find the minimal bounding hexagon
    
    sqrt3 = np.sqrt(3)
    
    # Row 0: 3 hexagons
    # Centers: -sqrt3, 0, sqrt3 at y=0
    centers = []
    for i in range(3):
        x = (i - 1) * sqrt3
        centers.append((x, 0.0))
        
    # Row 1: 5 hexagons
    # Centers: -2sqrt3, -sqrt3/2, sqrt3/2, 3sqrt3/2, 2sqrt3 at y=1.5
    # Wait, the x-coords above are not evenly spaced by sqrt3.
    # -2sqrt3 to -sqrt3/2 is 1.5sqrt3.
    # We need spacing sqrt3.
    # Let's list them:
    # -2sqrt3
    # -2sqrt3 + sqrt3 = -sqrt3
    # -sqrt3 + sqrt3 = 0 ? No, offset is needed.
    
    # Let's re-evaluate the x-coords for Row 1.
    # To be adjacent to Row 0, x must be Row0_x +/- sqrt3/2.
    # Row 0 x: -sqrt3, 0, sqrt3.
    # Possible x for Row 1:
    # -sqrt3 - sqrt3/2 = -1.5sqrt3
    # -sqrt3 + sqrt3/2 = -0.5sqrt3
    # 0 - sqrt3/2 = -0.5sqrt3 (dup)
    # 0 + sqrt3/2 = 0.5sqrt3
    # sqrt3 - sqrt3/2 = 0.5sqrt3 (dup)
    # sqrt3 + sqrt3/2 = 1.5sqrt3
    
    # So the "adjacent" positions are -1.5sqrt3, -0.5sqrt3, 0.5sqrt3, 1.5sqrt3.
    # That's 4 positions. We need 5.
    # We can extend to -2.5sqrt3 and 2.5sqrt3?
    # Or just use a symmetric set of 5 that fits.
    # Let's use: -2sqrt3, -sqrt3, 0, sqrt3, 2sqrt3?
    # Distance from 0 to sqrt3 is sqrt3.
    # Distance from (0,0) to (sqrt3, 1.5) is sqrt(3 + 2.25) = sqrt(5.25) != sqrt3.
    # So (0,0) and (sqrt3, 1.5) are NOT adjacent.
    
    # To ensure adjacency and compactness, we should use the lattice.
    # Let's just use a simple dense grid.
    # I will use the 7-cluster + 4 strategy which is safer.
    
    # 7-Cluster Centers
    c7 = [(0.0, 0.0)]
    for k in range(6):
        angle = k * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        c7.append((x, y))
        
    # Add 4 more.
    # Place them at the "corners" of the 7-cluster, extending outwards.
    # The 7-cluster has vertices at distance sqrt3+1.
    # Let's place them at distance 2*sqrt3 from center, at angles 0, pi/2, pi, 3pi/2?
    # This creates a cross shape.
    c_extra = []
    for k in range(4):
        angle = k * np.pi / 2
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        c_extra.append((x, y))
        
    all_centers = c7 + c_extra
    
    # Calculate minimal bounding hexagon side length S
    # Assume outer hexagon is aligned with inner (angle 0)
    # We need to find S such that all hexagons fit.
    # A unit hexagon at (cx, cy) with angle 0 fits in an outer hexagon of side S at (0,0)
    # if its vertices are inside.
    # Vertices of unit hex at (0,0) angle 0:
    # (1, 0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1, 0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
    
    hex_verts = [
        (1.0, 0.0),
        (0.5, sqrt3/2),
        (-0.5, sqrt3/2),
        (-1.0, 0.0),
        (-0.5, -sqrt3/2),
        (0.5, -sqrt3/2)
    ]
    
    def point_in_hex(px, py, S):
        # Check if (px, py) is inside regular hexagon centered at 0,0 with side S
        # Inradius r = S * sqrt3 / 2
        # Circumradius R = S
        # Convert to polar? Or use half-planes.
        # Half-planes defined by normals at 0, 60, 120, 180, 240, 300 degrees.
        # Normal n_k = (cos(k*pi/3), sin(k*pi/3))
        # Constraint: dot(p, n_k) <= S * sqrt3 / 2
        r_in = S * sqrt3 / 2
        for k in range(6):
            angle = k * np.pi / 3
            nx = np.cos(angle)
            ny = np.sin(angle)
            if px * nx + py * ny > r_in:
                return False
        return True
        
    def fits(S):
        for cx, cy in all_centers:
            for vx, vy in hex_verts:
                px = cx + vx
                py = cy + vy
                if not point_in_hex(px, py, S):
                    return False
        return True
        
    # Binary search for S
    lo = 3.0
    hi = 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
            
    S_opt = hi
    
    # Format output
    inner = [[cx, cy, 0.0] for cx, cy in all_centers]
    return inner, (0.0, 0.0), S_opt, 0.0

# EVOLVE-BLOCK-END
