"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Arrange 12 unit hexagons in a compact honeycomb cluster
    # Use a symmetric 3-4-3-2 or similar arrangement
    
    # Let's try axial coordinates for a compact shape
    # A good 12-hex cluster: 
    # Row -1: 2 hexagons
    # Row 0:  4 hexagons  
    # Row 1:  3 hexagons
    # Row 2:  3 hexagons
    # Total: 12
    
    # Or try:
    # Row 0: 4
    # Row 1: 5
    # Row 2: 3
    # Total: 12
    
    positions = []
    # Row 0: 4 hexagons
    for c in range(4):
        positions.append((0, c))
    # Row 1: 5 hexagons
    for c in range(5):
        positions.append((1, c))
    # Row 2: 3 hexagons
    for c in range(3):
        positions.append((2, c))
    
    # Convert to Cartesian (pointy-top hexagons)
    inner = []
    for r, c in positions:
        x = np.sqrt(3) * (c + r / 2.0)
        y = 1.5 * r
        inner.append([float(x), float(y), 0.0])
    
    # Compute all vertices of all inner hexagons
    all_verts = []
    for cx, cy, ang in inner:
        for i in range(6):
            theta = np.radians(60 * i)
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            all_verts.append([vx, vy])
    all_verts = np.array(all_verts)
    
    # Find minimum enclosing regular hexagon
    # Try different centers and rotations
    # The center should be near the centroid
    cx0 = np.mean(all_verts[:, 0])
    cy0 = np.mean(all_verts[:, 1])
    
    best_side = float('inf')
    best_center = (cx0, cy0)
    best_angle = 0.0
    
    # Try several rotation angles
    for rot in np.linspace(0, 60, 13):  # 0 to 60 degrees in steps of 5
        # For a regular hexagon with rotation rot, the 6 outward normal directions
        # are at angles rot + 30 + 60*k for k=0..5
        normals = []
        for k in range(6):
            theta = np.radians(rot + 30 + 60 * k)
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        
        # For a given center (ox, oy), the minimum side length is:
        # s = max over all normals n and all vertices v of dot(v - c, n) / (sqrt(3)/2)
        # We want to minimize over c. The optimal c for fixed normals is the one
        # that equalizes the max projections, but as a good approximation, use centroid.
        
        # More carefully: for fixed normals, the minimal enclosing hexagon centered at c
        # has side length s(c) = max_k max_v dot(v - c, n_k) / (sqrt(3)/2)
        # To minimize s(c), we want to minimize the max of 6 quantities, each linear in c.
        # This is a linear programming problem, but let's just use the centroid as a good start.
        
        ox, oy = cx0, cy0
        max_proj = 0
        for n in normals:
            projs = np.dot(all_verts - np.array([ox, oy]), n)
            max_proj = max(max_proj, np.max(projs))
        
        side = max_proj / (np.sqrt(3) / 2)
        if side < best_side:
            best_side = side
            best_center = (ox, oy)
            best_angle = rot
    
    return inner, best_center, float(best_side), float(best_angle)
# EVOLVE-BLOCK-END
