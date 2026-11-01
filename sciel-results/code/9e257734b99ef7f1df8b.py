import numpy as np

def construct():
    # Define the 11 centers on a hexagonal lattice
    # Axial coordinates (q, r)
    # A 3x3 block centered at (0,0) has 9 points.
    # We add 2 more points to make 11.
    # The 3x3 block fits in a hexagon of side K=3 (L=4).
    # Let's verify the extent.
    # Points:
    # (-1, -1), (-1, 0), (-1, 1)
    # (0, -1), (0, 0), (0, 1)
    # (1, -1), (1, 0), (1, 1)
    # Add (2, 0) and (-2, 0)? 
    # If we add (2,0), x extent becomes 3*sqrt(3)/2 + sqrt(3) = 2.5*sqrt(3)?
    # Let's just use the 3x3 block and 2 points that fit within K=3.
    # Actually, the 3x3 block itself has K=3.
    # We can place the 11th and 10th hexagons in the "gaps" if we shift?
    # No, just use a known compact shape.
    
    # Centers in Cartesian coordinates
    # Lattice spacing sqrt(3)
    s = np.sqrt(3)
    
    # 3x3 block
    centers = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            x = s * (q + r/2.0)
            y = 1.5 * r
            centers.append((x, y))
            
    # Add 2 more points. 
    # The 3x3 block has 9 points.
    # We need 2 more.
    # Place them at (2, 0) and (-2, 0) in axial coords?
    # Axial (2,0) -> x = 2s, y=0.
    # Axial (-2,0) -> x = -2s, y=0.
    # Check if they fit in K=3.
    # Max x in 3x3 block is 1.5s.
    # New max x is 2s.
    # K=3 implies max projection A = 3*sqrt(3)/2 = 1.5s.
    # 2s > 1.5s. So they don't fit in K=3.
    # So we need K=4? L=5?
    # Wait.
    # The 3x3 block max x is 1.5s.
    # 1.5s = 1.5 * sqrt(3) = 2.598.
    # K=3 -> A = 3 * sqrt(3)/2 = 2.598.
    # So the 3x3 block fits exactly in K=3.
    # If we add points at 2s, they are outside.
    # So we cannot add points at 2s.
    # We must fit all 11 points in K=3.
    # This implies we cannot extend the 3x3 block.
    # We must use a different shape.
    # The 7-point cluster fits in K=2.
    # Can we fit 4 more points in K=3?
    # Yes, easily.
    # So L=4 is safe.
    
    # Let's just use the 7-point cluster and 4 points from the second shell
    # that fit in K=3.
    # Second shell points are at distance 3 (Cartesian) from center.
    # K=3 -> Radius 3.
    # So they fit on the boundary.
    
    # 7-point cluster:
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Add 4 from second shell:
    # (2,0), (0,2), (-2,2), (0,-2)?
    # Let's pick 4 that are symmetric.
    # (2,0), (2,-2), (-2,0), (-2,2)?
    
    # Axial to Cartesian:
    def axial_to_cart(q, r):
        return (s * (q + r/2.0), 1.5 * r)
        
    axial_coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (0,2), (-2,2), (0,-2)
    ]
    
    inner = []
    for q, r in axial_coords:
        x, y = axial_to_cart(q, r)
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), 4.0, 0.0
