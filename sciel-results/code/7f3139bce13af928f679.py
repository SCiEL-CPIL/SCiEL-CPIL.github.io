import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 7-hexagon cluster (center + 6 neighbors)
    # Centers of the 7 hexagons
    sqrt3 = np.sqrt(3)
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (-sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (sqrt3/2, -1.5),
        (-sqrt3/2, 1.5),
        (-sqrt3/2, -1.5)
    ]
    
    # Add 5 more hexagons
    # We add them to the outer positions to minimize the bounding box expansion
    # The 7-cluster is a hexagon of side ~2.73.
    # Adding hexagons to the sides extends the radius.
    # Let's add them to the "corners" of the lattice? No, sides.
    # Let's add them at:
    # Right-Top, Top, Top-Left, Left, Bottom-Left
    # Coordinates for new hexagons (distance sqrt3 from existing outer hexagons)
    
    # Outer hexagons of the 7-cluster:
    # 1: (sqrt3, 0) -> Right
    # 2: (sqrt3/2, 1.5) -> Top-Right
    # 3: (-sqrt3/2, 1.5) -> Top-Left
    # 4: (-sqrt3, 0) -> Left
    # 5: (-sqrt3/2, -1.5) -> Bottom-Left
    # 6: (sqrt3/2, -1.5) -> Bottom-Right
    
    # We skip Bottom-Right to keep some balance? Or skip one?
    # Let's add to 1, 2, 3, 4, 5.
    
    # New positions:
    # Adjacent to 1 (Right): (2*sqrt3, 0)
    # Adjacent to 2 (Top-Right): (sqrt3, 3.0) ? 
    #   Center (sqrt3/2, 1.5). Neighbor up-right?
    #   Vector from center to 2 is (sqrt3/2, 1.5).
    #   Next step in same direction: (sqrt3/2, 1.5) + (sqrt3/2, 1.5) = (sqrt3, 3.0).
    # Adjacent to 3 (Top-Left): (-sqrt3/2, 1.5) + (-sqrt3/2, 1.5) = (-sqrt3, 3.0).
    # Adjacent to 4 (Left): (-2*sqrt3, 0).
    # Adjacent to 5 (Bottom-Left): (-sqrt3/2, -1.5) + (-sqrt3/2, -1.5) = (-sqrt3, -3.0).
    
    extra_centers = [
        (2*sqrt3, 0.0),
        (sqrt3, 3.0),
        (-sqrt3, 3.0),
        (-2*sqrt3, 0.0),
        (-sqrt3, -3.0)
    ]
    
    all_centers = centers + extra_centers
    
    # All hexagons have angle 0.0 (flat-top)
    inner = [[x, y, 0.0] for x, y in all_centers]
    
    # Calculate the bounding box of the cluster
    xs = [c[0] for c in all_centers]
    ys = [c[1] for c in all_centers]
    
    # The cluster is not a perfect hexagon anymore.
    # We need to find the minimum S such that all points (plus hexagon radius) fit in the outer hexagon.
    # Approximation: Find the max distance from origin for all points, and add the circumradius of a unit hexagon (1.0).
    # This gives a circle bounding the cluster. Then fit a hexagon in that circle.
    # S_hex = R_circle.
    
    max_r = 0
    for x, y in all_centers:
        r = np.sqrt(x*x + y*y)
        if r > max_r:
            max_r = r
            
    # The farthest point of the cluster is max_r + 1 (circumradius of unit hex).
    # The outer hexagon must have circumradius >= max_r + 1.
    # So S >= max_r + 1.
    
    S = max_r + 1.0
    
    # Center the cluster at (0,0)
    center = (0.0, 0.0)
    
    return inner, center, S, 0.0
# EVOLVE-BLOCK-END
