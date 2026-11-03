import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 hexagon centers in a 3-4-3-1 lattice arrangement
    # Lattice vectors: u = (sqrt(3), 0), v = (sqrt(3)/2, 1.5)
    # We use a coordinate system where the lattice is aligned with axes
    # Row 0 (y=0): 3 hexagons
    # Row 1 (y=1.5): 4 hexagons
    # Row 2 (y=3.0): 3 hexagons
    # Row 3 (y=4.5): 1 hexagon
    
    sqrt3 = np.sqrt(3)
    centers = [
        [-sqrt3, 0.0],
        [0.0, 0.0],
        [sqrt3, 0.0],
        
        [-1.5*sqrt3, 1.5],
        [-0.5*sqrt3, 1.5],
        [0.5*sqrt3, 1.5],
        [1.5*sqrt3, 1.5],
        
        [-sqrt3, 3.0],
        [0.0, 3.0],
        [sqrt3, 3.0],
        
        [0.0, 4.5]
    ]
    centers = np.array(centers)
    
    # Center the cluster at the origin to balance the bounding box
    # Calculate centroid
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    def hex_support(angle_rad):
        # Support function of a unit regular hexagon (pointy-top, vertex at 0)
        # h(theta) = cos(theta mod 60 deg) is not quite right for all ranges,
        # but due to symmetry h(theta) = h(theta mod 60) is not true for width,
        # wait. The hexagon is symmetric under 60 deg rotation.
        # So h(theta) is periodic with 60 deg.
        # Within [0, 30], h(theta) = cos(theta) ?
        # At 0, h=1. At 30, h=sqrt(3)/2.
        # Yes, h(theta) = cos(theta) for theta in [-30, 30] relative to vertex.
        # So h(theta) = cos(theta % (2*pi/3) - pi/3)? No.
        # Let's just use the formula: h(theta) = max(cos(theta - k*pi/3)) for k in 0..5
        # Or simpler: h(theta) = cos(theta % (pi/3)) is wrong because cos is symmetric around 0.
        # The function is periodic with pi/3.
        # In [0, pi/6], it goes from 1 to sqrt(3)/2.
        # So h(theta) = cos(theta) for theta in [0, pi/6]?
        # Let's check. Vertex at 0. Side normal at 30.
        # Distance to side is sqrt(3)/2.
        # So h(30) = sqrt(3)/2. cos(30) = sqrt(3)/2. Correct.
        # So h(theta) = cos(theta) for theta in [-30, 30] degrees.
        # Due to periodicity, h(theta) = cos(theta % (pi/3) - 0)? No.
        # h(theta) = cos(theta % (pi/3)) is not correct because cos(60) = 0.5, but h(60)=1.
        # The period is 60 degrees.
        # h(theta) = cos(theta) for theta in [-30, 30].
        # So h(theta) = cos(theta % (pi/3) - 0) ? No.
        # Let's map theta to [-30, 30].
        # t = theta % (pi/3)
        # if t > pi/6: t = t - pi/3
        # return cos(t)
        
        t = angle_rad % (np.pi / 3)
        if t > np.pi / 6:
            t = t - np.pi / 3
        return np.cos(t)

    def cluster_width(theta_rad):
        # Width of the center points in direction theta
        u = np.array([np.cos(theta_rad), np.sin(theta_rad)])
        projections = centers @ u
        w_centers = np.max(projections) - np.min(projections)
        # Add the width of the hexagons themselves
        w_hex = 2 * hex_support(theta_rad)
        return w_centers + w_hex

    def score(beta_rad):
        # beta is the rotation of the container relative to the cluster
        # Container normals are at beta + k * 60
        max_width = 0.0
        for k in range(6):
            theta = beta_rad + k * np.pi / 3
            w = cluster_width(theta)
            if w > max_width:
                max_width = w
        return max_width / np.sqrt(3)

    # Optimize beta in [0, 30 degrees] due to symmetry
    # Use a simple grid search followed by refinement
    best_beta = 0.0
    best_score = float('inf')
    
    # Coarse search
    for deg in np.linspace(0, 30, 100):
        b = np.radians(deg)
        s = score(b)
        if s < best_score:
            best_score = s
            best_beta = b
            
    # Fine search
    for deg in np.linspace(best_beta - np.radians(1), best_beta + np.radians(1), 100):
        b = np.radians(deg)
        s = score(b)
        if s < best_score:
            best_score = s
            best_beta = b

    side_length = best_score
    
    # Construct the return values
    # Inner hexagons: coordinates are 'centers', angle is 0.0
    # We can rotate the inner hexagons to 0.0 and rotate the outer container by best_beta
    # Or rotate inner by -best_beta and keep outer at 0.
    # The interface allows outer angle.
    # Let's keep inner coordinates as calculated (centered at 0,0) and angles 0.0
    # And set the outer hexagon angle to best_beta.
    
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    center = (0.0, 0.0)
    side = side_length
    angle = np.degrees(best_beta)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
