import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Calculate the minimal side length k for the hexagonal cluster
    # N = 3k(k-1) + 1
    # k(k-1) >= (N-1)/3
    # k^2 - k - (N-1)/3 >= 0
    # k = (1 + sqrt(1 + 4*(N-1)/3)) / 2
    val = (N - 1) / 3.0
    k = math.ceil((1 + math.sqrt(1 + 4 * val)) / 2)
    
    # Ensure k is at least 1
    k = max(1, k)
    
    # Generate hexagonal lattice points in axial coordinates (q, r)
    # Physical distance between centers:
    # dx = sqrt(3) * (q + r/2)
    # dy = 1.5 * r
    # Hexagonal distance from center (0,0) is max(|q|, |r|, |q+r|)
    # We want points within distance k-1 from center to form a hexagon of side k
    
    points = []
    # Iterate over a square range large enough to cover the hexagon
    range_k = k
    for q in range(-range_k, range_k + 1):
        for r in range(-range_k, range_k + 1):
            # Check if (q, r) is within the hexagon of radius k-1
            # Hex distance: max(|q|, |r|, |q+r|) <= k-1
            hex_dist = max(abs(q), abs(r), abs(q + r))
            if hex_dist <= k - 1:
                # Convert to Cartesian
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                points.append((x, y, q, r))
    
    # Sort points by distance from center to ensure we pick the most central ones first
    # This helps if N is not exactly 3k(k-1)+1, ensuring the shape is as compact as possible
    points.sort(key=lambda p: (p[2]**2 + p[3]**2 + p[2]*p[3], p[0]**2 + p[1]**2))
    
    # Take the first N points
    selected_points = points[:N]
    
    # Format inner hexagons: [x, y, angle_degrees]
    # Angle is 0.0 because they are aligned with the standard orientation
    inner = []
    for x, y, _, _ in selected_points:
        inner.append([x, y, 0.0])
    
    # Calculate outer side length
    # The furthest point from the center in the cluster is at distance (k-1)*sqrt(3) from center (center of hex)
    # Plus the radius of the unit hexagon (1.0)
    # So outer hexagon radius (side length) is (k-1)*sqrt(3) + 1
    outer_side = (k - 1) * math.sqrt(3) + 1.0
    
    # Outer center is (0, 0), angle 0.0
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
