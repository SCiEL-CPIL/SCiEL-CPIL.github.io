import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Find the smallest integer S such that the capacity of a hexagon of side S
    # is >= N.
    # Capacity of a regular hexagon of side S (in units of inner hex side 1)
    # tiled by unit hexagons is H_{S-1} = 3(S-1)^2 + 3(S-1) + 1 = 3S^2 - 3S + 1.
    
    S = 1
    while True:
        capacity = 3 * S * S - 3 * S + 1
        if capacity >= N:
            break
        S += 1
        
    # Generate axial coordinates (q, r) for the hexagonal cluster
    # Condition: max(|q|, |r|, |q+r|) < S
    inner = []
    for q in range(-S + 1, S):
        for r in range(-S + 1, S):
            if abs(q + r) < S:
                # Convert axial coordinates to Cartesian coordinates
                # Assuming flat-top hexagons, side length 1.
                # Distance between centers is sqrt(3).
                # x = sqrt(3) * (q + r/2)
                # y = 1.5 * r
                # This corresponds to a specific orientation.
                # Let's verify the orientation matches the outer hexagon.
                # If outer is flat-top, the lattice should be aligned.
                
                x = math.sqrt(3) * (q + 0.5 * r)
                y = 1.5 * r
                inner.append([x, y, 0.0])
                
    # We might have generated more than N hexagons if we just looped through all valid coords
    # The condition max(|q|,|r|,|q+r|) < S generates exactly 3S^2 - 3S + 1 points.
    # We only need the first N.
    # The order doesn't strictly matter for validity, as long as they are within the bounds.
    # However, to be safe and deterministic, we just take the first N.
    
    if len(inner) > N:
        inner = inner[:N]
        
    center = (0.0, 0.0)
    side = float(S)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
