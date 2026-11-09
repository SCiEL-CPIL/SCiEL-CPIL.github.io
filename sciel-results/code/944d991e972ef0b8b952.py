"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct 32 points on the triangular lattice maximizing unit-distance pairs.
    Uses analytical construction based on Harborth's theorem for n=32.
    """
    # Generate hexagon of radius 3 (37 points)
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            # Hex distance condition: |a| + |b| + |a+b| <= 6
            if abs(a) + abs(b) + abs(a + b) <= 6:
                points.append((a, b))
    
    # Points to remove to optimize edge count for n=32
    # Removing 3 corners and 2 specific edge points reduces edge loss to 14
    # resulting in 90 - 14 = 76 edges, which is the theoretical maximum.
    remove_set = {
        (3, 0),   # Corner
        (0, 3),   # Corner
        (-3, 3),  # Corner
        (2, 1),   # Edge point adjacent to (3,0) and (1,2)
        (1, 2)    # Edge point adjacent to (0,3) and (2,1)
    }
    
    result = []
    for p in points:
        if p not in remove_set:
            result.append([p[0], p[1]])
            
    # Assertions for safety
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
        
    return result
# EVOLVE-BLOCK-END
