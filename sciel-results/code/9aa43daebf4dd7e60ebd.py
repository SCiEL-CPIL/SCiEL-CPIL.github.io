"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct 32 points on the triangular lattice (integer pairs [a, b] representing
    a + b*omega where omega = exp(i*pi/3)) such that:
    - All points are distinct
    - |a|, |b| <= 64
    - Mutual distance >= 1 (in Euclidean terms on the triangular lattice)
    - Maximize the number of pairs at distance exactly 1
    
    The optimum is 76 edges (Harborth's theorem for n=32).
    
    Strategy: Use a triangular lattice spiral/hexagonal packing approach.
    The triangular lattice has 6 neighbors per point. A hexagonal arrangement
    of points minimizes the perimeter for a given number of points, maximizing
    internal edges.
    
    For n points arranged in a hexagonal shape on the triangular lattice,
    the number of unit-distance edges is maximized when the shape is as
    compact as possible (hexagonal).
    
    Let's build a hexagonal cluster of 32 points on the triangular lattice.
    The triangular lattice can be parameterized by axial coordinates (a, b).
    The distance squared between (a1, b1) and (a2, b2) in the triangular lattice
    (where omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2) is:
    
    Point = a + b*omega = (a + b/2) + i*(b*sqrt(3)/2)
    
    So x = a + b/2, y = b*sqrt(3)/2
    
    Distance squared = (a1-a2 + (b1-b2)/2)^2 + 3*(b1-b2)^2/4
    = (da + db/2)^2 + 3*db^2/4
    = da^2 + da*db + db^2/4 + 3*db^2/4
    = da^2 + da*db + db^2
    
    So dist^2 = da^2 + da*db + db^2 where da = a1-a2, db = b1-b2.
    
    For unit distance: da^2 + da*db + db^2 = 1
    Solutions: (da, db) in {(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)}
    These are the 6 neighbors.
    """
    
    # Generate a hexagonal cluster of 32 points on the triangular lattice.
    # Use axial coordinates (a, b).
    # A hexagon of "radius" r (in terms of hex distance) has 1 + 3*r*(r+1) points.
    # r=0: 1 point
    # r=1: 7 points
    # r=2: 19 points
    # r=3: 37 points
    # So 32 points is between r=2 (19) and r=3 (37).
    
    # Let's build a compact shape. Start with the hexagon of radius 2 (19 points),
    # then add 13 more points to make 32, choosing them to maximize edges.
    
    # Hexagon of radius 2 in axial coordinates:
    # All (a, b) such that max(|a|, |b|, |a+b|) <= 2? 
    # Actually, hex distance in axial coords: d = (|a| + |b| + |a+b|) / 2
    # Wait, let me reconsider. In axial coordinates for a hex grid,
    # the hex distance is (|a| + |b| + |a-b|) / 2 if using offset... 
    # Let me just use the standard: for point (a,b) in our lattice,
    # the 6 neighbors are (a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1).
    
    # Let's generate points in order of increasing "distance from center" and pick 32.
    # Center at (0,0).
    
    def hex_dist(a, b):
        # For our lattice with neighbors as defined, the hex distance is:
        # d = (|a| + |b| + |a + b|) / 2 ... let me verify.
        # Actually, let's just use BFS or a simple formula.
        # The distance from (0,0) to (a,b) in steps of the 6 neighbor moves.
        # This is equivalent to: (|a| + |b| + |a+b|) / 2 if we use cube coords properly.
        # Let me use cube coordinates: x=a, z=b, y=-a-b.
        # Then hex distance = (|x| + |y| + |z|) / 2 = (|a| + |a+b| + |b|) / 2
        return (abs(a) + abs(b) + abs(a + b)) // 2
    
    # Generate all points with hex_dist <= 3 (that gives 37 points), then trim to 32.
    # Or better: generate points in order of hex_dist, break ties to maximize edges.
    
    points = []
    # Add center
    points.append((0, 0))
    
    # Add rings
    for r in range(1, 4):
        ring = []
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                if hex_dist(a, b) == r:
                    ring.append((a, b))
        # Sort ring points to maximize connectivity - just add them all for now
        points.extend(ring)
    
    # We have 1 + 6 + 12 + 18 = 37 points. Need to remove 5 to get 32.
    # Remove points that contribute the fewest edges (i.e., have the fewest neighbors in the set).
    # Greedy: repeatedly remove the point with the fewest neighbors in the current set.
    
    point_set = set(points)
    
    def count_neighbors(p, pset):
        a, b = p
        neighbors = [(a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1)]
        return sum(1 for n in neighbors if n in pset)
    
    # Remove 5 points
    for _ in range(5):
        # Find point with minimum neighbor count (excluding the center if possible to keep it)
        min_count = float('inf')
        min_point = None
        for p in point_set:
            c = count_neighbors(p, point_set)
            if c < min_count or (c == min_count and min_point is not None and (p[0]**2 + p[1]**2) > (min_point[0]**2 + min_point[1]**2)):
                min_count = c
                min_point = p
        point_set.remove(min_point)
    
    # Convert to list of [a, b]
    result = [[a, b] for a, b in point_set]
    
    # Verify we have exactly 32 distinct points
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
    
    return result
# EVOLVE-BLOCK-END
