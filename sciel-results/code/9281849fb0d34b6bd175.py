"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We need 32 points on the triangular lattice.
    # The triangular lattice has points (a, b) with neighbors:
    # (a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)
    # Distance between (a1,b1) and (a2,b2) is sqrt((a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2)
    # We want to maximize the number of pairs at distance exactly 1 (i.e., adjacent).
    # Harborth's theorem says the optimum is floor(3n - sqrt(12n - 3)) = 76 for n=32.
    
    # Strategy: Use a hexagonal/spiral pattern that maximizes adjacency.
    # A filled hexagon of radius r has 1 + 3*r*(r+1) points.
    # r=2: 1+3*2*3 = 19
    # r=3: 1+3*3*4 = 37
    # So a hexagon of radius 3 has 37 points, and we need 32.
    # We can take a hexagon of radius 3 and remove 5 points from the outer ring.
    # The outer ring (ring 3) has 6*3 = 18 points.
    # Removing 5 points from the outer ring should still give us many adjacencies.
    
    # Let's generate the hexagon of radius 3 and pick 32 points.
    # Points in hexagon of radius r: all (a,b) with a^2 + a*b + b^2 <= r^2
    
    # Actually, let's think more carefully. The hexagon of radius r in the triangular lattice
    # (using axial coordinates) consists of all (a, b) such that max(|a|, |b|, |a+b|) <= r.
    # Wait, that's for the hex grid with a different coordinate system.
    
    # Let me use the standard hexagonal distance: dist((a1,b1), (a2,b2)) = 
    #   (|a1-a2| + |a1-a2 + b1-b2| + |b1-b2|) / 2
    # But actually, for our lattice, the distance squared is a^2 + ab + b^2 where (a,b) is the difference.
    # Two points are adjacent if their difference is one of the 6 neighbor vectors.
    
    # Let's just generate a large set of candidate points and use a greedy or optimization approach.
    
    import time
    deadline = time.time() + 10  # Safety margin
    
    # Generate candidate points: hexagon of radius 4 (has 1 + 3*4*5 = 61 points)
    # Actually radius 3 has 37, which is enough.
    
    # Let's generate all points in a hexagon of radius 3 (axial coordinates where dist = (|a|+|b|+|a+b|)/2)
    # But our coordinates are different. Let me just enumerate a square region.
    
    # Points (a, b) with |a| <= 6, |b| <= 6, and a^2 + a*b + b^2 <= 9 (hexagon radius 3)
    candidates = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if a*a + a*b + b*b <= 9:
                candidates.append((a, b))
    
    # This gives 37 points. We need to select 32 of them to maximize adjacencies.
    # A full hexagon of radius 3 has all points, and the number of adjacent pairs is:
    # Each point in the interior has 6 neighbors, but we count each edge once.
    # For a hexagon of radius r, the number of edges is 3*r*(r+1) + ... 
    # Actually, for r=3: let's just compute it.
    
    # Let's try all 37 points first and see how many adjacencies we get.
    # Then we can remove 5 points that are on the boundary to minimize loss of adjacencies.
    
    def count_adjacencies(points):
        point_set = set(points)
        count = 0
        for p in points:
            a, b = p
            for da, db in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                if (a+da, b+db) in point_set:
                    count += 1
        return count // 2  # Each edge counted twice
    
    # Start with all 37 points
    all_points = list(candidates)
    initial_adj = count_adjacencies(all_points)
    
    # Now remove 5 points one at a time, choosing the one that loses the fewest adjacencies
    selected = set(all_points)
    for _ in range(5):
        best_to_remove = None
        best_loss = float('inf')
        for p in list(selected):
            # Count how many neighbors of p are in selected
            a, b = p
            neighbors_in = sum(1 for da, db in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] 
                              if (a+da, b+db) in selected)
            if neighbors_in < best_loss:
                best_loss = neighbors_in
                best_to_remove = p
        if best_to_remove is not None:
            selected.remove(best_to_remove)
    
    result = [[a, b] for a, b in selected]
    
    # Verify we have exactly 32 distinct points
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    
    # Verify all coordinates are within bounds
    for a, b in result:
        assert abs(a) <= 64 and abs(b) <= 64
    
    # Verify mutual distance at least 1 (i.e., no two points are the same, which is already ensured)
    # Distance between distinct lattice points is at least 1 by definition of the lattice.
    
    return result
# EVOLVE-BLOCK-END
