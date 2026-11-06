"""Rewritten program: direct construction of a 32-point triangular lattice cluster with 76 edges."""
import time

def construct():
    """
    Return 32 distinct integer pairs [a, b] on the triangular lattice (omega = exp(i*pi/3))
    such that mutual distances are at least 1, maximizing the number of pairs at distance exactly 1.
    Harborth's theorem says the optimum is 76.
    
    Strategy: Build a "hexagonal" cluster of 32 points on the triangular lattice that is known
    to achieve 76 edges. We use a known optimal configuration: a compact cluster.
    
    The triangular lattice in (a,b) coordinates: point (a,b) corresponds to a + b*omega.
    Distance between (a1,b1) and (a2,b2):
      dx = (a1-a2) + (b1-b2)*cos(pi/3) = (a1-a2) + (b1-b2)/2
      dy = (b1-b2)*sin(pi/3) = (b1-b2)*sqrt(3)/2
      dist^2 = dx^2 + dy^2 = (a1-a2 + (b1-b2)/2)^2 + 3*(b1-b2)^2/4
             = (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2
    
    So distance^2 = (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2
    Distance = 1 when (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2 = 1.
    
    The six neighbors of (0,0) are: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1).
    Check: (1,0): 1+0+0=1 ✓; (0,1): 0+0+1=1 ✓; (1,-1): 1-1+1=1 ✓.
    """
    # Known optimal 32-point configuration achieving 76 edges on triangular lattice.
    # We construct a hexagonal-like cluster.
    
    # One approach: use a "lozenge" or hexagonal shape.
    # Let's build the cluster layer by layer around the origin.
    # A hexagon of radius r has 3r(r+1)+1 points.
    # r=0: 1, r=1: 7, r=2: 19, r=3: 37.
    # So radius 2 gives 19 points, we need 13 more.
    # Radius 3 gives 37, we need to remove 5.
    
    # Let's try a different approach: use a known optimal set.
    # For n=32, the optimal cluster is well-studied. Let me construct one.
    
    # Points in (a,b) coordinates. Let's build a compact shape.
    # Start with hexagon radius 2 (19 points) and add 13 more optimally.
    
    # Hexagon radius 2: all (a,b) with max(|a|,|b|,|a+b|) <= 2... 
    # Actually in triangular lattice, the hexagon of radius r is:
    # |a| <= r, |b| <= r, |a-b| <= r (need to verify the coordinate system)
    
    # Let me just enumerate and verify.
    # Neighbors of (a,b): (a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1)
    
    # Let's build a cluster by starting from origin and adding points that maximize new edges.
    # Greedy approach with backtracking if needed.
    
    # Actually, let me use a known result: for the triangular lattice, optimal clusters
    # for small n are "hexagonal" or near-hexagonal.
    
    # Let me try: take the hexagon of radius 2 (19 pts) and extend it.
    # Hex radius 2 in (a,b): points where the "taxicab-like" distance from origin <= 2.
    # In triangular lattice coords, distance from (0,0) to (a,b) in lattice steps is:
    # The minimum number of steps is max(|a|, |b|, |a+b|) ... let me verify.
    # From (0,0) to (1,0): 1 step. max(1,0,1)=1 ✓
    # From (0,0) to (1,-1): 1 step. max(1,-1,0)=1 ✓ (using |a+b|=|0|=0, max(1,1,0)=1)
    # Wait, I need to be careful. Let me just use the direct formula.
    
    # Let me just hardcode a known-good 32-point set. I'll construct it carefully.
    
    # Build a hexagonal cluster. In (a,b) coords with the neighbor structure above:
    # Layer 0: (0,0) - 1 point
    # Layer 1: (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1) - 6 points, total 7
    # Layer 2: points at distance 2 from origin. These are:
    #   (2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2)  -- 6 "corner" directions
    #   (1,1),(-1,-1),(2,-1),(-2,1),(1,-2),(-1,2)  -- 6 "side" points
    #   Total layer 2: 12 points, cumulative: 19
    
    # Let me verify (1,1) is at distance 2: (1,1) from (0,0): 1+1+1=3, sqrt(3)≈1.73 ≠ 2.
    # Hmm, so (1,1) is NOT at lattice distance 2. Let me reconsider.
    
    # The lattice distance (number of steps) from (0,0) to (a,b):
    # We can move in 6 directions. The distance is the minimum number of steps.
    # For (a,b), the distance is: if a+b >= 0 and a >= 0 and b >= 0: a+b? No...
    # Let me just compute: to go from (0,0) to (a,b), we need net displacement (a,b).
    # Each step changes (a,b) by one of: (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1).
    # This is equivalent to: in terms of "hex coordinates", the distance is:
    # d = max(|a|, |b|, |a+b|) ... let me test.
    # (1,0): max(1,0,1)=1 ✓
    # (1,-1): max(1,1,0)=1 ✓  
    # (1,1): max(1,1,2)=2. So (1,1) is at distance 2. ✓
    # (2,0): max(2,0,2)=2 ✓
    # (0,2): max(0,2,2)=2 ✓
    # (2,-2): max(2,2,0)=2 ✓
    
    # So layer 2 (distance exactly 2): all (a,b) with max(|a|,|b|,|a+b|)=2
    # That's: 
    # a+b=2: (2,0),(1,1),(0,2) -- but need |a|<=2,|b|<=2
    # a+b=-2: (-2,0),(-1,-1),(0,-2)
    # a=2: (2,0),(2,-1),(2,-2)
    # a=-2: (-2,0),(-2,1),(-2,2)
    # b=2: (0,2),(-1,2),(-2,2)
    # b=-2: (0,-2),(1,-2),(2,-2)
    # Unique: (2,0),(1,1),(0,2),(-2,0),(-1,-1),(0,-2),(2,-1),(2,-2),(-2,1),(-2,2),(-1,2),(1,-2)
    # That's 12 points. Cumulative: 7+12=19. ✓
    
    # Now I need 13 more points from layer 3 (distance 3) to maximize edges.
    # Layer 3: max(|a|,|b|,|a+b|)=3, total 18 points.
    # I need to pick 13 of them to maximize total edges.
    
    # Let me just do a greedy: start with the 19-point hex, then add points from layer 3
    # one at a time, each time picking the one that adds the most new edges.
    
    points = set()
    # Layer 0
    points.add((0,0))
    # Layer 1
    for a,b in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
        points.add((a,b))
    # Layer 2
    for a,b in [(2,0),(1,1),(0,2),(-2,0),(-1,-1),(0,-2),(2,-1),(2,-2),(-2,1),(-2,2),(-1,2),(1,-2)]:
        points.add((a,b))
    
    # Layer 3 candidates
    layer3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            d = max(abs(a), abs(b), abs(a+b))
            if d == 3:
                layer3.append((a,b))
    
    # Greedy: add 13 points from layer3, each maximizing new edges
    def edge_count(pts):
        cnt = 0
        lst = list(pts)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                a1,b1 = lst[i]
                a2,b2 = lst[j]
                da = a1-a2
                db = b1-b2
                if da*da + da*db + db*db == 1:
                    cnt += 1
        return cnt
    
    current = set(points)
    candidates = list(layer3)
    
    for _ in range(13):
        best_pt = None
        best_gain = -1
        for pt in candidates:
            new_set = current | {pt}
            gain = edge_count(new_set) - edge_count(current)
            if gain > best_gain:
                best_gain = gain
                best_pt = pt
        if best_pt is not None:
            current.add(best_pt)
            candidates.remove(best_pt)
    
    # Verify we have 32 points
    assert len(current) == 32, f"Got {len(current)} points"
    
    # Verify all coordinates are within bounds
    for a,b in current:
        assert abs(a) <= 64 and abs(b) <= 64
    
    # Verify mutual distance at least 1
    lst = list(current)
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            a1,b1 = lst[i]
            a2,b2 = lst[j]
            da = a1-a2
            db = b1-b2
            d2 = da*da + da*db + db*db
            assert d2 >= 1, f"Points {lst[i]} and {lst[j]} too close: d2={d2}"
    
    # Count edges
    final_edges = edge_count(current)
    
    return [[a, b] for a, b in sorted(current)]

# Quick self-test
if __name__ == "__main__":
    result = construct()
    assert len(result) == 32
    # Count edges
    cnt = 0
    for i in range(32):
        for j in range(i+1, 32):
            a1,b1 = result[i]
            a2,b2 = result[j]
            da = a1-a2
            db = b1-b2
            if da*da + da*db + db*db == 1:
                cnt += 1
    print(f"Edges: {cnt}")
    print(f"Expected: 76")
