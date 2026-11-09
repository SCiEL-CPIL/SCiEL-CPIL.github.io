"""Rewritten program: Harborth spiral for 32 pennies on triangular lattice."""
# EVOLVE-BLOCK-START
def construct():
    """
    Generate 32 lattice points (a, b) on the triangular lattice such that:
      - All points are distinct integer pairs with |a|, |b| <= 64
      - Mutual distance >= 1 (in Euclidean sense on the triangular lattice)
      - Number of pairs at distance exactly 1 is maximized (target: 76)
    
    The triangular lattice point a + b*omega where omega = exp(i*pi/3) = (1/2, sqrt(3)/2).
    Distance squared between (a1,b1) and (a2,b2) is:
      dx = a1 - a2, dy = b1 - b2
      dist^2 = dx^2 + dy^2 + dx*dy
    
    Adjacent pairs (distance 1) satisfy dx^2 + dy^2 + dx*dy = 1, which gives:
      (dx,dy) in {(1,0), (0,1), (1,-1), (-1,0), (0,-1), (-1,1)}
    
    We use a spiral growth pattern on the triangular lattice to achieve the
    Harborth optimum of 76 edges for n=32.
    
    Strategy: Build a compact hexagonal-ish cluster by growing a spiral.
    For the triangular lattice, a good compact shape is a "triangular number"
    arrangement or a hexagon. For n=32, we use a spiral that fills layers
    of a hexagonal region.
    
    The hexagonal numbers are 1, 7, 19, 37, ... so 32 falls between 19 and 37.
    We start with the center and add layers, stopping when we have 32 points.
    
    A better approach: use the known optimal configuration. For n=32 on the
    triangular lattice, the optimum is 76 edges. A spiral/hexagonal packing
    achieves this.
    
    Let's generate points by expanding rings around the origin:
    Ring 0: (0,0) -> 1 point
    Ring 1: 6 points at distance 1 from origin
    Ring 2: 12 points at distance 2 from origin (but only those not already counted)
    Ring 3: 18 points...
    
    Cumulative: 1, 7, 19, 37...
    For 32 points: take ring 0 (1) + ring 1 (6) + ring 2 (12) = 19, then 13 more from ring 3.
    
    But we need to be careful about which points to pick from ring 3 to maximize
    internal edges. Let's use a systematic spiral.
    
    Alternative: Use a known optimal pattern. The "Harborth spiral" on the
    triangular lattice for n=32 can be constructed by a specific ordering.
    
    Let me use a different approach: generate all points in a bounding box,
    then select 32 that form a connected subgraph with maximum edges.
    
    Actually, the simplest correct approach: use the hexagonal spiral.
    """
    # Construct the optimal 32-point configuration directly.
    # Base: Hexagon of radius 2 (19 points).
    # Add 13 points from Ring 3 to minimize perimeter.
    
    points = set()
    
    # Add Ring 0
    points.add((0, 0))
    
    # Add Ring 1
    for a, b in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]:
        points.add((a, b))
        
    # Add Ring 2
    for a, b in [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1), 
                 (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1), (2, -1)]:
        # Note: (2,-1) is duplicate in my manual list, let's generate properly
        pass
        
    # Proper generation of Ring 2
    ring2 = []
    a, b = 2, 0
    sides = [(-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1)]
    for da, db in sides:
        for _ in range(2):
            a += da
            b += db
            ring2.append((a, b))
            
    for p in ring2:
        points.add(p)
        
    # We have 19 points. Need 13 more from Ring 3.
    # Generate Ring 3
    ring3 = []
    a, b = 3, 0
    for da, db in sides:
        for _ in range(3):
            a += da
            b += db
            ring3.append((a, b))
            
    # Select 13 points from Ring 3 that maximize edges.
    # The first 13 points in the spiral order are a good heuristic.
    for p in ring3[:13]:
        points.add(p)
        
    points = list(points)
    points = points[:32]
    
    for a, b in points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
        
    return [[a, b] for a, b in points]
# EVOLVE-BLOCK-END
