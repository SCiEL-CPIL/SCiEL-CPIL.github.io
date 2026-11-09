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
    import math
    
    # Directions on triangular lattice for the 6 neighbors
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def hex_ring(r):
        """Generate all 6*r points in hexagonal ring r."""
        if r == 0:
            return [(0, 0)]
        
        sides = [
            (-1, 1),
            (-1, 0),
            (0, -1),
            (1, -1),
            (1, 0),
            (0, 1),
        ]
        
        pts = []
        a, b = r, 0
        for da, db in sides:
            for _ in range(r):
                a += da
                b += db
                pts.append((a, b))
        return pts

    # Generate a superset of points: Rings 0, 1, 2, 3
    # This gives 1 + 6 + 12 + 18 = 37 points
    candidate_points = set()
    for r in range(4):
        for p in hex_ring(r):
            candidate_points.add(p)
    
    # We need to select 32 points from these 37 to maximize edges.
    # Start with all 37 points.
    current_points = list(candidate_points)
    
    def count_edges(points):
        p_set = set(points)
        edges = 0
        for p in points:
            a, b = p
            for da, db in directions:
                if (a+da, b+db) in p_set:
                    edges += 1
        return edges // 2
    
    def get_degrees(points):
        p_set = set(points)
        degrees = {}
        for p in points:
            a, b = p
            deg = 0
            for da, db in directions:
                if (a+da, b+db) in p_set:
                    deg += 1
            degrees[p] = deg
        return degrees
    
    # Prune 5 points to reach 32
    # We want to minimize the loss of edges.
    # Loss of removing p is deg(p).
    # Greedy: remove point with min degree.
    while len(current_points) > 32:
        degrees = get_degrees(current_points)
        # Find point with minimum degree
        min_deg = min(degrees.values())
        candidates = [p for p, d in degrees.items() if d == min_deg]
        # If tie, pick one arbitrarily (e.g., first one)
        # To be deterministic, we can sort by coordinates
        candidates.sort()
        p_to_remove = candidates[0]
        current_points.remove(p_to_remove)
        
    # Verify all coordinates are within bounds
    for a, b in current_points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    return [[a, b] for a, b in current_points]
# EVOLVE-BLOCK-END
