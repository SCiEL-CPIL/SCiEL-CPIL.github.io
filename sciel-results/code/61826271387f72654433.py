"""
Harborth spiral for 32 pennies on triangular lattice.
Uses hexagonal ring growth with optimized selection of the final partial ring
to maximize the number of unit-distance pairs (edges).
"""

def construct():
    """
    Return exactly 32 distinct integer pairs [a, b] with |a|, |b| <= 64
    denoting lattice points a + b*omega on the triangular lattice,
    maximizing the number of pairs at distance exactly 1.
    """
    
    # The 6 neighbor directions on the triangular lattice
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def hex_ring(r):
        """Generate all 6*r points in hexagonal ring r (r >= 1)."""
        sides = [
            (-1, 1),  # (r,0) to (0,r)
            (-1, 0),  # (0,r) to (-r,r)
            (0, -1),  # (-r,r) to (-r,0)
            (1, -1),  # (-r,0) to (0,-r)
            (1, 0),   # (0,-r) to (r,-r)
            (0, 1),   # (r,-r) to (r,0)
        ]
        pts = []
        a, b = r, 0
        for da, db in sides:
            for _ in range(r):
                a += da
                b += db
                pts.append((a, b))
        return pts
    
    def count_edges(point_set):
        """Count the number of pairs at distance exactly 1."""
        s = set(point_set)
        count = 0
        for (a, b) in s:
            for da, db in directions:
                if (a + da, b + db) in s:
                    count += 1
        return count // 2
    
    # Build the core: ring 0 + ring 1 + ring 2 = 1 + 6 + 12 = 19 points
    core = [(0, 0)]
    for r in range(1, 3):
        core.extend(hex_ring(r))
    
    # We need 32 - 19 = 13 more points from ring 3
    ring3 = hex_ring(3)
    n_needed = 32 - len(core)
    
    # Try all contiguous windows of length n_needed on ring3 (circular)
    # Also try the full ring3 if it fits (it has 18 points, we need 13)
    best_edges = -1
    best_subset = None
    
    # Contiguous windows on the circular ring
    m = len(ring3)
    for start in range(m):
        subset = [ring3[(start + i) % m] for i in range(n_needed)]
        total = core + subset
        e = count_edges(total)
        if e > best_edges:
            best_edges = e
            best_subset = subset
    
    # Also try some non-contiguous selections for robustness:
    # Greedy: pick points one by one that maximize marginal edge gain
    def greedy_select(core_pts, candidate_pts, n):
        selected = []
        remaining = list(candidate_pts)
        for _ in range(n):
            best_gain = -1
            best_idx = 0
            for i, p in enumerate(remaining):
                current = core_pts + selected
                e_before = count_edges(current)
                e_after = count_edges(current + [p])
                gain = e_after - e_before
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            selected.append(remaining.pop(best_idx))
        return selected
    
    greedy_subset = greedy_select(core, ring3, n_needed)
    greedy_edges = count_edges(core + greedy_subset)
    if greedy_edges > best_edges:
        best_edges = greedy_edges
        best_subset = greedy_subset
    
    points = core + best_subset
    points = points[:32]
    
    # Verify constraints
    for a, b in points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    assert len(set(points)) == 32, "Points are not distinct"
    
    return [[a, b] for a, b in points]
