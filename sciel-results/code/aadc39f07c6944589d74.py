"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Generate the 37-point hexagon H_3 in axial coordinates
    # Point (a, b) is in H_3 if max(|a|, |b|, |a+b|) <= 3
    def in_hex(a, b, r):
        return max(abs(a), abs(b), abs(a + b)) <= r
    
    # All 37 points of H_3
    all_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if in_hex(a, b, 3):
                all_points.append((a, b))
    
    # Verify we have 37 points
    # assert len(all_points) == 37
    
    # Define neighbor relation
    def neighbors(p):
        a, b = p
        return [(a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)]
    
    # We need to remove 5 points to maximize remaining edges
    # Total edges in H_3: let's compute
    # For a hexagon of radius r, edges = 3r(2r+1) = 3*3*7 = 63? Let me compute directly.
    
    point_set = set(all_points)
    total_edges = 0
    for p in all_points:
        for q in neighbors(p):
            if q in point_set and p < q:  # count each edge once
                total_edges += 1
    
    # Now try removing each combination of 5 points and count edges
    # To be fast, try removing points with lowest "exposure" first
    # Strategy: greedily remove the point whose removal loses the fewest edges
    
    current_set = set(all_points)
    remaining_edges = total_edges
    
    for _ in range(5):
        best_removal = None
        best_remaining = -1
        for p in current_set:
            # Count edges that would be lost by removing p
            lost = 0
            for q in neighbors(p):
                if q in current_set:
                    lost += 1
            rem = remaining_edges - lost
            if rem > best_remaining:
                best_remaining = rem
                best_removal = p
        current_set.discard(best_removal)
        remaining_edges = best_remaining
    
    # Convert to list of [a, b] pairs
    result = [[a, b] for a, b in current_set]
    return result
# EVOLVE-BLOCK-END
