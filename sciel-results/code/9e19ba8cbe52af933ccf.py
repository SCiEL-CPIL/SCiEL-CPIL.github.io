import math

def construct():
    # Generate points on triangular lattice
    # Axial coordinates (a, b)
    # Distance squared: a^2 + ab + b^2
    # Hex distance: (|a| + |b| + |a+b|) / 2
    
    points = []
    # Start with origin
    points.append((0, 0))
    
    # Add rings
    # We need 32 points.
    # Ring 0: 1 point.
    # Ring 1: 6 points. Total 7.
    # Ring 2: 12 points. Total 19.
    # Ring 3: 18 points. We need 13 more.
    
    # Generate all points in rings 1, 2, 3
    candidates = []
    for r in range(1, 4):
        # Points in ring r
        # We can iterate a, b in a box and filter by hex distance
        for a in range(-r, r+1):
            for b in range(-r, r+1):
                if a == 0 and b == 0:
                    continue
                # Calculate hex distance
                # d = (|a| + |b| + |a+b|) // 2
                d = (abs(a) + abs(b) + abs(a+b)) // 2
                if d == r:
                    candidates.append((a, b, r))
                    
    # Sort candidates by ring, then by angle or just index to ensure deterministic order
    # For ring 3, we want to pick points that connect well.
    # First, add all points from rings 1 and 2.
    current_points = [(0,0)]
    for p in candidates:
        if p[2] < 3:
            current_points.append((p[0], p[1]))
            
    # Now we have 19 points.
    # We need 13 more from ring 3.
    # To maximize edges, we should pick points that have the most neighbors in current_points.
    # Initially, all ring 3 points have 2 neighbors in ring 2.
    # So we pick one, then the next should prefer neighbors in ring 3.
    
    remaining = [p for p in candidates if p[2] == 3]
    
    # Helper to count neighbors
    def count_neighbors(p, point_set):
        a, b = p
        count = 0
        for pa, pb in point_set:
            da = a - pa
            db = b - pb
            if da*da + da*db + db*db == 1:
                count += 1
        return count

    # Greedy selection
    while len(current_points) < 32 and remaining:
        best_p = None
        best_score = -1
        for p in remaining:
            score = count_neighbors(p, current_points)
            if score > best_score:
                best_score = score
                best_p = p
        if best_p:
            current_points.append((best_p[0], best_p[1]))
            remaining.remove(best_p)
        else:
            break
            
    # Convert to list of lists
    return [[a, b] for a, b in current_points]
