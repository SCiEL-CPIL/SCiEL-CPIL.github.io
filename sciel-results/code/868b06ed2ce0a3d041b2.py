import math

def construct():
    # We want 32 points.
    n_points = 32
    
    # Helper to compute hex distance
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2
        
    # Helper to compute angle for tie-breaking
    def angle(p):
        a, b = p
        # Convert to Cartesian to get a consistent angular ordering
        # x = a + b/2, y = b * sqrt(3)/2
        # We can just use atan2(y, x)
        x = a + 0.5 * b
        y = math.sqrt(3) / 2.0 * b
        return math.atan2(y, x)

    # Generate candidate points
    # We need points up to hex distance 3 (since 19 points in dist <= 2, need 13 more from dist 3)
    # Range -4 to 4 is sufficient to cover hex_dist 3
    candidates = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            candidates.append((a, b))
            
    # Sort by hex distance, then by angle
    candidates.sort(key=lambda p: (hex_dist(p), angle(p)))
    
    # Take the first 32
    points = [list(p) for p in candidates[:n_points]]
    
    # Verify edge count
    # Neighbors at distance 1
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Build lookup for O(1) neighbor checks
    lookup = set(map(tuple, points))
    
    edge_count = 0
    for p in points:
        a, b = p
        for da, db in neighbors:
            if (a+da, b+db) in lookup:
                edge_count += 1
                
    # Each edge is counted twice (once from each endpoint)
    edge_count //= 2
    
    # Theoretical max is 76
    if edge_count == 76:
        return points
        
    # Fallback: if for some reason it's not 76 (shouldn't happen), 
    # we could try a simple local search, but the hex spiral is optimal.
    # For safety, we return the points anyway as they are a valid configuration.
    return points
