import time

def construct():
    # Deadline guard, though this deterministic solution is instant.
    start_time = time.time()
    deadline = start_time + 140

    # Function to calculate squared distance between two points (a1, b1) and (a2, b2)
    # on triangular lattice with omega = exp(i*pi/3).
    # Distance squared = da^2 + db^2 + da*db
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + db * db + da * db

    # Generate all points in a hexagon of radius R=3
    # Axial coordinates (a, b) with |a| <= R, |b| <= R, |a+b| <= R
    R = 3
    hex_points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if abs(a + b) <= R:
                hex_points.append([a, b])
    
    # We have 37 points. We need 32.
    # Calculate the degree (number of neighbors at distance 1) for each point
    # within the full hexagon.
    degrees = {}
    point_index = {p: i for i, p in enumerate(hex_points)}
    
    for i, p in enumerate(hex_points):
        deg = 0
        for j, q in enumerate(hex_points):
            if i == j:
                continue
            if dist_sq(p, q) == 1:
                deg += 1
        degrees[p] = deg
        
    # Sort points by degree, ascending. We want to remove points with lowest degree.
    # This preserves the high-connectivity core.
    sorted_points = sorted(hex_points, key=lambda x: degrees[x])
    
    # Remove the first 5 points (lowest degree)
    # These are likely the corners of the hexagon.
    points_to_keep = sorted_points[5:]
    
    # Verify we have 32 points
    if len(points_to_keep) != 32:
        # Fallback: just take first 32
        points_to_keep = hex_points[:32]
        
    # Ensure all coordinates are within |a|, |b| <= 64
    # R=3 is well within bounds.
    
    # Calculate final edge count to ensure validity (optional for return, but good for sanity)
    # The task asks to return the points. The evaluator calculates the score.
    
    return [[a, b] for a, b in points_to_keep]
