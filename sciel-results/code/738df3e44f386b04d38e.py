import math

def construct():
    n = 32
    
    def to_cartesian(a, b):
        return (a + 0.5 * b, b * math.sqrt(3) / 2)
    
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2

    # Generate points in rings 0, 1, 2
    points_by_ring = {0: [], 1: [], 2: [], 3: []}
    
    max_k = 3
    for a in range(-max_k, max_k + 1):
        for b in range(-max_k, max_k + 1):
            d = hex_dist(a, b)
            if d <= max_k:
                points_by_ring[d].append((a, b))
                
    # R0, R1, R2 are fixed
    selected_points = []
    for k in range(3):
        selected_points.extend(points_by_ring[k])
        
    # We need 13 more points from Ring 3
    ring3 = points_by_ring[3]
    # Sort ring3 by angle
    ring3.sort(key=lambda p: math.atan2(to_cartesian(p[0], p[1])[1], to_cartesian(p[0], p[1])[0]))
    
    # Precompute neighbors in R2 for each point in Ring 3
    # Neighbors of (a,b) are (a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1)
    r2_set = set(points_by_ring[2])
    
    def count_r2_neighbors(a, b):
        count = 0
        if (a+1, b) in r2_set: count += 1
        if (a-1, b) in r2_set: count += 1
        if (a, b+1) in r2_set: count += 1
        if (a, b-1) in r2_set: count += 1
        if (a+1, b-1) in r2_set: count += 1
        if (a-1, b+1) in r2_set: count += 1
        return count
        
    weights = [count_r2_neighbors(p[0], p[1]) for p in ring3]
    
    # Find best contiguous block of 13 in the circular ring3
    m = len(ring3) # 18
    k_needed = 13
    best_start = 0
    best_sum = -1
    
    for i in range(m):
        s = 0
        for j in range(k_needed):
            s += weights[(i + j) % m]
        if s > best_sum:
            best_sum = s
            best_start = i
            
    # Select the best block
    for j in range(k_needed):
        selected_points.append(ring3[(best_start + j) % m])
        
    # Ensure we have exactly 32 points
    # (R0:1, R1:6, R2:12, R3:13) -> 32
    
    result = [[a, b] for a, b in selected_points]
    
    return result
