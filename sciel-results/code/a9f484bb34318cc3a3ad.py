import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0  # 10s is plenty for 32 points
    
    # State: list of points (a, b)
    points = [(0, 0)]
    point_set = {(0, 0)}
    
    # Target size
    n = 32
    
    # Directions for neighbors
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Helper: distance squared on triangular lattice
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
    
    # Helper: count edges among a list of points
    def count_edges(pts):
        count = 0
        m = len(pts)
        for i in range(m):
            for j in range(i+1, m):
                if dist_sq(pts[i], pts[j]) == 1:
                    count += 1
        return count
    
    # Helper: hex distance from origin
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a+b)) // 2
    
    # Build the known optimal hex-spiral configuration:
    # Core: all points with hex_dist <= 2 (19 points)
    # Ring 3: 13 contiguous points from the ring of 18
    core = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if hex_dist(a, b) <= 2:
                core.add((a, b))
    
    # Generate ring 3 points
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) == 3:
                ring3.append((a, b))
    
    # Sort ring3 by angle to form the cycle
    def angle(p):
        a, b = p
        x = a + b / 2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
    
    ring3.sort(key=angle)
    r3_len = len(ring3)  # should be 18
    
    # Evaluate all 18 contiguous blocks of 13 from ring3
    best_score = -1
    best_points = None
    
    for i in range(r3_len):
        subset = [ring3[(i + k) % r3_len] for k in range(13)]
        full_set = list(core) + subset
        
        if len(set(full_set)) != 32:
            continue
        if any(abs(p[0]) > 64 or abs(p[1]) > 64 for p in full_set):
            continue
        
        edges = count_edges(full_set)
        if edges > best_score:
            best_score = edges
            best_points = [list(p) for p in full_set]
    
    # If we found the optimal 76, return immediately
    if best_score == 76 and best_points is not None:
        return best_points
    
    # If not 76, fall back to the greedy construction (should not happen)
    if best_points is not None:
        # Use the best block found
        points = [tuple(p) for p in best_points]
        point_set = set(points)
    else:
        # Greedy fallback
        points = [(0, 0)]
        point_set = {(0, 0)}
        while len(points) < n:
            best_candidate = None
            best_conn = -1
            best_dist = float('inf')
            
            candidates = set()
            for a, b in points:
                for da, db in dirs:
                    na, nb = a + da, b + db
                    if (na, nb) not in point_set:
                        candidates.add((na, nb))
            
            for a, b in candidates:
                conn_count = 0
                for da, db in dirs:
                    if (a + da, b + db) in point_set:
                        conn_count += 1
                hex_d = hex_dist(a, b)
                if conn_count > best_conn or (conn_count == best_conn and hex_d < best_dist):
                    best_conn = conn_count
                    best_dist = hex_d
                    best_candidate = (a, b)
            
            if best_candidate is None:
                break
            points.append(best_candidate)
            point_set.add(best_candidate)
            
            if time.time() > deadline:
                break
    
    return [list(p) for p in points]
