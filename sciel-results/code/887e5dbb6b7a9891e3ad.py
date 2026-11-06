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
    
    while len(points) < n:
        best_candidate = None
        best_score = -1
        best_dist = float('inf')
        
        # Generate candidates: neighbors of all current points
        candidates = set()
        for a, b in points:
            for da, db in dirs:
                na, nb = a + da, b + db
                if (na, nb) not in point_set:
                    candidates.add((na, nb))
        
        # Evaluate candidates
        for a, b in candidates:
            # Count connections to existing points
            conn_count = 0
            for da, db in dirs:
                if (a + da, b + db) in point_set:
                    conn_count += 1
            
            # Score: maximize connections
            # Tie-break: minimize distance to center (0,0) to keep compact
            # Hex distance to origin
            dist = max(abs(a), abs(b), abs(a+b)) # Approximation, or exact hex dist
            # Exact hex dist: (abs(a) + abs(b) + abs(a+b)) // 2
            hex_dist = (abs(a) + abs(b) + abs(a+b)) // 2
            
            if conn_count > best_score or (conn_count == best_score and hex_dist < best_dist):
                best_score = conn_count
                best_dist = hex_dist
                best_candidate = (a, b)
        
        if best_candidate is None:
            break
            
        a, b = best_candidate
        points.append((a, b))
        point_set.add((a, b))
        
        if time.time() > deadline:
            break
            
    # Return list of lists
    return [list(p) for p in points]
