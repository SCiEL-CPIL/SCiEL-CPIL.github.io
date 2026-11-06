import time
import math

def construct():
    """
    Constructs 32 points on the triangular lattice maximizing the number of unit distances.
    Uses a hexagonal spiral as a seed and improves it with local search.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    # 1. Generate initial solution (Hex Spiral)
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            candidates.append((a, b))
    
    # Sort by hex distance, then by angle to create a deterministic spiral-like ordering
    candidates.sort(key=lambda p: (hex_dist(p), angle(p)))
    
    # Select the 32 closest points
    initial_points = [p for p in candidates[:32]]
    
    best_points = [p[:] for p in initial_points]
    best_score = count_edges(best_points)
    
    # 2. Local Search
    # We will perform a local search where we try to move each point to a nearby empty position.
    # This is a form of 1-opt or 2-opt (swap with empty).
    
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    # Build a set for O(1) lookup
    point_set = set((p[0], p[1]) for p in current_points)
    
    # Search radius for moving a point
    search_radius = 2
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_points)):
            if time.time() > deadline:
                break
            ax, ay = current_points[i][0], current_points[i][1]
            
            # Try all positions within search_radius of current position
            for da in range(-search_radius, search_radius + 1):
                for db in range(-search_radius, search_radius + 1):
                    if da == 0 and db == 0:
                        continue
                    nx, ny = ax + da, ay + db
                    # Skip if same position
                    if (nx, ny) == (ax, ay):
                        continue
                    # Skip if already occupied
                    if (nx, ny) in point_set:
                        continue
                    # Check bounds
                    if abs(nx) > 64 or abs(ny) > 64:
                        continue
                    
                    # Calculate change in edge count
                    # Edges lost: edges from old_point to all other points
                    edges_lost = 0
                    for j in range(len(current_points)):
                        if j == i:
                            continue
                        if dist_sq((ax, ay), current_points[j]) == 1:
                            edges_lost += 1
                    
                    # Edges gained: edges from new_point to all other points
                    edges_gained = 0
                    for j in range(len(current_points)):
                        if j == i:
                            continue
                        if dist_sq((nx, ny), current_points[j]) == 1:
                            edges_gained += 1
                    
                    delta = edges_gained - edges_lost
                    if delta > 0:
                        # Perform the move
                        current_points[i] = [nx, ny]
                        point_set.remove((ax, ay))
                        point_set.add((nx, ny))
                        current_score += delta
                        improved = True
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                        
                        # Break out of loops to restart scan with new configuration
                        # This is a simple steepest ascent / hill climbing approach
                        break
                if improved:
                    break
            if improved:
                break
                
    return [[a, b] for a, b in best_points]
