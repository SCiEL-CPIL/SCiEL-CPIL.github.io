import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s safety margin

    # Distance squared on triangular lattice
    # Points are a + b*omega, omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # (a1 + b1*omega) - (a2 + b2*omega) = (a1-a2) + (b1-b2)*omega
    # |z|^2 = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)(b1-b2)
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

    # Generate candidates in a box
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            candidates.append([a, b])

    # Sort by hex distance, then by angle
    candidates.sort(key=lambda p: (hex_dist(p), angle(p)))

    # Take first 32 as initial solution
    result = [p[:] for p in candidates[:32]]
    best_edges = count_edges(result)

    # Local search: try to swap points to increase edge count
    # The 6 neighbor offsets
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]

    # Build a set for O(1) lookup
    point_set = set((p[0], p[1]) for p in result)

    # Precompute which points are in the result
    in_result = {}
    for p in result:
        in_result[(p[0], p[1])] = True

    # Try local moves: for each point, try moving it to a nearby empty position
    # that increases the edge count
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(result)):
            if time.time() > deadline:
                break
            ax, ay = result[i][0], result[i][1]
            # Try all positions within hex distance 2 of current position
            for da in range(-2, 3):
                for db in range(-2, 3):
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
                    # Try the move
                    # Remove old position
                    old_point = (ax, ay)
                    new_point = (nx, ny)
                    
                    # Calculate change in edge count
                    # Edges lost: edges from old_point to all other points
                    edges_lost = 0
                    for j in range(len(result)):
                        if j == i:
                            continue
                        if dist_sq(old_point, result[j]) == 1:
                            edges_lost += 1
                    
                    # Edges gained: edges from new_point to all other points
                    edges_gained = 0
                    for j in range(len(result)):
                        if j == i:
                            continue
                        if dist_sq(new_point, result[j]) == 1:
                            edges_gained += 1
                    
                    delta = edges_gained - edges_lost
                    if delta > 0:
                        # Perform the move
                        result[i] = [nx, ny]
                        point_set.remove(old_point)
                        point_set.add(new_point)
                        best_edges += delta
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Final verification
    final_edges = count_edges(result)
    
    return result
