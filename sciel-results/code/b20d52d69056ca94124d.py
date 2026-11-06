import math
import time
import random

def construct():
    """
    Returns a list of 32 distinct integer pairs [a, b] representing points on the triangular lattice
    that maximize the number of unit-distance edges (target: 76).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Distance squared on triangular lattice: (a,b) represents a + b*omega, omega=exp(i*pi/3)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            a1, b1 = points[i]
            for j in range(i+1, n):
                a2, b2 = points[j]
                da = a1 - a2
                db = b1 - b2
                if da*da + da*db + db*db == 1:
                    count += 1
        return count

    # Hex distance from origin
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Generate all points in a box
    all_points = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if hex_dist([a, b]) <= 3:
                all_points.append([a, b])

    # Inner: radius <= 2 (19 points)
    inner = [p for p in all_points if hex_dist(p) <= 2]
    # Outer: radius == 3 (18 points)
    outer = [p for p in all_points if hex_dist(p) == 3]

    # We need 32 total, so pick 13 from outer
    # Start with inner (19 points), then add 13 outer points to maximize edges
    
    # Order outer points by angle to get perimeter order
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    outer_sorted = sorted(outer, key=angle)
    n_out = len(outer_sorted)
    k_needed = 32 - len(inner)  # 13

    # Evaluate all contiguous blocks of 13
    best_set = None
    best_edges = -1

    for start in range(n_out):
        block = [outer_sorted[(start + i) % n_out] for i in range(k_needed)]
        candidate = inner + block
        edges = count_edges(candidate)
        if edges > best_edges:
            best_edges = edges
            best_set = candidate

    # Local search: try swapping points to increase edge count
    # We allow swapping any point in the set with any point in the candidate pool (inner+outer)
    # But we must maintain exactly 32 distinct points.
    # Since 'inner' is fixed in the initial heuristic, we might want to allow swapping inner points too.
    # Let's define the pool of all 37 points (19 inner + 18 outer).
    pool = all_points # 37 points
    
    # Current best set is a list of 32 points
    current_set = best_set[:]
    current_edges = best_edges
    current_set_tuples = set(tuple(p) for p in current_set)
    
    # Simulated Annealing / Hill Climbing
    # We will try random swaps of a point in current_set with a point in pool not in current_set
    # If it improves, accept. If not, accept with some probability (annealing) or just reject (greedy).
    # Given the small search space, greedy hill climbing with random restarts or just greedy is good.
    # Let's do a few passes of greedy hill climbing.
    
    def get_unselected(pool, selected_set):
        return [p for p in pool if tuple(p) not in selected_set]
    
    improved = True
    # Limit iterations to stay within time
    max_iterations = 10000
    iter_count = 0
    
    while improved and time.time() < deadline - 5 and iter_count < max_iterations:
        improved = False
        iter_count += 1
        unselected = get_unselected(pool, current_set_tuples)
        
        # Shuffle indices to avoid bias
        indices = list(range(32))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            p_old = current_set[i]
            p_old_tuple = tuple(p_old)
            
            # Try swapping with each unselected point
            # To speed up, we can pre-calculate edges lost by removing p_old
            edges_lost = 0
            for j in range(32):
                if j == i: continue
                if dist_sq(p_old, current_set[j]) == 1:
                    edges_lost += 1
            
            for p_new in unselected:
                p_new_tuple = tuple(p_new)
                # Edges gained by adding p_new
                edges_gained = 0
                for j in range(32):
                    if j == i: continue
                    if dist_sq(p_new, current_set[j]) == 1:
                        edges_gained += 1
                
                if edges_gained > edges_lost:
                    # Accept swap
                    current_set[i] = p_new
                    current_set_tuples.discard(p_old_tuple)
                    current_set_tuples.add(p_new_tuple)
                    current_edges += (edges_gained - edges_lost)
                    improved = True
                    # Re-calculate unselected list as it changed
                    unselected = get_unselected(pool, current_set_tuples)
                    break # Restart loop for this i with new set? Or just continue? 
                    # Usually better to restart the scan to be safe, but for speed, just continue.
                    # However, 'unselected' changed, so the inner loop is invalid for next p_new?
                    # Actually, we broke the inner loop, so we go to next i.
                    # But 'unselected' is now stale for the next i if we don't update it.
                    # We updated it. So it's fine.
                    pass

    # Final check
    if len(current_set) != 32:
        return best_set
        
    # Verify distinctness and bounds
    if len(set(tuple(p) for p in current_set)) != 32:
        return best_set
        
    for p in current_set:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return best_set
            
    # Return the better of the initial heuristic and the refined set
    if current_edges >= best_edges:
        return current_set
    else:
        return best_set

# Set a seed for reproducibility in local testing, but random is fine for evaluator
random.seed(42)

if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Quick edge count for verification
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db
    count = 0
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if dist_sq(res[i], res[j]) == 1:
                count += 1
    print(count)
