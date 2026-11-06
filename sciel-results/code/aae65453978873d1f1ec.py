import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def hex_dist(a, b):
        return max(abs(a), abs(b), abs(a + b))
    
    def point_dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
    
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if point_dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Generate all points with hex_dist <= 4 (61 points total)
    all_pts = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if hex_dist(a, b) <= 4:
                all_pts.append((a, b))
    
    # Initial candidate: Best subset of 32 from R<=3 hexagon
    inner_pts = [p for p in all_pts if hex_dist(p[0], p[1]) <= 3]
    # inner_pts has 37 points.
    
    best_score = 0
    best_result = []
    
    outer_pts_3 = [p for p in inner_pts if hex_dist(p[0], p[1]) == 3]
    
    # Search for the best subset of 32 points from the 37-point hexagon
    for remove_combo in combinations(outer_pts_3, 5):
        if time.time() > deadline:
            break
        remove_set = set(remove_combo)
        candidate = [p for p in inner_pts if p not in remove_set]
        score = count_edges(candidate)
        if score > best_score:
            best_score = score
            best_result = candidate
            
    # Now, try to improve best_result (32 points) by swapping with points from R=4
    # Pool of candidates for swap-in: points in R=4
    outer_pts_4 = [p for p in all_pts if hex_dist(p[0], p[1]) == 4]
    
    # Current set
    current_set = set(best_result)
    current_score = best_score
    
    # Hill climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try swapping each point in current_set with each point in outer_pts_4
        # That's 32 * 24 = 768 swaps.
        # Calculate delta score.
        
        # Precompute neighbors of each point in current_set within the full lattice?
        # Or just calculate score change.
        
        # To speed up, we can calculate the score of the new set.
        # But 768 * count_edges(32) is 768 * 496 ops = ~380k ops. Fast.
        
        best_swap_score = current_score
        best_swap_in = None
        best_swap_out = None
        
        for p_out in best_result:
            for p_in in outer_pts_4:
                if p_in in current_set:
                    continue
                # Create new set
                # Remove p_out, add p_in
                # Calculate score
                # Optimization: 
                # Score change = (edges gained by p_in) - (edges lost by p_out) + (interaction change)
                # Edges lost by p_out: count neighbors of p_out in (current_set - {p_out})
                # Edges gained by p_in: count neighbors of p_in in (current_set - {p_out})
                # Note: p_in and p_out might be neighbors?
                
                neighbors_out = 0
                neighbors_in = 0
                for p in current_set:
                    if p == p_out:
                        continue
                    d_out = point_dist_sq(p_out, p)
                    if d_out == 1:
                        neighbors_out += 1
                    d_in = point_dist_sq(p_in, p)
                    if d_in == 1:
                        neighbors_in += 1
                
                # Check if p_in and p_out are neighbors
                if point_dist_sq(p_in, p_out) == 1:
                    # If they are neighbors, the edge between them is removed (lost) and not added.
                    # Wait, p_out is removed. p_in is added.
                    # The edge (p_in, p_out) is not in the new set.
                    # In the old set, (p_out, p_in) was NOT an edge because p_in was not in the set.
                    # So no change for that pair.
                    # The calculation above handles this:
                    # neighbors_out counts edges from p_out to others in set.
                    # neighbors_in counts edges from p_in to others in set (excluding p_out).
                    # If p_in and p_out were neighbors, that edge is lost (was not there) and not gained.
                    # Actually, if p_in was NOT in the set, the edge (p_in, p_out) didn't exist.
                    # So we just lose edges connected to p_out and gain edges connected to p_in.
                    pass
                
                delta = neighbors_in - neighbors_out
                new_score = current_score + delta
                
                if new_score > best_swap_score:
                    best_swap_score = new_score
                    best_swap_in = p_in
                    best_swap_out = p_out
        
        if best_swap_in is not None:
            # Apply swap
            current_set.remove(best_swap_out)
            current_set.add(best_swap_in)
            best_result = list(current_set)
            current_score = best_swap_score
            improved = True
            best_score = max(best_score, current_score)
            
    # Additional local search: try moving points to nearby empty positions
    # This is inspired by the "move to nearby empty position" strategy from inspiration 1
    # which can escape local optima that the swap-only approach might get stuck in.
    
    # Build a set for O(1) lookup
    point_set = set((p[0], p[1]) for p in best_result)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_result)):
            if time.time() > deadline:
                break
            ax, ay = best_result[i][0], best_result[i][1]
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
                    for j in range(len(best_result)):
                        if j == i:
                            continue
                        if point_dist_sq(old_point, best_result[j]) == 1:
                            edges_lost += 1
                    
                    # Edges gained: edges from new_point to all other points
                    edges_gained = 0
                    for j in range(len(best_result)):
                        if j == i:
                            continue
                        if point_dist_sq(new_point, best_result[j]) == 1:
                            edges_gained += 1
                    
                    delta = edges_gained - edges_lost
                    if delta > 0:
                        # Perform the move
                        best_result[i] = [nx, ny]
                        point_set.remove(old_point)
                        point_set.add(new_point)
                        current_score += delta
                        best_score = max(best_score, current_score)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return [[a, b] for a, b in best_result]
