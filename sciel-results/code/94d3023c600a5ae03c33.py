import math
import time

def construct():
    start_time = time.time()
    
    def hex_dist(a, b):
        return max(abs(a), abs(b), abs(a + b))
    
    def point_dist_sq(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        da = a1 - a2
        db = b1 - b2
        return da*da + da*db + db*db
    
    # Generate all points with hex_dist <= 3 (37 points total)
    all_pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) <= 3:
                all_pts.append((a, b))
    
    # Sort by (hex_dist, angle) for spiral ordering
    def sort_key(p):
        a, b = p
        d = hex_dist(a, b)
        x = a + b * 0.5
        y = b * (math.sqrt(3) / 2)
        angle = math.atan2(y, x)
        return (d, angle)
    
    all_pts.sort(key=sort_key)
    
    # Try different subsets of 32 from the 37 points
    # The 37 points are sorted in spiral order. We want to remove 5 points.
    # Removing points from the outer ring might be best to keep the center dense.
    # Or removing points that are "far" from each other in the subset.
    
    # Let's just try all combinations of 5 points to remove?
    # 37 choose 5 is 3224496. That's a lot, but maybe we can prune.
    # Actually, let's just try removing the last 5 points (outermost).
    # Or try removing points from the outer ring (hex_dist == 3).
    
    # Points with hex_dist == 3 are the outer ring.
    # Count them: 6 * 3 = 18 points.
    # We need to remove 5.
    
    # Let's try a few heuristics:
    # 1. Remove the last 5 points in spiral order (outermost, specific angles)
    # 2. Remove 5 points from the outer ring that are most "spread out"
    # 3. Remove 5 points from the outer ring that are least connected
    
    # Let's calculate the score for the current best (first 32)
    result = all_pts[:32]
    best_score = 0
    for i in range(32):
        for j in range(i+1, 32):
            if point_dist_sq(result[i], result[j]) == 1:
                best_score += 1
    
    # Try removing different combinations of 5 points from the 37
    # We can be smarter: only consider removing points from the outer ring (hex_dist == 3)
    # If we remove a point from the inner part, we likely lose more connections.
    
    outer_pts = [p for p in all_pts if hex_dist(p[0], p[1]) == 3]
    inner_pts = [p for p in all_pts if hex_dist(p[0], p[1]) < 3]
    
    # Inner points count: 1 (center) + 6 (ring 1) + 12 (ring 2) = 19
    # Outer points count: 18
    # Total 37. We keep 32. So we keep all 19 inner, and 13 of the 18 outer.
    # So we remove 5 outer points.
    
    # Let's try all combinations of 5 outer points to remove.
    # 18 choose 5 = 8568. This is very fast.
    
    from itertools import combinations
    
    best_result = result
    best_score = 0
    
    # Precompute distances between all pairs? No, just calculate on the fly for each candidate.
    # For each combination of 5 to remove:
    #   candidate = inner_pts + [p for p in outer_pts if p not in remove_set]
    #   count edges
    
    # To speed up, we can precompute adjacency.
    # Let's map point to index in all_pts.
    # Actually, just iterating 8568 combinations and counting edges (32*31/2 = 496 pairs) is ~4M ops. Trivial.
    
    for remove_combo in combinations(outer_pts, 5):
        remove_set = set(remove_combo)
        candidate = [p for p in all_pts if p not in remove_set]
        
        score = 0
        for i in range(32):
            for j in range(i+1, 32):
                if point_dist_sq(candidate[i], candidate[j]) == 1:
                    score += 1
        
        if score > best_score:
            best_score = score
            best_result = candidate
    
    return [[a, b] for a, b in best_result]
