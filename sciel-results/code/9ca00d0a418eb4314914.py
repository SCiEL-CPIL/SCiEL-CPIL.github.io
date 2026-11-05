import random
import time
def construct():
    points = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
              (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1), 
              (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1),
              (3, 0), (3, -1), (3, -2), (3, -3), (2, -3), (1, -3), 
              (0, -3), (-1, -2), (-2, -1), (-3, 0), (-3, 1), (-3, 2), 
              (-3, 3)]
    
    # Define distance squared
    def dist_sq(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        da = a1 - a2
        db = b1 - b2
        return da*da + db*db + da*db

    # Count edges
    def count_edges(pts):
        count = 0
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(pts[i], pts[j]) == 1:
                    count += 1
        return count

    # Initial score
    current_score = count_edges(points)
    
    # Candidate pool: All points in rings 0 to 4
    # We can generate them or just use the ones we have plus some extra
    # To keep it simple, we use the points in the current set and their neighbors
    # Or just a fixed set of candidates from rings 0-4
    candidates = set()
    # Add all points in rings 0-4
    for r in range(5):
        if r == 0:
            candidates.add((0,0))
        else:
            # Generate ring r points
            # This is tedious to write inline, so we'll just use the points we already have 
            # and maybe expand the pool slightly if needed.
            # Given the small size, using the current 32 and their immediate neighbors is sufficient.
            pass
            
    # Let's build a proper candidate pool of size ~100 centered at 0
    # Hexagonal number system
    for a in range(-5, 6):
        for b in range(-5, 6):
            # Check if within hex radius 4
            # Hex distance: max(|a|, |b|, |a+b|) is not correct for axial?
            # For axial coords (a,b) where point is a + b*omega:
            # Distance is max(|a|, |b|, |a+b|) ? No.
            # Let's just use dist_sq to origin <= 16 (radius 4)
            if dist_sq((a,b), (0,0)) <= 16:
                candidates.add((a,b))
                
    candidates = list(candidates)
    
    # Local Search
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < 10: # 10s limit
        improved = False
        # Shuffle candidates for variety
        random.shuffle(candidates)
        
        # Try swapping each point in current set with each candidate not in set
        # This is O(N * M). N=32, M~100. 3200 swaps.
        # Counting edges is O(N^2) = 1024.
        # Total ops ~ 3M. Fast enough.
        
        # Optimization: Precompute adjacency or just re-count
        # Re-counting is safer and fast enough for 10s budget
        
        current_set = set(points)
        
        for i in range(len(points)):
            if time.time() - start_time > 10: break
            p_to_remove = points[i]
            for c in candidates:
                if c in current_set: continue
                
                # Swap p_to_remove with c
                # New set: points without p_to_remove, plus c
                # We can calculate delta score
                # Edges lost: edges between p_to_remove and other points in set
                # Edges gained: edges between c and other points in set (excluding p_to_remove)
                
                edges_lost = 0
                edges_gained = 0
                for j in range(len(points)):
                    if i == j: continue
                    other = points[j]
                    if dist_sq(p_to_remove, other) == 1:
                        edges_lost += 1
                    if dist_sq(c, other) == 1:
                        edges_gained += 1
                        
                # Note: If c is adjacent to p_to_remove, that edge is lost and not gained?
                # No, p_to_remove is removed. c is added.
                # Edge (p_to_remove, other) is lost if dist=1.
                # Edge (c, other) is gained if dist=1.
                # The edge between c and p_to_remove is irrelevant as p_to_remove is gone.
                
                delta = edges_gained - edges_lost
                if delta > 0:
                    # Perform swap
                    points[i] = c
                    current_set.remove(p_to_remove)
                    current_set.add(c)
                    current_score += delta
                    improved = True
                    # Restart loop or continue?
                    # Breaking inner loop to re-shuffle or continue?
                    # Usually, accepting one move and continuing is fine.
                    # But we need to update points[i] which we did.
                    # The 'current_set' is updated.
                    # We should break the candidate loop for this i?
                    # Or just continue?
                    # If we continue, we might swap the same point again?
                    # c is now in current_set, so 'if c in current_set: continue' will skip it.
                    # p_to_remove is no longer in points[i], it's gone.
                    # So it's safe to continue.
                    pass
                elif delta == 0:
                    # Optional: accept neutral moves to escape local minima?
                    # For now, only accept improvements.
                    pass
                    
    return [[a, b] for a, b in points]
