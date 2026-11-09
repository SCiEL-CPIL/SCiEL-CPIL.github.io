import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial 6x5 Grid
    x_coords = np.arange(6) - 2
    y_coords = np.arange(5) - 2
    points = []
    for x in x_coords:
        for y in y_coords:
            points.append((int(x), int(y)))
            
    pts = np.array(points, dtype=np.int32)
    n = 30
    
    # Precompute indices for triples
    # We need to know which triples involve a specific point i.
    # triples_with[i] = list of (j, k) such that i < j < k or i > j < k etc?
    # It's easier to just store (j, k) pairs where j, k != i.
    # But to avoid double counting in the histogram, we should have a canonical set of triples.
    # Canonical: i < j < k.
    # If we move point i, we affect:
    # 1. Triples (i, j, k) where i < j < k.
    # 2. Triples (j, i, k) where j < i < k.
    # 3. Triples (j, k, i) where j < k < i.
    
    # Let's precompute for each i, the list of (j, k) pairs from the canonical set that include i.
    # And also the "role" of i in the determinant calculation?
    # Det(P_i, P_j, P_k) is symmetric in magnitude.
    # det = | (P_j - P_i) x (P_k - P_i) |
    
    # Precomputation:
    # For each i, store list of (j, k) such that {i, j, k} is a canonical triple.
    # We also need to know which point is the "base" for the vector subtraction to reuse logic?
    # Actually, |det(A,B,C)| = |det(A-C, B-C)|.
    # If we move i, and the triple is (i, j, k):
    #   Old: | (P_j - P_i) x (P_k - P_i) |
    #   New: | (P_j - P_i') x (P_k - P_i') |
    # If the triple is (j, i, k) with j < i < k:
    #   Canonical base is j? No, canonical is just the set.
    #   Let's always compute relative to the smallest index in the triple?
    #   No, that changes when we move indices? No, indices are fixed positions in the list.
    #   Wait, "indices" in the list are fixed slots. The points move.
    #   So triple (0, 1, 2) always refers to slots 0, 1, 2.
    #   If slot 0 moves, we recalculate det(P_0', P_1, P_2).
    
    # So for each slot i, we precompute all pairs (j, k) from the set of all triples T.
    # T = all (a, b, c) with a < b < c.
    # For slot i, affected triples are those containing i.
    # List: affected[i] = [(j, k, base_idx) ...] where base_idx is the index used as reference?
    # Actually, we can just compute det(P_j - P_i, P_k - P_i) if i is one of the points.
    # The formula |det| is invariant to which point is the base.
    # So for any triple containing i, we can compute it as |det(P_j - P_i, P_k - P_i)| using the other two points j, k.
    
    affected = [[] for _ in range(n)]
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                all_triples.append((a, b, c))
                # Add to affected lists
                affected[a].append((b, c))
                affected[b].append((a, c))
                affected[c].append((a, b))
                
    # Histogram of areas
    # Areas are integers (doubled area).
    # Max possible area: points in [-30,30], max det ~ 60*60*2 = 7200?
    # Actually, max width 60, max height 60. Area ~ 1800. Doubled ~ 3600.
    # Let's use a dict or a large array.
    # Dict is safer.
    
    area_counts = {}
    
    def get_area(p1, p2, p3):
        # p1, p2, p3 are tuples (x, y)
        # Using p1 as base
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p3[0] - p1[0]
        dy2 = p3[1] - p1[1]
        return abs(dx1 * dy2 - dy1 * dx2)
        
    # Initialize histogram
    current_max = 0
    for a, b, c in all_triples:
        area = get_area(pts[a], pts[b], pts[c])
        if area > 0:
            area_counts[area] = area_counts.get(area, 0) + 1
            if area_counts[area] > current_max:
                current_max = area_counts[area]
                
    best_score = current_max
    best_pts = [tuple(p) for p in pts]
    
    # Search Loop
    # We will try to move one point at a time.
    # To speed up, we only recalculate the affected triples.
    
    for _ in range(20000): # Iterations
        if time.time() > deadline:
            break
            
        idx = random.randint(0, n-1)
        old_p = best_pts[idx]
        
        # Try a few random moves
        for _ in range(10):
            new_x = old_p[0] + random.randint(-2, 2)
            new_y = old_p[1] + random.randint(-2, 2)
            
            if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                continue
            new_p = (new_x, new_y)
            if new_p == old_p:
                continue
            
            # Check collision
            collision = False
            for j in range(n):
                if j != idx and best_pts[j] == new_p:
                    collision = True
                    break
            if collision:
                continue
            
            # Calculate delta score
            # We need to update area_counts for all triples in affected[idx]
            
            # Store old areas to revert if we reject
            old_areas = []
            new_areas = []
            
            valid_move = True
            for j, k in affected[idx]:
                # The triple is (idx, j, k) in some order.
                # We compute area using idx as base.
                # Old area:
                old_area = get_area(old_p, best_pts[j], best_pts[k])
                # New area:
                new_area = get_area(new_p, best_pts[j], best_pts[k])
                
                old_areas.append(old_area)
                new_areas.append(new_area)
                
                if old_area > 0:
                    area_counts[old_area] -= 1
                    if area_counts[old_area] == 0:
                        # Potential max drop?
                        pass
                        
                if new_area > 0:
                    area_counts[new_area] = area_counts.get(new_area, 0) + 1
                    
            # Determine new max
            # This is the expensive part if we scan every time.
            # But we can keep a count of how many areas have the current max?
            # Or just scan the dict. Dict size is small.
            new_max = max(area_counts.values()) if area_counts else 0
            
            if new_max > best_score:
                best_score = new_max
                best_pts[idx] = new_p
                # Update pts array for next iterations?
                # We are using best_pts list.
                break # Accept and restart
            else:
                # Revert changes
                for old_area, new_area in zip(old_areas, new_areas):
                    if old_area > 0:
                        area_counts[old_area] += 1
                    if new_area > 0:
                        area_counts[new_area] -= 1
                        if area_counts[new_area] == 0:
                            del area_counts[new_area]
                            
    return best_pts
