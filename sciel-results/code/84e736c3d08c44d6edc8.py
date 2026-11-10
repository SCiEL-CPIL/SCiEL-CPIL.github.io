import time
import random

N = 64
TOTAL_POINTS = N * N

# Precompute coordinates
XS = [i // N for i in range(TOTAL_POINTS)]
YS = [i % N for i in range(TOTAL_POINTS)]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 320
    best_set_indices = []
    best_size = 0
    points_list = list(range(TOTAL_POINTS))
    
    def can_add(p, S, dist_sets):
        if p in dist_sets:
            return False
        
        dists_from_p = []
        # Check distinctness of distances from P
        seen_dists = set()
        for a in S:
            dx = XS[p] - XS[a]
            dy = YS[p] - YS[a]
            d = dx*dx + dy*dy
            
            # Check if this distance was already seen from P (apex P conflict)
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Check if this distance exists in A's distance set (apex A conflict)
            if d in dist_sets[a]:
                return False
            
            dists_from_p.append(d)
        
        return True

    def add_point(p, S, dist_sets):
        # We already computed distances in can_add? No, can_add doesn't return them.
        # To avoid double computation, we can combine or just accept the overhead.
        # Given the speed, double computation is likely fine.
        new_dists = set()
        for a in S:
            dx = XS[p] - XS[a]
            dy = YS[p] - YS[a]
            d = dx*dx + dy*dy
            new_dists.add(d)
            dist_sets[a].add(d)
        
        dist_sets[p] = new_dists
        S.append(p)

    # Greedy construction with random restarts
    while time.time() < deadline:
        random.shuffle(points_list)
        current_set = []
        dist_sets = {}
        
        for p in points_list:
            if can_add(p, current_set, dist_sets):
                add_point(p, current_set, dist_sets)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set_indices = current_set.copy()
            
            # Optional: Local search improvement
            # Try to swap points to improve size?
            # For now, rely on many restarts.

    # Convert indices back to (x, y) tuples
    return [(XS[i], YS[i]) for i in best_set_indices]
# EVOLVE-BLOCK-END
