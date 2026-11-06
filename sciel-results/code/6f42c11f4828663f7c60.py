"""Improved program: Local search for max edges on triangular lattice."""
import time
import random

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _edge_count(Sset):
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    if len(Sset) != 32:
        return False
    return all(abs(a) <= 64 and abs(b) <= 64 for a, b in Sset)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Initial guess: Hexagon side 3 (19 pts) + 13 from Layer 3
    # Layer 0: (0,0)
    # Layer 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Layer 2: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1)? No.
    # Correct L2:
    l2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), 
          (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2)] # Wait, (2,1) and (1,2) are L3?
    # Dist(2,1) = 3. Dist(1,2) = 3.
    # L2 points are dist 2.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (1,0)? No.
    # Let's generate L2 properly.
    l2 = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            d = (abs(a) + abs(a+b) + abs(b)) // 2
            if d == 2:
                l2.append((a,b))
                
    l1 = []
    for a in range(-1, 2):
        for b in range(-1, 2):
            d = (abs(a) + abs(a+b) + abs(b)) // 2
            if d == 1:
                l1.append((a,b))
                
    l0 = [(0,0)]
    
    base_set = set(l0 + l1 + l2)
    
    # Generate L3 candidates
    l3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            d = (abs(a) + abs(a+b) + abs(b)) // 2
            if d == 3:
                l3.append((a,b))
                
    # Pick 13 from L3 to maximize edges
    # Greedy: pick one with most neighbors in base_set, then update
    current_set = set(base_set)
    candidates = list(l3)
    
    for _ in range(13):
        best_p = None
        best_score = -1
        for p in candidates:
            score = sum(1 for q in _neighbors(p) if q in current_set)
            if score > best_score:
                best_score = score
                best_p = p
        if best_p:
            current_set.add(best_p)
            candidates.remove(best_p)
            
    best_set = set(current_set)
    best_score = _edge_count(best_set)
    
    # Local Search
    # Candidate pool: L3, L4, and some L2 swaps?
    # Just swap points in current_set with points in a larger pool
    pool = set()
    for a in range(-4, 5):
        for b in range(-4, 5):
            pool.add((a,b))
            
    pool.difference_update(best_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize order of swaps
        swaps = list(zip(list(best_set), list(pool)))
        random.shuffle(swaps)
        for p_out, p_in in swaps:
            if time.time() > deadline:
                break
            new_set = set(best_set)
            new_set.remove(p_out)
            new_set.add(p_in)
            score = _edge_count(new_set)
            if score > best_score:
                best_set = new_set
                best_score = score
                improved = True
                # Update pool
                pool.remove(p_in)
                pool.add(p_out)
                break # Restart shuffle? Or just continue?
                
    return [[a, b] for a, b in best_set]
# EVOLVE-BLOCK-END
