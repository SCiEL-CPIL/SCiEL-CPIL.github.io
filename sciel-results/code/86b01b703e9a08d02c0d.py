"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_K4 = 1820

# Pre-compute edge indices
EDGE_IDX = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        EDGE_IDX[(i, j)] = idx
        idx += 1

# Pre-compute K4s and their edge indices
K4_EDGES = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    edges = [
        EDGE_IDX[(a, b)], EDGE_IDX[(a, c)], EDGE_IDX[(a, d)],
        EDGE_IDX[(b, c)], EDGE_IDX[(b, d)], EDGE_IDX[(c, d)]
    ]
    K4_EDGES.append(tuple(edges))

# Pre-compute for each edge, which K4s it belongs to
EDGE_TO_K4 = [[] for _ in range(NUM_EDGES)]
for k_idx, edges in enumerate(K4_EDGES):
    for e in edges:
        EDGE_TO_K4[e].append(k_idx)

# Convert K4_EDGES to a numpy array for faster access? 
# Actually, list of tuples is fine if we unpack manually.
# Let's keep K4_EDGES as list of tuples.

def _count_violations_fast(colors, k4_status):
    """
    Update k4_status and return total violations.
    colors: list or np.array of length NUM_EDGES
    k4_status: list of bools, length NUM_K4
    """
    total_v = 0
    for k_idx in range(NUM_K4):
        e0, e1, e2, e3, e4, e5 = K4_EDGES[k_idx]
        c0, c1, c2, c3, c4, c5 = colors[e0], colors[e1], colors[e2], colors[e3], colors[e4], colors[e5]
        # Check if distinct count < 5
        # Fast check: if len(set) < 5
        # Using set is C-optimized and fast for 6 elements
        if len({c0, c1, c2, c3, c4, c5}) < 5:
            if not k4_status[k_idx]:
                k4_status[k_idx] = True
                total_v += 1
        else:
            if k4_status[k_idx]:
                k4_status[k_idx] = False
                total_v -= 1
    return total_v

def _local_search(K, time_limit, seed):
    """
    Run local search for a valid coloring using exactly K colors (0..K-1).
    Returns the coloring if valid, else None.
    """
    random.seed(seed)
    deadline = time.time() + time_limit
    
    # Initialization: Random or Structured
    # Structured: (i+j) % K might not work for small K if K < 17?
    # Let's try random first, then repair.
    # Or try (i+j) % K and see violations.
    
    best_violations = float('inf')
    best_colors = None
    best_k4_status = None
    
    # Try a few initializations
    inits = []
    
    # 1. (i+j) % K
    inits.append([(i + j) % K for i in range(N) for j in range(i + 1, N)])
    
    # 2. Random
    for _ in range(10):
        inits.append([random.randint(0, K - 1) for _ in range(NUM_EDGES)])
        
    # 3. Greedy random: pick random color for each edge, but avoid creating violations if possible? 
    # Too slow for init.
    
    for init in inits:
        if time.time() > deadline - 1:
            break
            
        colors = init
        k4_status = [False] * NUM_K4
        total_v = 0
        
        # Compute initial violations
        for k_idx in range(NUM_K4):
            e0, e1, e2, e3, e4, e5 = K4_EDGES[k_idx]
            c0, c1, c2, c3, c4, c5 = colors[e0], colors[e1], colors[e2], colors[e3], colors[e4], colors[e5]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                k4_status[k_idx] = True
                total_v += 1
                
        if total_v < best_violations:
            best_violations = total_v
            best_colors = colors
            best_k4_status = k4_status
            
        if total_v == 0:
            return colors
        
    # Use the best initialization
    colors = best_colors
    k4_status = best_k4_status
    total_v = best_violations
    
    if total_v == 0:
        return colors
        
    # Local Search
    # We want to minimize total_v.
    # If total_v == 0, we are done.
    # We don't explicitly minimize color count in the search loop because we are constrained to K colors.
    # The "score" is determined by the smallest K for which we find a valid coloring.
    
    while time.time() < deadline - 1:
        # Pick a random violating K4
        # We need a list of violating K4s. Maintaining it dynamically is complex.
        # Simpler: Pick a random edge, change it, see if violations decrease.
        # Or: Pick a random K4. If it's violating, pick an edge in it, change color.
        
        # Let's pick a random edge
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_c = colors[e_idx]
        
        # Try a new color
        new_c = random.randint(0, K - 1)
        if new_c == old_c:
            continue
            
        # Check impact on affected K4s
        affected = EDGE_TO_K4[e_idx]
        new_total_v = total_v
        valid = True
        
        # We need to check if the change improves the situation
        # For each affected K4, check if it was violating and is now valid, or vice versa.
        
        # Store old statuses to revert if needed? 
        # We can just compute delta.
        
        # Optimization: 
        # If a K4 was valid (distinct >= 5), it might become invalid (distinct < 5) if we introduce a color that causes collision?
        # No, adding a color or changing a color can only reduce distinct count if we map to an existing color.
        # If we map to a new color (not in the K4), distinct count increases or stays same.
        # If we map to an existing color, distinct count decreases or stays same.
        
        # So, a valid K4 can become invalid. An invalid K4 can become valid.
        
        for k_idx in affected:
            edges = K4_EDGES[k_idx]
            e0, e1, e2, e3, e4, e5 = edges
            
            # Get colors with new_c substituted
            c0 = colors[e0] if e0 != e_idx else new_c
            c1 = colors[e1] if e1 != e_idx else new_c
            c2 = colors[e2] if e2 != e_idx else new_c
            c3 = colors[e3] if e3 != e_idx else new_c
            c4 = colors[e4] if e4 != e_idx else new_c
            c5 = colors[e5] if e5 != e_idx else new_c
            
            new_distinct = len({c0, c1, c2, c3, c4, c5})
            new_status = new_distinct < 5
            
            if k4_status[k_idx] and not new_status:
                new_total_v -= 1
            elif not k4_status[k_idx] and new_status:
                new_total_v += 1
                # We don't want to increase violations in a pure hill climbing step?
                # Actually, simulated annealing allows it.
                # Let's stick to strict improvement or equal for now, with random restarts.
                # If we allow increase, we need to track temperature.
                # Let's try strict improvement first.
                valid = False
                break
                
        if valid and new_total_v < total_v:
            # Apply move
            colors[e_idx] = new_c
            total_v = new_total_v
            # Update k4_status
            for k_idx in affected:
                edges = K4_EDGES[k_idx]
                e0, e1, e2, e3, e4, e5 = edges
                c0 = colors[e0]
                c1 = colors[e1]
                c2 = colors[e2]
                c3 = colors[e3]
                c4 = colors[e4]
                c5 = colors[e5]
                k4_status[k_idx] = len({c0, c1, c2, c3, c4, c5}) < 5
                
            if total_v == 0:
                return colors
        else:
            # If we are stuck, we might need to accept worse moves or restart.
            # For now, just continue.
            pass
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115 # 5s margin
    
    # Try K from 13 to 17
    # We know 17 works.
    # We want to find the smallest K.
    
    # Allocate time for each K
    # 13: 30s, 14: 30s, 15: 30s, 16: 10s, 17: 10s (fallback)
    
    # Actually, let's try 13, 14, 15 with equal time, then 16, then 17.
    
    # To be safe, if we find a valid 15, we return it.
    # Score for 15 is 17/15 = 1.133
    # Score for 14 is 17/14 = 1.214
    # Score for 13 is 17/13 = 1.308
    
    # We should prioritize finding ANY valid coloring with K < 17.
    # If we find K=13, it's better than K=14.
    # So we should try K=13 first?
    # But K=13 is harder.
    # Let's try K=15 first as it's likely to succeed quickly if at all.
    # Then K=14, then K=13.
    # If none work, return K=17.
    
    # Order: 15, 14, 13, 16, 17
    
    # Time allocation:
    # 15: 40s
    # 14: 30s
    # 13: 30s
    # 16: 5s (just in case 15-13 fail, 16 is easier than 17? No, 16 is easier than 15. But 17 is known.)
    # Actually, if 15 fails, 16 might succeed. 16 is easier than 15.
    # So order: 15, 14, 13, 16, 17.
    
    # Let's refine:
    # Try 15 for 40s.
    # Try 14 for 30s.
    # Try 13 for 30s.
    # Try 16 for 5s.
    # Return 17.
    
    # Wait, if 15 fails, it might be because 15 is impossible or just hard.
    # 16 is easier. So if 15 fails, try 16.
    
    # Let's do:
    # 1. Try 13 (20s)
    # 2. Try 14 (20s)
    # 3. Try 15 (40s)
    # 4. Try 16 (10s)
    # 5. Return 17
    
    # Actually, lower K is better score. So we want the lowest K.
    # If we find 13, we stop.
    
    # Let's try 13, 14, 15 in order.
    
    for K in [13, 14, 15, 16]:
        if time.time() > deadline - 10:
            break
            
        # Allocate remaining time, capped
        time_left = deadline - time.time()
        alloc = min(time_left - 5, 30) # Cap at 30s per K to allow trying others
        if alloc <= 0:
            continue
            
        res = _local_search(K, alloc, seed=42 + K)
        if res is not None:
            return res
            
    # Fallback: 17 colors
    # (i+j) % 17
    return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]

# EVOLVE-BLOCK-END
