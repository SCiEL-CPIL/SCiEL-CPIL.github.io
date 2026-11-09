"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
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

def _count_distinct_6(c0, c1, c2, c3, c4, c5):
    return len({c0, c1, c2, c3, c4, c5})

def _initial_violations(colors):
    """Compute initial violation status and count."""
    k4_status = [False] * NUM_K4
    total_v = 0
    for k_idx, edges in enumerate(K4_EDGES):
        c0, c1, c2, c3, c4, c5 = (colors[e] for e in edges)
        if _count_distinct_6(c0, c1, c2, c3, c4, c5) < 5:
            k4_status[k_idx] = True
            total_v += 1
    return k4_status, total_v

def _count_colors(colors):
    return len(set(colors))

def _construct():
    deadline = time.time() + 115
    random.seed(42)
    
    # Start with (i+j) % 17
    best = [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    # Check if valid
    k4_status, total_v = _initial_violations(best)
    
    if total_v > 0:
        # Try to find a valid starting point
        # Try (i+j) % k for k in 13..20
        found_valid = False
        for k in range(13, 21):
            if time.time() > deadline - 10: break
            cand = [(i + j) % k for i in range(N) for j in range(i + 1, N)]
            ks, tv = _initial_violations(cand)
            if tv == 0:
                best = cand
                k4_status = ks
                total_v = 0
                found_valid = True
                break
        if not found_valid:
            # Random search for valid
            for _ in range(5000):
                if time.time() > deadline - 5: break
                cand = [random.randint(0, 19) for _ in range(NUM_EDGES)]
                ks, tv = _initial_violations(cand)
                if tv == 0:
                    best = cand
                    k4_status = ks
                    total_v = 0
                    found_valid = True
                    break
                elif tv < total_v:
                    best = cand
                    k4_status = ks
                    total_v = tv
            if not found_valid:
                # If still invalid, return best effort
                return best

    # Local Search with Incremental Updates
    current = list(best)
    current_violations = total_v
    current_k4_status = k4_status
    
    # Track color frequencies to know num_colors
    color_freq = {}
    for c in current:
        color_freq[c] = color_freq.get(c, 0) + 1
    current_colors = len(color_freq)
    
    # Objective: Minimize colors if violations == 0, else minimize violations.
    
    for step in range(100000):
        if time.time() > deadline - 2:
            break
            
        if current_violations == 0:
            # Try to reduce colors
            # Pick a random edge
            e_idx = random.randint(0, NUM_EDGES - 1)
            old_c = current[e_idx]
            
            # Try changing to a random color from 0..12
            # If we pick a color that is already present, color count doesn't decrease.
            # To decrease, we must pick a color that makes freq[old_c] == 0.
            # So we should pick a new color that is NOT old_c, and check if it reduces count.
            # Actually, if we change old_c to new_c:
            # freq[old_c] -= 1
            # freq[new_c] += 1
            # Color count decreases if freq[old_c] becomes 0 AND freq[new_c] was > 0 (so it doesn't increase)
            # Color count stays same if (freq[old_c] > 1) or (freq[new_c] == 0)
            # Color count increases if freq[old_c] > 1 and freq[new_c] == 0
            
            # Strategy: Pick a random color new_c.
            new_c = random.randint(0, 15)
            if new_c == old_c:
                continue
                
            # Calculate impact on violations
            new_violations = current_violations
            
            # Update affected K4s
            affected_k4s = EDGE_TO_K4[e_idx]
            # We need to revert if it fails
            # Store old statuses
            old_statuses = []
            for k_idx in affected_k4s:
                old_statuses.append(current_k4_status[k_idx])
                
            # Apply change temporarily to 'current' for calculation?
            # No, we can just calculate distinct count using new_c for e_idx
            # But we need the other 5 colors.
            
            valid_move = True
            for k_idx in affected_k4s:
                edges = K4_EDGES[k_idx]
                # Get colors
                # One edge is e_idx, others are fixed
                # We can just iterate edges
                c0, c1, c2, c3, c4, c5 = 0,0,0,0,0,0
                # Unpack for speed
                e0, e1, e2, e3, e4, e5 = edges
                c0 = current[e0] if e0 != e_idx else new_c
                c1 = current[e1] if e1 != e_idx else new_c
                c2 = current[e2] if e2 != e_idx else new_c
                c3 = current[e3] if e3 != e_idx else new_c
                c4 = current[e4] if e4 != e_idx else new_c
                c5 = current[e5] if e5 != e_idx else new_c
                
                new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
                new_status = new_distinct < 5
                
                if current_k4_status[k_idx] and not new_status:
                    new_violations -= 1
                elif not current_k4_status[k_idx] and new_status:
                    new_violations += 1
                    valid_move = False # We don't want to introduce violations if we had 0
                    break
                # Update status for subsequent checks? No, we are checking against original state mostly.
                # But if multiple K4s share edges, we are only changing one edge, so each K4 is independent in terms of "was it violation before".
                
            if not valid_move:
                continue
                
            # Check color count change
            new_color_count = current_colors
            if color_freq[old_c] == 1:
                new_color_count -= 1
            if new_c not in color_freq:
                new_color_count += 1
                
            # Accept if violations stay 0 and colors decrease
            if new_violations == 0 and new_color_count < current_colors:
                # Apply move
                current[e_idx] = new_c
                color_freq[old_c] -= 1
                if color_freq[old_c] == 0:
                    del color_freq[old_c]
                color_freq[new_c] = color_freq.get(new_c, 0) + 1
                current_colors = new_color_count
                current_violations = new_violations
                # Update K4 statuses
                for i, k_idx in enumerate(affected_k4s):
                    # Re-calc status or use the logic from above
                    # We broke out of loop if invalid, so if we are here, all were non-violating or became non-violating
                    # Actually, if we didn't break, we need to update statuses
                    # We can just set them to False if we know they are valid?
                    # No, some might have been violations and became valid.
                    # Let's just re-calc or store the new_status from the loop.
                    # We didn't store new_status in the list.
                    # Let's re-calc for the 91 K4s. It's cheap.
                    pass # Optimization: we can store new_statuses in the loop
                
                # Let's refine the loop to store new statuses
                # Actually, simpler: just re-run _initial_violations on the whole graph? No, too slow.
                # Let's update statuses properly.
                
                # Re-do the loop to update statuses
                for k_idx in affected_k4s:
                    edges = K4_EDGES[k_idx]
                    e0, e1, e2, e3, e4, e5 = edges
                    c0 = current[e0]
                    c1 = current[e1]
                    c2 = current[e2]
                    c3 = current[e3]
                    c4 = current[e4]
                    c5 = current[e5]
                    current_k4_status[k_idx] = _count_distinct_6(c0, c1, c2, c3, c4, c5) < 5
                    
            else:
                # Revert (do nothing, we didn't modify current)
                pass
        else:
            # Try to reduce violations
            # Pick a random edge
            e_idx = random.randint(0, NUM_EDGES - 1)
            old_c = current[e_idx]
            new_c = random.randint(0, 19)
            if new_c == old_c:
                continue
                
            new_violations = current_violations
            valid_move = True
            
            for k_idx in EDGE_TO_K4[e_idx]:
                edges = K4_EDGES[k_idx]
                e0, e1, e2, e3, e4, e5 = edges
                c0 = current[e0] if e0 != e_idx else new_c
                c1 = current[e1] if e1 != e_idx else new_c
                c2 = current[e2] if e2 != e_idx else new_c
                c3 = current[e3] if e3 != e_idx else new_c
                c4 = current[e4] if e4 != e_idx else new_c
                c5 = current[e5] if e5 != e_idx else new_c
                
                new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
                new_status = new_distinct < 5
                
                if current_k4_status[k_idx] and not new_status:
                    new_violations -= 1
                elif not current_k4_status[k_idx] and new_status:
                    new_violations += 1
                    valid_move = False
                    break
                    
            if valid_move and new_violations < current_violations:
                current[e_idx] = new_c
                color_freq[old_c] -= 1
                if color_freq[old_c] == 0:
                    del color_freq[old_c]
                color_freq[new_c] = color_freq.get(new_c, 0) + 1
                # Update color count
                # Logic for color count change is same
                # But we don't care about color count if violations > 0, except for tie breaking
                # Let's update current_violations
                current_violations = new_violations
                
                # Update statuses
                for k_idx in EDGE_TO_K4[e_idx]:
                    edges = K4_EDGES[k_idx]
                    e0, e1, e2, e3, e4, e5 = edges
                    c0 = current[e0]
                    c1 = current[e1]
                    c2 = current[e2]
                    c3 = current[e3]
                    c4 = current[e4]
                    c5 = current[e5]
                    current_k4_status[k_idx] = _count_distinct_6(c0, c1, c2, c3, c4, c5) < 5

    return current

def construct():
    return _construct()
# EVOLVE-BLOCK-END
