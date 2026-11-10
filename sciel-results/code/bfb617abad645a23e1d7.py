import time
import random
import itertools

N = 16
NUM_VERTS = 8

# Precompute local edge map
local_edge_map = {}
idx = 0
for u in range(NUM_VERTS):
    for v in range(u+1, NUM_VERTS):
        local_edge_map[(u,v)] = idx
        idx += 1

# Precompute Q3 masks
q3_local_edges = []
for i in range(NUM_VERTS):
    for j in range(i+1, NUM_VERTS):
        diff = i ^ j
        if diff & (diff - 1) == 0:
            q3_local_edges.append((i, j))

q3_masks = set()
for perm in itertools.permutations(range(NUM_VERTS)):
    mapped_edges = frozenset((perm[u], perm[v]) for u, v in q3_local_edges)
    norm_edges = frozenset((min(u,v), max(u,v)) for u,v in mapped_edges)
    q3_masks.add(norm_edges)

q3_int_masks = []
for edges in q3_masks:
    mask = 0
    for u, v in edges:
        mask |= (1 << local_edge_map[(u,v)])
    q3_int_masks.append(mask)

# Precompute has_q3 table
# 2^28 is 268,435,456
# Use a bytearray for memory efficiency
has_q3 = bytearray(1 << 28)
for m in q3_int_masks:
    has_q3[m] = 1

# Fill superset property
# has_q3[M] = 1 if M is a superset of any Q3 mask
# We can do this by iterating M from 0 to 2^28-1
# But 2^28 loop in Python is slow (~10-20s).
# Alternative: Don't fill the table. Just check the 840 masks directly.
# Wait, 10.8s per scan was the estimate for 840 checks * 12870 subsets.
# If we use the table, we need to fill it first.
# Filling 2^28 entries in Python loop:
# for i in range(1, 1<<28):
#    if has_q3[i] == 0 and has_q3[i & (i-1)] == 1: has_q3[i] = 1
# This is 268M iterations. Too slow in Python.

# So we must stick to direct checking but optimize it.
# Or, use NumPy?
# We can store q3_int_masks in a NumPy array.
# And use vectorized operations?
# local_mask is a scalar.
# (local_mask & q3_array) == q3_array
# This is vectorized over the 840 masks.
# This is fast!

import numpy as np
q3_np = np.array(q3_int_masks, dtype=np.uint32)

def contains_q3_np(local_mask):
    # local_mask is int
    matches = (local_mask & q3_np) == q3_np
    if np.any(matches):
        # Return the first match index
        return np.argmax(matches)
    return -1

# Re-evaluate speed:
# 12870 subsets.
# For each, compute local_mask (Python loop, 28 ops).
# Then call contains_q3_np (NumPy, 840 ops vectorized).
# NumPy overhead is ~10-20us.
# 12870 * 20us = 0.25s per full scan.
# 60 removals * 0.25s = 15s.
# This is well within the 110s limit.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    global_edge_map = {}
    idx = 0
    for u in range(N):
        for v in range(u+1, N):
            global_edge_map[(u,v)] = idx
            idx += 1
            
    current_mask = (1 << 120) - 1
    best_mask = current_mask
    best_edge_count = 120
    
    subsets = list(itertools.combinations(range(N), NUM_VERTS))
    random.shuffle(subsets)
    
    while time.time() < deadline:
        found_q3 = False
        for S in subsets:
            local_mask = 0
            for i in range(NUM_VERTS):
                for j in range(i+1, NUM_VERTS):
                    gu = S[i]
                    gv = S[j]
                    g_idx = global_edge_map[(gu, gv)]
                    l_idx = local_edge_map[(i, j)]
                    if (current_mask >> g_idx) & 1:
                        local_mask |= (1 << l_idx)
            
            match_idx = contains_q3_np(local_mask)
            if match_idx != -1:
                found_q3 = True
                q3_mask = q3_int_masks[match_idx]
                bits = [i for i in range(28) if (q3_mask >> i) & 1]
                l_idx = random.choice(bits)
                lu, lv = -1, -1
                for (u,v), k in local_edge_map.items():
                    if k == l_idx:
                        lu, lv = u, v
                        break
                gu = S[lu]
                gv = S[lv]
                g_idx = global_edge_map[(gu, gv)]
                
                current_mask &= ~(1 << g_idx)
                
                edge_count = bin(current_mask).count('1')
                if edge_count > best_edge_count:
                    best_edge_count = edge_count
                    best_mask = current_mask
                break
        
        if not found_q3:
            break
            
    edges = []
    for (u,v), idx in global_edge_map.items():
        if (best_mask >> idx) & 1:
            edges.append([u, v])
            
    return edges
