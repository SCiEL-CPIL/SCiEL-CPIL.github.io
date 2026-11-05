import time
import random
import itertools

N = 16
DEADLINE = 145.0

# Precompute all 3-subsets of vertices
# We need to check if adding (u,v) creates Q3.
# Q3 has bipartition X, Y of size 4.
# If (u,v) is in Q3, u in X, v in Y (or vice versa).
# So we need 3 other vertices in X and 3 in Y.
# X = {u} U U, Y = {v} U W.
# U, W are disjoint 3-subsets of V \ {u, v}.

# Precompute list of all 3-subsets
_SUBSETS3 = list(itertools.combinations(range(N), 3))
# Map from mask to list of vertices? Or just use masks.
# For speed, precompute masks for all 3-subsets
_SUBSETS3_MASKS = []
for s in _SUBSETS3:
    m = 0
    for v in s:
        m |= (1 << v)
    _SUBSETS3_MASKS.append(m)

# To check if adding (u, v) creates Q3:
# We iterate over all pairs of 3-subsets (U, W) from V \ {u, v} such that U and W are disjoint.
# For each such pair, let X = {u} U U, Y = {v} U W.
# We check if the bipartite graph between X and Y in the current graph G (plus edge u-v)
# contains Q3.
# This is equivalent to: The missing edges between X and Y in G (minus u-v if it was missing)
# contain a perfect matching.
# Note: u-v is being added, so it is NOT missing.
# So we check if the missing edges in G between X and Y contain a perfect matching
# that does NOT include (u, v).
# Actually, if (u, v) was missing in G, it is removed from the missing set.
# If (u, v) was present in G, we wouldn't be adding it.
# So we assume (u, v) is NOT in G.
# The missing edges in G between X and Y include (u, v).
# After adding (u, v), the missing edges are M_G - {(u, v)}.
# We need M_G - {(u, v)} to contain a perfect matching.

# To make this fast:
# Precompute for each pair of disjoint 4-sets (X, Y) the bitmask of the 16 edges.
# But we can just check on the fly.
# Number of pairs (U, W) is approx 60,000.
# For each, we check 24 perfect matchings.
# 60,000 * 24 = 1.4 million ops per edge check.
# If we try 1000 edges, that's 1.4 billion ops. Too slow in Python.

# Optimization:
# We can precompute for each 4-set X and 4-set Y (disjoint) the "missing mask" relative to a global state? No, state changes.
# However, we can precompute the list of 24 perfect matchings as bitmasks of the 16 possible edges between X and Y.
# For a specific X, Y, the edges are indexed 0..15.
# Edge (x_i, y_j) has index i*4 + j.
# A perfect matching is a set of 4 edges with distinct rows and cols.
# We can represent a perfect matching as a 16-bit mask.
# There are 24 such masks.
# Let PM_MASKS = [mask1, mask2, ..., mask24].
# For a given graph state, we compute the 16-bit mask of missing edges between X and Y.
# Let MISS_MASK be this 16-bit mask.
# We need to check if there exists a PM_MASK such that (MISS_MASK & PM_MASK) == PM_MASK.
# This is a subset check.
# We can precompute for each possible MISS_MASK (16 bits -> 65536 possibilities) whether it contains a PM.
# Let CONTAINS_PM = [False] * 65536.
# for i in range(65536):
#    for pm in PM_MASKS:
#       if (i & pm) == pm:
#          CONTAINS_PM[i] = True
#          break

# Now, for a specific X, Y, we just need to compute MISS_MASK.
# MISS_MASK has bit (i*4+j) set if edge (X[i], Y[j]) is MISSING in G.
# We can compute this quickly if we have adjacency bitmasks.
# adj[u] is a bitmask of neighbors.
# Edge (u, v) exists if (adj[u] >> v) & 1.

# Let's precompute CONTAINS_PM.

PM_MASKS = []
# Generate all 24 perfect matchings of 4x4
# Represented as 16-bit masks
for perm in itertools.permutations(range(4)):
    mask = 0
    for i, j in enumerate(perm):
        mask |= (1 << (i * 4 + j))
    PM_MASKS.append(mask)

CONTAINS_PM = [False] * 65536
for i in range(65536):
    for pm in PM_MASKS:
        if (i & pm) == pm:
            CONTAINS_PM[i] = True
            break

# Precompute list of (X_mask, Y_mask, U_mask, W_mask) for all disjoint 3-subsets U, W?
# No, we iterate over U, W dynamically for each (u, v).
# But we can precompute for each u, v, the list of valid (U, W) pairs?
# That's 16*15/2 = 120 pairs.
# For each pair, ~60k (U, W) combinations.
# Total precomputation: 120 * 60k = 7.2 million entries.
# Each entry: (U_mask, W_mask, X_order, Y_order)?
# We need to map the 16 edges to the 16-bit mask indices.
# The index depends on the ordering of X and Y.
# We can fix the ordering: sorted order of vertices in X and Y.
# So for a specific X, Y, the edge (x_i, y_j) is at index i*4+j.
# We need to know, for a given pair (u, v) and subsets U, W, what are the sorted X and Y?
# And then compute the missing mask.

# To speed up, we can precompute for each pair (u, v) the list of "contexts".
# A context is (X_mask, Y_mask, mapping_u_to_idx, mapping_v_to_idx, U_mask, W_mask).
# Actually, simpler:
# For each u, v:
#   For each U (3-subset of V\{u,v}):
#     For each W (3-subset of V\{u,v,U}):
#       X = sorted({u} U U)
#       Y = sorted({v} U W)
#       Store (X_mask, Y_mask, X_list, Y_list)
#       But storing lists is slow.
#       We can just store the masks and re-sort? No, sorting is slow.
#       We can precompute the "index map" for each (X_mask, Y_mask).
#       There are only C(16,4) * C(12,4) / 2 disjoint pairs?
#       C(16,4) = 1820.
#       Pairs of disjoint 4-sets: 1820 * 1820 / something?
#       Actually, number of disjoint 4-set pairs is C(16,4) * C(12,4) / 2 ? No.
#       Choose 4 for X: 1820. Choose 4 for Y from remaining 12: 495.
#       Total ordered pairs: 1820 * 495 = 900,900.
#       This is a lot to precompute for each (u,v).
       
# Alternative: On-the-fly check with early termination.
# The number of (U, W) pairs is small enough if we are careful.
# But 60k checks per edge addition is high.

# Let's try a different angle:
# The "Knowledge" says K_3,13 + intra edges works.
# Let's just implement a fast local search on a good seed.
# Seed: K_3,13.
# Add intra-B edges greedily.
# Then local search.

def _init():
    pass

def _has_q3_in_bipartite(adj_masks, X, Y):
    """X, Y are lists of 4 vertices. Check if bipartite subgraph contains Q3."""
    # Compute missing mask
    miss_mask = 0
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            if not (adj_masks[x] >> y) & 1:
                miss_mask |= (1 << (i * 4 + j))
    return CONTAINS_PM[miss_mask]

def _creates_q3_containing_edge(u, v, adj_masks):
    """Check if adding (u, v) creates Q3."""
    # u, v are the new edge.
    # We need to check all X, Y such that u in X, v in Y (or vice versa)
    # and X, Y disjoint, size 4.
    # X = {u} U U, Y = {v} U W.
    # U, W disjoint 3-subsets of V \ {u, v}.
    
    # Optimization: Iterate over U, W.
    # To avoid generating subsets every time, we can precompute
    # for each u, v, the list of (U_mask, W_mask, X_list, Y_list)
    # But memory might be an issue or precomputation time.
    
    # Let's iterate over 3-subsets of the 14 remaining vertices.
    # There are C(14,3) = 364 subsets.
    # For each U, we need W from the remaining 11 vertices. C(11,3) = 165.
    # Total 364 * 165 = 60,060 pairs.
    
    # We need sorted X and Y to use the fixed index mapping.
    # This is expensive.
    
    # Alternative: Don't use fixed index mapping.
    # Just check if the missing edges contain a perfect matching using a fast algorithm.
    # For a 4x4 bipartite graph, checking perfect matching is very fast.
    # We can use a bitmask DP or just check the 24 permutations.
    # Checking 24 permutations is 24 * 4 = 96 bit ops.
    # 60k * 96 = 5.7 million ops.
    # This is feasible in Python if optimized (no function calls in inner loop).
    
    remaining = [x for x in range(N) if x != u and x != v]
    
    # Precompute list of 3-subsets of remaining?
    # We can just use itertools.combinations(remaining, 3)
    # But that creates iterators.
    
    # Let's try to be smart.
    # If we assume the graph is dense, many checks will fail early?
    # No, we need to check if ANY Q3 is formed.
    
    # Let's use the precomputed CONTAINS_PM but we need the miss_mask in the correct order.
    # The order in CONTAINS_PM assumes X and Y are sorted.
    # So we MUST sort X and Y.
    
    # Sorting 4 elements is fast.
    # 60k * (sort 4 + sort 4 + 16 checks)
    # 60k * 20 ops = 1.2 million ops.
    # This is fine.
    
    for U in itertools.combinations(remaining, 3):
        U_set = set(U)
        for W in itertools.combinations(remaining, 3):
            # Check disjoint
            if U_set.intersection(W):
                continue
            X = [u] + list(U)
            Y = [v] + list(W)
            X.sort()
            Y.sort()
            
            # Compute miss_mask
            mm = 0
            for i in range(4):
                x = X[i]
                adj_x = adj_masks[x]
                for j in range(4):
                    y = Y[j]
                    if not (adj_x >> y) & 1:
                        mm |= (1 << (i * 4 + j))
            
            if CONTAINS_PM[mm]:
                return True
    return False

def _is_q3_free(adj_masks):
    """Full check."""
    # Iterate over all disjoint 4-set pairs
    # This is expensive, only use for final verification.
    for X in itertools.combinations(range(N), 4):
        X_set = set(X)
        for Y in itertools.combinations(range(N), 4):
            if X_set.intersection(Y):
                continue
            X_list = list(X)
            Y_list = list(Y)
            X_list.sort()
            Y_list.sort()
            
            mm = 0
            for i in range(4):
                x = X_list[i]
                adj_x = adj_masks[x]
                for j in range(4):
                    y = Y_list[j]
                    if not (adj_x >> y) & 1:
                        mm |= (1 << (i * 4 + j))
            
            if CONTAINS_PM[mm]:
                return False
    return True

def _greedy_build(seed, deadline):
    rng = random.Random(seed)
    adj_masks = [0] * N
    
    # Start with K_3,13
    # Part A: 0, 1, 2. Part B: 3..15
    for u in range(3):
        for v in range(3, 16):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
    # Try to add intra-B edges
    # And intra-A edges (less likely to help)
    # And edges between A and B are already maxed.
    
    candidates = []
    for u in range(3, 16):
        for v in range(u+1, 16):
            candidates.append((u, v))
    for u in range(3):
        for v in range(u+1, 3):
            candidates.append((u, v))
            
    rng.shuffle(candidates)
    
    for u, v in candidates:
        if time.time() > deadline:
            break
        if (adj_masks[u] >> v) & 1:
            continue
            
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
        if _creates_q3_containing_edge(u, v, adj_masks):
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
    # Local Search
    # Try to add any missing edge
    # Try to swap edges
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        # Try adding a random missing edge
        u = rng.randint(0, 15)
        v = rng.randint(0, 15)
        if u == v or (adj_masks[u] >> v) & 1:
            continue
            
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
        if _creates_q3_containing_edge(u, v, adj_masks):
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
        else:
            # Keep it
            pass

    # Verify
    if not _is_q3_free(adj_masks):
        return None
        
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if (adj_masks[u] >> v) & 1:
                result.append([u, v])
    return result

def construct():
    _init()
    
    best_edges = None
    best_count = 0
    
    start_time = time.time()
    seed = 0
    
    # Run multiple seeds
    while time.time() < DEADLINE:
        result = _greedy_build(seed, DEADLINE)
        if result is not None:
            cnt = len(result)
            if cnt > best_count:
                best_count = cnt
                best_edges = result
        
        seed += 1
        
        if seed > 50: # Safety
            break
            
    if best_edges is None:
        # Fallback: K_3,13
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
        
    return best_edges
