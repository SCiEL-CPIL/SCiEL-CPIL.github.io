import numpy as np
from itertools import combinations

def construct():
    # Paley tournament on 67 vertices
    p = 67
    # squares mod 67
    squares = {x * x % p for x in range(1, p)}
    
    # out[i] is a bitmask of vertices j such that i -> j
    out = [0] * p
    for i in range(p):
        mask = 0
        for j in range(p):
            if i == j:
                continue
            if (j - i) % p in squares:
                mask |= (1 << j)
        out[i] = mask
        
    # Generate all 4-sets
    vertices = list(range(p))
    all_4sets = list(combinations(vertices, 4))
    num_sets = len(all_4sets)
    
    # Precompute masks for each 4-set
    masks = [0] * num_sets
    # Also store the vertices of each set for quick "contains v" check?
    # Or just use the combination tuple.
    set_vertices = all_4sets
    
    # Map set index to mask
    for idx, s in enumerate(set_vertices):
        m = 0
        for v in s:
            m |= (1 << v)
        masks[idx] = m
        
    # counts[idx] = number of vertices dominating set idx
    counts = np.zeros(num_sets, dtype=np.int32)
    
    # doms[v] = list of set indices dominated by v
    # sets_with[v] = list of set indices containing v
    doms = [[] for _ in range(p)]
    sets_with = [[] for _ in range(p)]
    
    # Initialize counts and lists
    # This is O(p * num_sets) which is 67 * 750k ~ 50M. Might be slow in Python loop.
    # Optimize:
    # For each v, iterate through all sets?
    # Better: For each set, find dominators?
    # For each set S, a vertex v dominates S if out[v] & mask[S] == mask[S].
    
    # Let's do it set by set to populate doms and sets_with.
    # Actually, sets_with is easy: for each set, for each v in set, append idx to sets_with[v].
    for idx, s in enumerate(set_vertices):
        for v in s:
            sets_with[v].append(idx)
            
    # Now populate counts and doms.
    # For each v, check all sets?
    # Or for each set, check all v?
    # 750k sets * 67 vertices = 50M checks.
    # Each check is bitwise AND and compare.
    # In Python, this loop might take ~10-20 seconds.
    # Let's try to vectorize or optimize.
    
    # Alternative:
    # For each v, generate all 4-subsets of N+(v) and increment counts.
    # N+(v) is given by out[v].
    # We need to extract the set of vertices from the bitmask.
    
    for v in range(p):
        # Get neighbors of v
        neighbors = [j for j in range(p) if (out[v] >> j) & 1]
        # Generate 4-subsets of neighbors
        for s in combinations(neighbors, 4):
            # Find index of this set
            # We need a mapping from tuple to index
            pass
            
    # The mapping from tuple to index is the bottleneck.
    # Let's create a dictionary mapping tuple to index.
    tuple_to_idx = {s: i for i, s in enumerate(set_vertices)}
    
    # Re-run initialization with this map
    counts = np.zeros(num_sets, dtype=np.int32)
    doms = [[] for _ in range(p)]
    
    for v in range(p):
        neighbors = [j for j in range(p) if (out[v] >> j) & 1]
        for s in combinations(neighbors, 4):
            idx = tuple_to_idx[s]
            counts[idx] += 1
            doms[v].append(idx)
            
    # Initialize unique_counts
    unique_counts = np.zeros(p, dtype=np.int32)
    for idx in range(num_sets):
        if counts[idx] == 1:
            # Find the unique dominator
            s = set_vertices[idx]
            m = masks[idx]
            for v in range(p):
                if (out[v] & m) == m:
                    unique_counts[v] += 1
                    break # Found it
            
    active = set(range(p))
    
    # Pruning loop
    while True:
        # Find a free vertex
        free_v = -1
        for v in range(p):
            if v in active and unique_counts[v] == 0:
                free_v = v
                break
                
        if free_v == -1:
            break
            
        v = free_v
        active.remove(v)
        
        # Update counts for sets dominated by v (that are still active)
        for idx in doms[v]:
            s = set_vertices[idx]
            # Check if all vertices in s are active
            if all(u in active for u in s):
                old_count = counts[idx]
                counts[idx] = old_count - 1
                if old_count == 2:
                    # Now count is 1. Find the new unique dominator.
                    m = masks[idx]
                    for u in active:
                        if u != v and (out[u] & m) == m:
                            unique_counts[u] += 1
                            break
                # If old_count was 1, it would be 0, which is invalid.
                # But we only pick free_v if unique_counts[v]==0.
                # Does unique_counts[v]==0 imply v is not the unique dominator of any active set?
                # Yes. unique_counts[v] counts the number of active sets for which v is the UNIQUE dominator.
                # So if unique_counts[v]==0, v is not the unique dominator of any active set.
                # So old_count cannot be 1 for any set in doms[v] that is active.
                # So old_count >= 2.
                
        # Update counts for sets containing v (that were active)
        for idx in sets_with[v]:
            s = set_vertices[idx]
            # The set s contains v. Since v is removed, s is no longer an active 4-set.
            # We need to check if it WAS active before removal.
            # It was active if all other 3 vertices were in active (before removing v).
            # Since v was in active, and we just removed it, we check the other 3.
            other_verts = [u for u in s if u != v]
            if all(u in active for u in other_verts):
                # This set was active.
                old_count = counts[idx]
                if old_count == 1:
                    # Find the unique dominator
                    m = masks[idx]
                    for u in active:
                        if (out[u] & m) == m:
                            unique_counts[u] -= 1
                            break
                # If old_count > 1, no unique dominator, so no change to unique_counts.
                # The set is now inactive, so we don't care about its count anymore.
                # But we should probably mark it or just ignore it in future checks.
                # Since we check "all(u in active for u in s)" in the next steps,
                # and v is not in active, this set will never be considered active again.
                # So we don't need to explicitly invalidate it.

    # Construct the result
    remaining_vertices = sorted(list(active))
    n = len(remaining_vertices)
    
    # Map old indices to new indices
    old_to_new = {old: new for new, old in enumerate(remaining_vertices)}
    
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            old_i = remaining_vertices[i]
            old_j = remaining_vertices[j]
            # Check orientation in original graph
            # 0 means i -> j, 1 means j -> i
            if (out[old_i] >> old_j) & 1:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': n, 'orientation': orientation}
