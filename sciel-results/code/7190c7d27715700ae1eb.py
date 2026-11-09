import time
import sys
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    n = 16
    num_edges_total = n * (n - 1) // 2 # 120
    
    # 1. Edge Indexing
    edge_idx_map = {}
    inv_edge_map = [None] * num_edges_total
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = idx
            inv_edge_map[idx] = (u, v)
            idx += 1
            
    # 2. Precompute Local Q3 Masks (on 0..7)
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    def local_edge_index(u, v):
        if u > v: u, v = v, u
        # Map (u,v) in 0..7 to 0..27
        # Number of edges before u: sum_{i=0}^{u-1} (7-i)
        # Edges with u: v - u - 1
        count = 0
        for i in range(u):
            count += (7 - i)
        count += (v - u - 1)
        return count

    local_q3_masks = set()
    # Only need to generate unique masks. 
    # Iterating all 8! permutations is 40320.
    for p in permutations(range(8)):
        mask = 0
        valid = True
        for u, v in q3_abstract_edges:
            lu = p[u]
            lv = p[v]
            le = local_edge_index(lu, lv)
            mask |= (1 << le)
        # Check if it has exactly 12 bits? Yes, Q3 has 12 edges.
        # But we need to ensure it's a simple graph mapping.
        # Since p is a permutation, it's a bijection.
        if mask.bit_count() == 12:
            local_q3_masks.add(mask)
            
    local_q3_list = list(local_q3_masks)
    
    # 3. Precompute Global Q3 Masks
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))
    
    # Precompute local edge index to (u,v) pair for quick lookup? 
    # Or just precompute the mapping for each subset.
    
    # To speed up, precompute the list of local edge indices for each local mask
    local_q3_edge_indices = []
    for m in local_q3_list:
        edges = []
        for i in range(28):
            if m & (1 << i):
                edges.append(i)
        local_q3_edge_indices.append(edges)
        
    # Iterate subsets
    for subset in all_subsets:
        # Map local edge idx (0..27) to global edge idx (0..119)
        # local edge idx i corresponds to pair (u_loc, v_loc)
        # We need a mapping l2g[local_idx] = global_idx
        l2g = [0] * 28
        
        # Build l2g
        # Iterate local pairs
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_index(u_loc, v_loc)
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
                
        # Generate global masks
        for edge_list in local_q3_edge_indices:
            g_mask = 0
            for le in edge_list:
                ge = l2g[le]
                g_mask |= (1 << ge)
            global_q3_masks.append(g_mask)
            
    num_q3s = len(global_q3_masks)
    
    if time.time() > deadline:
         # Fallback: return a sparse valid graph
         # K_8,8 is bipartite. Q3 is bipartite. 
         # Does K_8,8 contain Q3? 
         # Q3 has 4 vertices in each part. K_8,8 has 8 in each.
         # We can pick 4 from left, 4 from right. 
         # The induced subgraph is K_4,4. 
         # Q3 is a subgraph of K_4,4? 
         # Q3 has 12 edges. K_4,4 has 16 edges. 
         # Yes, Q3 is a subgraph of K_4,4. So K_8,8 contains Q3.
         # We need a Q3-free graph.
         # A random sparse graph is likely Q3-free.
         # Or just return a matching.
         return [[i, i+1] for i in range(0, 16, 2)]

    # 4. Greedy Hitting Set
    # active[i] = True if Q3 i is not yet hit
    active = [True] * num_q3s
    # counts[e] = number of active Q3s containing edge e
    counts = [0] * num_edges_total
    # q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    removed_edges = [False] * num_edges_total
    
    while True:
        if time.time() > deadline:
            break
            
        # Find edge with max count
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if removed_edges[e]:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        # Remove best_edge
        removed_edges[best_edge] = True
        
        # Update active Q3s and counts
        for q3_idx in q3s_containing_edge[best_edge]:
            if active[q3_idx]:
                active[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb

    # 5. Local Search: Add back edges
    # Current state: removed_edges is True for edges in R.
    # Kept edges are those with removed_edges[e] == False.
    # We try to add back edges from R.
    # To check if adding e back is safe:
    # For every Q3 Q containing e, check if E(Q) \ {e} is subset of Kept Edges.
    
    # Precompute for each edge e, the list of Q3s containing e is already available.
    # We need a fast way to check "is E(Q) \ {e} subset of Kept?".
    # We can maintain a bitmask of Kept edges? 
    # Or just check the counts?
    # If we maintain `active` as "Q3s that are currently formed by Kept edges", 
    # then initially (after greedy), `active` is all False (all Q3s are hit).
    # Wait, the greedy algorithm ensures all Q3s are hit by R. 
    # So in the graph G = K16 - R, there are NO Q3s.
    # So `active` (representing Q3s present in G) is all False.
    
    # If we add back edge e, we need to check if any Q3 becomes "active" (present in G).
    # A Q3 Q becomes active if:
    # 1. Q contains e.
    # 2. All other edges in Q are in G (i.e., not in R).
    
    # So, for each e in R:
    #   Check all Q in q3s_containing_edge[e]:
    #     If all other edges in Q are NOT in R (i.e., kept), then adding e creates Q.
    #     If no such Q exists, we can safely add e back (remove from R).
    
    # Optimization: 
    # We can iterate through R in some order.
    # Since adding an edge might allow adding another, we should loop until no changes.
    
    # To speed up the check "all other edges in Q are kept":
    # We can precompute for each Q, the list of its 12 edge indices.
    # Then for a candidate e, and a Q containing e, we check the other 11 edges.
    
    # Let's extract edge lists for Q3s to speed up this check.
    q3_edge_lists = []
    for mask in global_q3_masks:
        edges = []
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            edges.append(e_idx)
            m ^= lsb
        q3_edge_lists.append(edges)
        
    # Local Search Loop
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to add back each removed edge
        for e in range(num_edges_total):
            if not removed_edges[e]:
                continue
            # Check if e can be added back
            safe = True
            for q3_idx in q3s_containing_edge[e]:
                # Check if Q3 q3_idx is "almost" present in G
                # i.e., all edges in Q except e are in G
                edges_in_q = q3_edge_lists[q3_idx]
                # We know e is in edges_in_q
                # Check the others
                for other_e in edges_in_q:
                    if other_e == e:
                        continue
                    if removed_edges[other_e]:
                        # This other edge is removed, so Q is not formed
                        safe = True
                        break
                else:
                    # All other edges are kept, so adding e completes Q
                    safe = False
                    break
            if safe:
                removed_edges[e] = False
                changed = True

    # Construct result
    result_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges
