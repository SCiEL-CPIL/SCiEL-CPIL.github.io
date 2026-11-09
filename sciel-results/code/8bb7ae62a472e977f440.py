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
        count = 0
        for i in range(u):
            count += (7 - i)
        count += (v - u - 1)
        return count

    local_q3_masks = set()
    for p in permutations(range(8)):
        mask = 0
        valid = True
        for u, v in q3_abstract_edges:
            lu = p[u]
            lv = p[v]
            le = local_edge_index(lu, lv)
            mask |= (1 << le)
        if mask.bit_count() == 12:
            local_q3_masks.add(mask)
            
    local_q3_list = list(local_q3_masks)
    
    # 3. Precompute Global Q3 Masks
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))
    
    local_q3_edge_indices = []
    for m in local_q3_list:
        edges = []
        for i in range(28):
            if m & (1 << i):
                edges.append(i)
        local_q3_edge_indices.append(edges)
        
    for subset in all_subsets:
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_index(u_loc, v_loc)
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
                
        for edge_list in local_q3_edge_indices:
            g_mask = 0
            for le in edge_list:
                ge = l2g[le]
                g_mask |= (1 << ge)
            global_q3_masks.append(g_mask)
            
    num_q3s = len(global_q3_masks)
    
    if time.time() > deadline:
        return [[i, i+1] for i in range(0, 16, 2)]

    # 4. Data Structures for Search
    # q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    # q3_edge_lists[i] = list of 12 edge indices in Q3 i
    q3_edge_lists = []
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        edges_in_q = []
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            edges_in_q.append(e_idx)
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
        q3_edge_lists.append(edges_in_q)
        
    # Initialize: Greedy Hitting Set to get a valid start
    # active[i] = True if Q3 i is not yet hit
    active = [True] * num_q3s
    counts = [0] * num_edges_total
    removed_edges = [False] * num_edges_total # True if removed from K16
    
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    while True:
        if time.time() > deadline:
            break
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
        removed_edges[best_edge] = True
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

    # Helper: Check if edge e can be added to current graph (defined by removed_edges)
    # A Q3 is formed if all its edges are NOT removed.
    # So, adding e is safe if for all Q3s containing e, at least one other edge in Q3 IS removed.
    def can_add(e):
        for q3_idx in q3s_containing_edge[e]:
            # Check if all other edges in this Q3 are present (not removed)
            # If yes, adding e completes the Q3 -> Invalid
            # If no (some other edge is removed), it's safe
            is_complete = True
            for other_e in q3_edge_lists[q3_idx]:
                if other_e == e:
                    continue
                if removed_edges[other_e]:
                    is_complete = False
                    break
            if is_complete:
                return False
        return True

    # Optimization Loop
    # We try to add edges. If we can't, we try swaps.
    # To avoid O(N^2) swap checks being too slow, we limit iterations or use randomization.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Add edges
        for e in range(num_edges_total):
            if removed_edges[e]:
                if can_add(e):
                    removed_edges[e] = False
                    improved = True
        
        if time.time() > deadline:
            break
            
        # Phase 2: Swaps (1-for-1)
        # If we couldn't add any edge, try to swap.
        # This phase is more expensive. We might want to limit it or run it less frequently.
        # Let's try to find a swap that allows a future addition or just maintains density while changing structure.
        # Actually, if we are in a local optimum for "add", a swap might open up new "add" possibilities.
        
        # For each removed edge v, if it cannot be added, find why.
        # The "blockers" are Q3s where all other edges are present.
        # If we remove one of those other edges (u), v might become addable.
        
        # To keep it fast, we'll iterate through removed edges and try a few swaps.
        # We'll use a simple heuristic: try swapping v with u if u is in a blocking Q3 for v.
        
        # Precompute blocking Q3s for each removed edge? That's dynamic.
        # Let's just try: for each v (removed), if not can_add(v), pick a random blocking Q3, pick a random u from it, try swap.
        
        # To be deterministic and avoid randomness overhead, we'll do a single pass of potential swaps.
        # We'll collect candidates.
        
        # Note: A swap removes u and adds v. Net change 0.
        # It's only useful if it leads to more adds later.
        # Given time constraints, we'll perform a limited number of swap attempts.
        
        swap_attempts = 0
        max_swap_attempts = 50 # Limit per pass to avoid slowdown
        
        for v in range(num_edges_total):
            if not removed_edges[v]:
                continue
            if can_add(v):
                # Should have been added in Phase 1, but maybe state changed?
                # In our loop, we don't update 'improved' inside the swap logic immediately to allow multiple swaps?
                # No, let's stick to Phase 1 for adds.
                pass
            
            # Check if v is blocked
            blocked = False
            blocking_q3 = -1
            for q3_idx in q3s_containing_edge[v]:
                is_complete = True
                for other_e in q3_edge_lists[q3_idx]:
                    if other_e == v:
                        continue
                    if removed_edges[other_e]:
                        is_complete = False
                        break
                if is_complete:
                    blocked = True
                    blocking_q3 = q3_idx
                    break
            
            if blocked and swap_attempts < max_swap_attempts:
                # Try to remove one edge u from blocking_q3 to unblock v
                # We want to remove an edge u that is "cheap" to remove or whose removal doesn't break other things?
                # Since we are maintaining validity, removing u is always safe (makes graph sparser).
                # The risk is that we lose an edge. But if we add v, we are neutral.
                # The benefit is structural change.
                
                # Pick the first edge u in blocking_q3 (excluding v) that is present (not removed)
                # Actually, all edges in blocking_q3 except v are present by definition of "blocking".
                # So we can pick any u in blocking_q3 \ {v}.
                
                # To be strategic, maybe pick u that is in the most Q3s? No, that would be bad to remove.
                # Pick u that is in the FEWEST Q3s? That might be safer.
                # Or just pick the first one.
                
                candidate_u = -1
                for u in q3_edge_lists[blocking_q3]:
                    if u != v:
                        candidate_u = u
                        break
                
                if candidate_u != -1:
                    # Perform swap: remove candidate_u, add v
                    # Check if adding v is safe *after* removing candidate_u.
                    # Since candidate_u is in the blocking Q3, removing it breaks that Q3.
                    # Are there other Q3s containing v that are still complete?
                    # If yes, the swap is invalid (we'd create a different Q3).
                    
                    safe_swap = True
                    for q3_idx_check in q3s_containing_edge[v]:
                        # Check if this Q3 is complete after removing candidate_u
                        # It is complete if all edges in Q3 except v are present.
                        # If candidate_u is in this Q3, it is now removed, so Q3 is broken.
                        # If candidate_u is NOT in this Q3, we need to check if all other edges are present.
                        
                        is_complete = True
                        for other_e in q3_edge_lists[q3_idx_check]:
                            if other_e == v:
                                continue
                            if other_e == candidate_u:
                                # This edge is being removed, so Q3 is not complete
                                is_complete = False
                                break
                            if removed_edges[other_e]:
                                is_complete = False
                                break
                        if is_complete:
                            safe_swap = False
                            break
                    
                    if safe_swap:
                        removed_edges[candidate_u] = True
                        removed_edges[v] = False
                        improved = True
                        swap_attempts += 1
                        # Restart the add phase? Or just continue.
                        # Continuing might allow more swaps.
                        # But we should probably break to re-evaluate adds.
                        break # Break out of v loop to restart add phase?
                        # Actually, let's just break the inner loop and let the outer while handle it.
                        # We set improved=True, so it will loop again.
                        # Breaking the for v loop ensures we don't do multiple swaps in one "pass" which might be complex.
                        # Let's break the for loop.
                        break

    # Construct result
    result_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges
