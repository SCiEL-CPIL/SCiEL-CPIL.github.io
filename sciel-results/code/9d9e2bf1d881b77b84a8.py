import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    total_edges = 120
    
    # 1. Edge indices
    edge_index = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
    
    # 2. Subsets of size 8
    subsets_8 = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets_8)
    
    # 3. Local edge indices for K8
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1
    num_local_edges = 28
            
    # 4. Generate Q3 masks for K8
    # Canonical Q3 edges: (i, j) where i^j is power of 2
    q3_canonical_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            d = i ^ j
            if d & (d-1) == 0:
                q3_canonical_edges.append((i, j))
    
    q3_masks_for_k8 = []
    seen_masks = set()
    
    for perm in itertools.permutations(range(8)):
        mask = 0
        for (u, v) in q3_canonical_edges:
            pu = perm[u]
            pv = perm[v]
            e_idx = local_edge_idx[pu][pv]
            mask |= (1 << e_idx)
        if mask not in seen_masks:
            seen_masks.add(mask)
            q3_masks_for_k8.append(mask)
            
    # 5. Precompute "Rest Masks" for fast Q3 check
    # For each local edge b (0..27), store list of masks (Q \ {b}) for all Q3s Q containing b.
    rest_masks_for_local_edge = [[] for _ in range(num_local_edges)]
    for q_mask in q3_masks_for_k8:
        m = q_mask
        while m:
            lsb = m & -m
            b = lsb.bit_length() - 1
            rest = q_mask ^ lsb
            rest_masks_for_local_edge[b].append(rest)
            m ^= lsb
            
    # 6. Map global edges to subsets and local indices
    edge_to_subsets = [[] for _ in range(total_edges)]
    local_idx = [[-1]*total_edges for _ in range(num_subsets)]
    
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i+1, 8):
                u_glob = s[i]
                v_glob = s[j]
                g_idx = edge_index[u_glob][v_glob]
                l_pos = local_edge_idx[i][j]
                
                edge_to_subsets[g_idx].append(s_idx)
                local_idx[s_idx][g_idx] = l_pos

    # Helper: Check if adding local edge b to subset mask M creates a Q3
    def is_blocked(M, b):
        for rest in rest_masks_for_local_edge[b]:
            if (M & rest) == rest:
                return True
        return False

    # Helper: Check if global edge e_idx can be added to G given subset_masks
    def can_add_edge(e_idx, subset_masks):
        for s_idx in edge_to_subsets[e_idx]:
            b = local_idx[s_idx][e_idx]
            if is_blocked(subset_masks[s_idx], b):
                return False
        return True

    # Greedy Construction
    def greedy_construct(order):
        G = 0
        subset_masks = [0] * num_subsets
        for e_idx in order:
            if can_add_edge(e_idx, subset_masks):
                G |= (1 << e_idx)
                for s_idx in edge_to_subsets[e_idx]:
                    b = local_idx[s_idx][e_idx]
                    subset_masks[s_idx] |= (1 << b)
        return G, subset_masks

    # Check if swap (remove e_rem, add e_add) is valid
    def is_valid_swap(e_rem, e_add, subset_masks):
        # The swap is valid if for ALL Q3s Q containing e_add:
        #   Either G does NOT contain Q \ {e_add}, OR e_rem is in Q \ {e_add}.
        # Note: We assume e_add is NOT currently in G.
        for s_idx in edge_to_subsets[e_add]:
            b_add = local_idx[s_idx][e_add]
            b_rem = local_idx[s_idx][e_rem]
            for rest in rest_masks_for_local_edge[b_add]:
                if (subset_masks[s_idx] & rest) == rest:
                    # G contains Q \ {e_add}, so we need e_rem to be in rest
                    if b_rem == -1 or not ((rest >> b_rem) & 1):
                        return False
        return True

    # Perform an add
    def do_add(e_add, G, subset_masks):
        G |= (1 << e_add)
        for s_idx in edge_to_subsets[e_add]:
            b = local_idx[s_idx][e_add]
            subset_masks[s_idx] |= (1 << b)
        return G

    # Perform a swap
    def do_swap(e_rem, e_add, G, subset_masks):
        # Remove e_rem
        G &= ~(1 << e_rem)
        for s_idx in edge_to_subsets[e_rem]:
            b_rem = local_idx[s_idx][e_rem]
            subset_masks[s_idx] &= ~(1 << b_rem)
        
        # Add e_add
        G |= (1 << e_add)
        for s_idx in edge_to_subsets[e_add]:
            b_add = local_idx[s_idx][e_add]
            subset_masks[s_idx] |= (1 << b_add)
        return G

    best_G = 0
    best_count = 0
    best_subset_masks = [0] * num_subsets
    
    # Phase 1: Multiple Random Greedy Restarts
    # Use different ordering strategies to find diverse good starts
    while time.time() < deadline - 40:
        edges = list(range(total_edges))
        # Random shuffle is the baseline
        random.shuffle(edges)
        G, subset_masks = greedy_construct(edges)
        count = G.bit_count()
        if count > best_count:
            best_count = count
            best_G = G
            best_subset_masks = subset_masks[:]

    # Phase 2: Targeted Local Search
    # We maintain the current state and try to improve it.
    current_G = best_G
    current_subset_masks = best_subset_masks[:]
    current_count = best_count
    
    # Lists for fast access
    present_edges = [i for i in range(total_edges) if (current_G >> i) & 1]
    missing_edges = [i for i in range(total_edges) if not (current_G >> i) & 1]
    
    # Helper to update lists
    def update_lists(G, present_edges, missing_edges):
        present_edges.clear()
        missing_edges.clear()
        for i in range(total_edges):
            if (G >> i) & 1:
                present_edges.append(i)
            else:
                missing_edges.append(i)

    # Local Search Loop
    while time.time() < deadline - 5:
        improved = False
        
        # 1. Try to add any missing edge
        # Shuffle missing edges to avoid bias
        random.shuffle(missing_edges)
        for e_add in missing_edges:
            if can_add_edge(e_add, current_subset_masks):
                current_G = do_add(e_add, current_G, current_subset_masks)
                missing_edges.remove(e_add)
                present_edges.append(e_add)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_G = current_G
                    best_subset_masks = current_subset_masks[:]
                improved = True
                break # Restart adding phase with new state
        
        if improved:
            continue
            
        # 2. If no simple add, try targeted swaps
        # For each missing edge, find the Q3s that block it.
        # Then try to swap out one of the edges in those blocking Q3s.
        
        # To speed up, we can limit the number of missing edges we try per iteration
        # or just iterate all if fast enough.
        # 120 edges * (subset checks) is fast.
        
        random.shuffle(missing_edges)
        swap_found = False
        for e_add in missing_edges:
            if swap_found:
                break
                
            # Identify blocking subsets
            blocking_subsets = []
            for s_idx in edge_to_subsets[e_add]:
                b_add = local_idx[s_idx][e_add]
                # If is_blocked, it means there is at least one rest mask fully contained
                if is_blocked(current_subset_masks[s_idx], b_add):
                    blocking_subsets.append(s_idx)
            
            if not blocking_subsets:
                continue # Should have been caught by can_add_edge, but safety check

            # For each blocking subset, look at the edges in the Q3s that are present
            # We want to remove one edge e_rem from the graph such that
            # for all Q3s Q containing e_add, if Q\{e_add} is in G, then e_rem is in Q\{e_add}.
            
            # Heuristic: Try removing edges that are in the intersection of the blocking Q3s?
            # Or just try random present edges that are in the same subset as e_add?
            
            # Let's gather candidate e_rem edges.
            # An edge e_rem is a candidate if it shares a subset with e_add.
            # Actually, e_rem must be in the specific rest masks that are causing the block.
            
            # Simplified: Just try random swaps involving e_add and a random present edge
            # that is in the same subset.
            
            candidates = []
            for s_idx in blocking_subsets:
                # Get all present edges in this subset
                # This is expensive to do from scratch. 
                # Instead, we can just pick a random present edge from the global list
                # and check if it's in the subset.
                pass
            
            # Better: Just try a few random present edges.
            # If e_rem is not in the subset, is_valid_swap will quickly fail?
            # No, is_valid_swap checks specific rest masks.
            
            # Let's try 10 random present edges
            for _ in range(10):
                if not present_edges:
                    break
                e_rem = random.choice(present_edges)
                if e_rem == e_add: # Should not happen as e_add is missing
                    continue
                if is_valid_swap(e_rem, e_add, current_subset_masks):
                    # Perform swap
                    current_G = do_swap(e_rem, e_add, current_G, current_subset_masks)
                    present_edges.remove(e_rem)
                    missing_edges.append(e_rem)
                    missing_edges.remove(e_add)
                    present_edges.append(e_add)
                    # Count stays same, but structure improved
                    swap_found = True
                    break
        
        if swap_found:
            continue
            
        # 3. If stuck, Perturbation
        # Remove 2-3 random edges and try to re-add
        num_rem = random.randint(2, 3)
        for _ in range(num_rem):
            if not present_edges:
                break
            e_rem = random.choice(present_edges)
            current_G &= ~(1 << e_rem)
            present_edges.remove(e_rem)
            missing_edges.append(e_rem)
            for s_idx in edge_to_subsets[e_rem]:
                b_rem = local_idx[s_idx][e_rem]
                current_subset_masks[s_idx] &= ~(1 << b_rem)
            current_count -= 1
            
        # Try to rebuild greedily from missing edges
        random.shuffle(missing_edges)
        for e_add in missing_edges:
            if can_add_edge(e_add, current_subset_masks):
                current_G = do_add(e_add, current_G, current_subset_masks)
                missing_edges.remove(e_add)
                present_edges.append(e_add)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_G = current_G
                    best_subset_masks = current_subset_masks[:]

    # Final Polish
    if time.time() < deadline - 1:
        current_G = best_G
        current_subset_masks = best_subset_masks[:]
        current_count = best_count
        missing_edges = [i for i in range(total_edges) if not (current_G >> i) & 1]
        
        for e_add in missing_edges:
            if can_add_edge(e_add, current_subset_masks):
                current_G = do_add(e_add, current_G, current_subset_masks)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_G = current_G
                    best_subset_masks = current_subset_masks[:]

    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result
