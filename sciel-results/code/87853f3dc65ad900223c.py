import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    
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
    total_edges = 120
    
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
    # If current subset mask M contains (Q \ {b}), then adding b creates Q.
    rest_masks_for_local_edge = [[] for _ in range(num_local_edges)]
    for q_mask in q3_masks_for_k8:
        # Iterate through bits in q_mask
        m = q_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            b = lsb.bit_length() - 1
            rest = q_mask ^ lsb
            rest_masks_for_local_edge[b].append(rest)
            m ^= lsb
            
    # 6. Map global edges to subsets and local indices
    edge_to_subsets = [[] for _ in range(total_edges)]
    # local_idx[s_idx][g_idx] = local bit position
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
    def can_add_edge(G, e_idx, subset_masks):
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
            if can_add_edge(G, e_idx, subset_masks):
                G |= (1 << e_idx)
                for s_idx in edge_to_subsets[e_idx]:
                    b = local_idx[s_idx][e_idx]
                    subset_masks[s_idx] |= (1 << b)
        return G, subset_masks

    best_G = 0
    best_count = 0
    best_subset_masks = [0] * num_subsets
    
    # Phase 1: Multiple Random Greedy Restarts
    # We do many fast greedy runs to find a good starting point
    while time.time() < deadline - 30:
        edges = list(range(total_edges))
        random.shuffle(edges)
        G, subset_masks = greedy_construct(edges)
        count = G.bit_count()
        if count > best_count:
            best_count = count
            best_G = G
            best_subset_masks = subset_masks[:]

    # Phase 2: Local Search (Simulated Annealing / Hill Climbing)
    # We use the fast is_blocked check to allow many iterations
    current_G = best_G
    current_subset_masks = best_subset_masks[:]
    current_count = best_count
    
    # Maintain list of missing edges for fast random selection
    missing_edges = [i for i in range(total_edges) if not (current_G >> i) & 1]
    present_edges = [i for i in range(total_edges) if (current_G >> i) & 1]
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    while T > min_T and time.time() < deadline - 5:
        improved = False
        # Try to add a random missing edge
        if missing_edges:
            e_add = random.choice(missing_edges)
            if can_add_edge(current_G, e_add, current_subset_masks):
                # Accept
                current_G |= (1 << e_add)
                missing_edges.remove(e_add)
                present_edges.append(e_add)
                for s_idx in edge_to_subsets[e_add]:
                    b = local_idx[s_idx][e_add]
                    current_subset_masks[s_idx] |= (1 << b)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_G = current_G
                    best_subset_masks = current_subset_masks[:]
                improved = True
        
        # If no simple add, try a swap (remove one, add one)
        if not improved and present_edges and missing_edges:
            e_rem = random.choice(present_edges)
            # Tentatively remove e_rem
            # We need to check if we can add some missing edge after removing e_rem
            # To save time, we just try to add a random missing edge in the state G - e_rem
            # We don't actually modify state yet, we just check validity of G - e_rem + e_add
            
            # Optimization: Instead of full check, we can just try to add e_add to G - e_rem
            # This requires checking subsets of e_add.
            # The state for subsets is current_subset_masks.
            # If we remove e_rem, the subset masks change for subsets containing e_rem.
            # This is complex to do "virtually".
            
            # Simpler Swap Strategy:
            # 1. Pick e_rem from present.
            # 2. Pick e_add from missing.
            # 3. Check if G - e_rem + e_add is valid.
            #    This is equivalent to: Is e_add valid in G - e_rem?
            #    The only Q3s that could be formed by adding e_add are those containing e_add.
            #    A Q3 containing e_add is valid in G - e_rem + e_add if:
            #    - It doesn't exist in G (since G is Q3-free).
            #    - It is not "blocked" by edges in G.
            #    - Wait, if it exists in G, it's not a NEW Q3.
            #    - The condition for G - e_rem + e_add to be Q3-free is that no Q3 exists in it.
            #    - Any Q3 in G - e_rem + e_add must contain e_add (since G - e_rem is Q3-free).
            #    - So we check: For every Q3 Q containing e_add, is Q a subset of G - e_rem + e_add?
            #    - This is true iff (Q \ {e_add}) is a subset of G - e_rem.
            #    - This is true iff (Q \ {e_add, e_rem}) is a subset of G - e_rem AND e_rem is NOT in Q \ {e_add}?
            #    - Actually: Q \ {e_add} subset of G - e_rem  <=>  (Q \ {e_add}) \ {e_rem} subset of G AND e_rem not in Q \ {e_add} OR e_rem in Q \ {e_add} is removed.
            #    - Let R = Q \ {e_add}.
            #    - We need R \ {e_rem} subset of G.
            #    - Since G is Q3-free, if e_rem is NOT in Q, then R is a subset of G? No, R is 11 edges. G might not contain all of them.
            #    - Wait, G is Q3-free, so G does NOT contain any Q.
            #    - So G does NOT contain Q.
            #    - If e_rem is not in Q, then Q is not a subset of G. So G is missing at least one edge from Q.
            #    - If we add e_add, we complete Q only if G contained all other edges in Q.
            #    - But G is Q3-free, so G never contained all edges of any Q.
            #    - Therefore, adding e_add to G never creates a Q3?
            #    - NO. G is Q3-free. Adding e_add creates a Q3 if G contains all other edges of that Q3.
            #    - So if G contains Q \ {e_add}, then adding e_add creates Q.
            #    - If we remove e_rem first:
            #       If e_rem is in Q \ {e_add}, then G - e_rem does NOT contain Q \ {e_add} (because it's missing e_rem).
            #       So adding e_add to G - e_rem will NOT create Q.
            #       If e_rem is NOT in Q, then G - e_rem still contains Q \ {e_add} (assuming G did).
            #       So if G contains Q \ {e_add} and e_rem is not in Q, then G - e_rem + e_add WILL create Q.
            
            # So, the swap is valid if for ALL Q3s Q containing e_add:
            #   Either G does NOT contain Q \ {e_add}, OR e_rem is in Q \ {e_add}.
            
            # Let's implement this check.
            valid_swap = True
            for s_idx in edge_to_subsets[e_add]:
                b_add = local_idx[s_idx][e_add]
                # Check all Q3s containing b_add in this subset
                for rest in rest_masks_for_local_edge[b_add]:
                    # rest represents Q \ {e_add} in local indices
                    # We need to check if G contains all edges in rest.
                    # If it does, we must ensure e_rem is in rest.
                    # How to check if G contains rest?
                    # We have current_subset_masks[s_idx] which is the mask of edges in G for this subset.
                    # If (current_subset_masks[s_idx] & rest) == rest, then G contains Q \ {e_add}.
                    # In that case, we MUST have e_rem in rest.
                    # e_rem is a global edge. We need its local index in this subset.
                    b_rem = local_idx[s_idx][e_rem]
                    if b_rem == -1:
                        # e_rem is not in this subset, so it cannot be in rest (which is local to this subset)
                        # So if G contains rest, the swap is invalid.
                        if (current_subset_masks[s_idx] & rest) == rest:
                            valid_swap = False
                            break
                    else:
                        # e_rem is in this subset. Check if it is in rest.
                        # rest is a bitmask of local edges.
                        if (rest >> b_rem) & 1:
                            # e_rem is in the Q3, so removing it breaks the Q3. Safe.
                            pass
                        else:
                            # e_rem is NOT in the Q3.
                            # So if G contains rest, adding e_add will create Q3.
                            if (current_subset_masks[s_idx] & rest) == rest:
                                valid_swap = False
                                break
                if not valid_swap:
                    break
            
            if valid_swap:
                # Perform swap
                # Remove e_rem
                current_G &= ~(1 << e_rem)
                present_edges.remove(e_rem)
                missing_edges.append(e_rem)
                for s_idx in edge_to_subsets[e_rem]:
                    b_rem = local_idx[s_idx][e_rem]
                    current_subset_masks[s_idx] &= ~(1 << b_rem)
                
                # Add e_add
                current_G |= (1 << e_add)
                missing_edges.remove(e_add)
                present_edges.append(e_add)
                for s_idx in edge_to_subsets[e_add]:
                    b_add = local_idx[s_idx][e_add]
                    current_subset_masks[s_idx] |= (1 << b_add)
                
                # Count stays same
                if current_count > best_count:
                    # No change in count, but maybe better structure for future?
                    # We keep best_count as max edges.
                    pass
                improved = True

        T *= cooling

    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result
