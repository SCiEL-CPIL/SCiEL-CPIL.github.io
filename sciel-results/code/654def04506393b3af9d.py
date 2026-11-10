import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    
    # Precomputation
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
            
    # 4. Generate Q3 masks for K8
    q3_masks_for_k8 = []
    seen_masks = set()
    for A in itertools.combinations(range(8), 4):
        B = tuple(v for v in range(8) if v not in A)
        for perm in itertools.permutations(B):
            mask = 0
            for i in range(4):
                a = A[i]
                m = perm[i]
                for j in range(4):
                    b = B[j]
                    if b != m:
                        e_idx = local_edge_idx[a][b]
                        mask |= (1 << e_idx)
            if mask not in seen_masks:
                seen_masks.add(mask)
                q3_masks_for_k8.append(mask)
                
    # 5. Map global edges to subsets and local indices
    edge_to_subsets = [[] for _ in range(total_edges)]
    local_idx = [[-1]*total_edges for _ in range(num_subsets)]
    local_to_global = [[-1]*28 for _ in range(num_subsets)]
    
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i+1, 8):
                u_glob = s[i]
                v_glob = s[j]
                g_idx = edge_index[u_glob][v_glob]
                l_pos = local_edge_idx[i][j]
                
                edge_to_subsets[g_idx].append(s_idx)
                local_idx[s_idx][g_idx] = l_pos
                local_to_global[s_idx][l_pos] = g_idx

    # 6. Precompute Q3 indices by local edge position
    q3_indices_by_local_edge = [[] for _ in range(28)]
    for q_idx, mask in enumerate(q3_masks_for_k8):
        for l in range(28):
            if (mask >> l) & 1:
                q3_indices_by_local_edge[l].append(q_idx)

    def can_add_edge(G, e_idx, subset_masks):
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_mask = subset_masks[s_idx] | (1 << l_pos)
            for q_idx in q3_indices_by_local_edge[l_pos]:
                if (new_mask & q3_masks_for_k8[q_idx]) == q3_masks_for_k8[q_idx]:
                    return False
        return True

    def find_blocking_q3(G, e_idx, subset_masks):
        # Returns list of global edge indices in G that are part of a blocking Q3 with e_idx
        blocking_edges = []
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_mask = subset_masks[s_idx] | (1 << l_pos)
            for q_idx in q3_indices_by_local_edge[l_pos]:
                q_mask = q3_masks_for_k8[q_idx]
                if (new_mask & q_mask) == q_mask:
                    # Found a blocking Q3
                    for l in range(28):
                        if (q_mask >> l) & 1:
                            g_idx = local_to_global[s_idx][l]
                            if g_idx != -1 and (G >> g_idx) & 1:
                                blocking_edges.append(g_idx)
                    break
        return blocking_edges

    best_G = 0
    best_count = 0
    
    # Multiple restarts
    while time.time() < deadline:
        G = 0
        subset_masks = [0] * num_subsets
        missing_edges = set(range(total_edges))
        
        # Greedy Phase
        edges = list(range(total_edges))
        random.shuffle(edges)
        
        for e_idx in edges:
            if time.time() > deadline:
                break
            if can_add_edge(G, e_idx, subset_masks):
                G |= (1 << e_idx)
                missing_edges.remove(e_idx)
                for s_idx in edge_to_subsets[e_idx]:
                    l_pos = local_idx[s_idx][e_idx]
                    subset_masks[s_idx] |= (1 << l_pos)
        
        # Local Search Phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any missing edge
            # Shuffle missing edges for randomness
            missing_list = list(missing_edges)
            random.shuffle(missing_list)
            
            for e_idx in missing_list:
                if can_add_edge(G, e_idx, subset_masks):
                    G |= (1 << e_idx)
                    missing_edges.remove(e_idx)
                    for s_idx in edge_to_subsets[e_idx]:
                        l_pos = local_idx[s_idx][e_idx]
                        subset_masks[s_idx] |= (1 << l_pos)
                    improved = True
                    break
            
            # 2. If no add, try a swap
            if not improved:
                # Pick a random missing edge
                if missing_edges:
                    e_idx = random.choice(list(missing_edges))
                    blocking_edges = find_blocking_q3(G, e_idx, subset_masks)
                    if blocking_edges:
                        random.shuffle(blocking_edges)
                        for f_idx in blocking_edges:
                            # Check if G - f_idx + e_idx is valid
                            valid = True
                            for s_idx2 in edge_to_subsets[e_idx]:
                                l_pos_e = local_idx[s_idx2][e_idx]
                                new_mask = subset_masks[s_idx2] | (1 << l_pos_e)
                                if local_idx[s_idx2][f_idx] != -1:
                                    new_mask &= ~(1 << local_idx[s_idx2][f_idx])
                                # Check Q3s containing l_pos_e
                                for q_idx in q3_indices_by_local_edge[l_pos_e]:
                                    if (new_mask & q3_masks_for_k8[q_idx]) == q3_masks_for_k8[q_idx]:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            
                            if valid:
                                # Perform swap
                                G = (G & ~(1 << f_idx)) | (1 << e_idx)
                                missing_edges.remove(e_idx)
                                missing_edges.add(f_idx)
                                
                                # Update subset_masks
                                for s_idx2 in edge_to_subsets[e_idx]:
                                    subset_masks[s_idx2] |= (1 << local_idx[s_idx2][e_idx])
                                for s_idx2 in edge_to_subsets[f_idx]:
                                    subset_masks[s_idx2] &= ~(1 << local_idx[s_idx2][f_idx])
                                
                                improved = True
                                break

        count = G.bit_count()
        if count > best_count:
            best_count = count
            best_G = G

    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result
