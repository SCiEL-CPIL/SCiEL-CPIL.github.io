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
    
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i+1, 8):
                u_glob = s[i]
                v_glob = s[j]
                g_idx = edge_index[u_glob][v_glob]
                l_pos = local_edge_idx[i][j]
                
                edge_to_subsets[g_idx].append(s_idx)
                local_idx[s_idx][g_idx] = l_pos

    def contains_q3(mask):
        if mask.bit_count() < 12:
            return False
        for q in q3_masks_for_k8:
            if (mask & q) == q:
                return True
        return False

    def can_add_edge(G, e_idx, subset_masks):
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_mask = subset_masks[s_idx] | (1 << l_pos)
            if contains_q3(new_mask):
                return False
        return True

    def find_blocking_q3(G, e_idx, subset_masks):
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_mask = subset_masks[s_idx] | (1 << l_pos)
            if contains_q3(new_mask):
                return s_idx
        return None

    best_G = 0
    best_count = 0
    
    # Strategy: Structured Initialization (Complete 4-partite)
    # Parts: 0-3, 4-7, 8-11, 12-15
    # Initial G: All edges between different parts.
    # This graph is Q3-free because Q3 is not a complete multipartite graph.
    # We then add intra-part edges via local search.
    
    # We will run multiple restarts, but the first one is the structured one.
    # Subsequent ones can be random greedy or perturbed structured.
    
    num_restarts = 500 # We will break based on time
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize G
        if r == 0:
            # Structured: Complete 4-partite
            G = 0
            subset_masks = [0] * num_subsets
            missing_edges = set(range(total_edges))
            
            # Add all inter-part edges
            for u in range(n):
                for v in range(u+1, n):
                    part_u = u // 4
                    part_v = v // 4
                    if part_u != part_v:
                        e_idx = edge_index[u][v]
                        G |= (1 << e_idx)
                        missing_edges.remove(e_idx)
                        for s_idx in edge_to_subsets[e_idx]:
                            l_pos = local_idx[s_idx][e_idx]
                            subset_masks[s_idx] |= (1 << l_pos)
        else:
            # Random Greedy Initialization for diversity
            G = 0
            subset_masks = [0] * num_subsets
            missing_edges = set(range(total_edges))
            edges = list(range(total_edges))
            random.shuffle(edges)
            for e_idx in edges:
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
                if missing_edges:
                    e_idx = random.choice(list(missing_edges))
                    s_idx_blocker = find_blocking_q3(G, e_idx, subset_masks)
                    if s_idx_blocker is not None:
                        s_vertices = subsets_8[s_idx_blocker]
                        candidate_removals = []
                        for i in range(8):
                            for j in range(i+1, 8):
                                u_glob = s_vertices[i]
                                v_glob = s_vertices[j]
                                g_idx = edge_index[u_glob][v_glob]
                                if (G >> g_idx) & 1:
                                    candidate_removals.append(g_idx)
                        
                        random.shuffle(candidate_removals)
                        
                        for f_idx in candidate_removals:
                            valid = True
                            for s_idx2 in edge_to_subsets[e_idx]:
                                new_mask = subset_masks[s_idx2] | (1 << local_idx[s_idx2][e_idx])
                                if local_idx[s_idx2][f_idx] != -1:
                                    new_mask &= ~(1 << local_idx[s_idx2][f_idx])
                                if contains_q3(new_mask):
                                    valid = False
                                    break
                            
                            if valid:
                                G = (G & ~(1 << f_idx)) | (1 << e_idx)
                                missing_edges.remove(e_idx)
                                missing_edges.add(f_idx)
                                
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
