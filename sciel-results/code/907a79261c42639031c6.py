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
        # Returns (s_idx, f_idx) where f_idx is an edge in G that is part of a Q3 with e_idx
        # Or None if no Q3 found (should not happen if can_add is false)
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_mask = subset_masks[s_idx] | (1 << l_pos)
            if contains_q3(new_mask):
                # Find a specific Q3 mask in new_mask
                # And find an edge f in that Q3 (excluding e_idx) that is in G
                # We need to map local mask back to global edges?
                # This is complex. 
                # Simpler: just return s_idx and let the caller try random edges in that subset?
                # Or we can find the q mask.
                for q in q3_masks_for_k8:
                    if (new_mask & q) == q:
                        # q is a local mask.
                        # Find global edges in q.
                        # We need to know which local bits correspond to which global edges in s_idx.
                        # We have local_idx[s_idx][g_idx] = l_pos.
                        # Inverse: for s_idx, local_pos -> g_idx?
                        # We can build a reverse map for s_idx.
                        # But doing this inside the loop is slow.
                        # Alternative: Just return s_idx.
                        # The caller will try removing random edges from G that are in s_idx.
                        return s_idx
        return None

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
                    s_idx_blocker = find_blocking_q3(G, e_idx, subset_masks)
                    if s_idx_blocker is not None:
                        # Try to remove an edge f from G that is in s_idx_blocker
                        # Get all edges in G that are in s_idx_blocker
                        # We can iterate through all edges in s_idx_blocker
                        # s_idx_blocker is a tuple of 8 vertices.
                        # Edges in this subset are C(8,2)=28.
                        # We can check which of these 28 are in G.
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
                            # Check if G - f_idx + e_idx is valid
                            # We need to check if this swap creates a Q3.
                            # New G' = G - f_idx + e_idx
                            # We only need to check Q3s containing e_idx.
                            # And ensure f_idx removal doesn't break anything (it doesn't, it only removes).
                            # Wait, G' must be Q3-free.
                            # G was Q3-free.
                            # G' adds e_idx and removes f_idx.
                            # Any new Q3 must contain e_idx.
                            # So we check: for all Q3s containing e_idx, is it fully present in G'?
                            # Q3 Q is in G' iff e_idx in Q and (Q \ {e_idx}) \subseteq G'
                            # (Q \ {e_idx}) \subseteq G' iff (Q \ {e_idx}) \subseteq G and f_idx not in (Q \ {e_idx})
                            # So we check if any Q3 containing e_idx has all other edges in G and does NOT contain f_idx.
                            
                            # This check is expensive if done naively.
                            # But we can use the subset_masks.
                            # Temporarily remove f_idx from subset_masks?
                            # Or just check:
                            # For each s_idx2 in edge_to_subsets[e_idx]:
                            #   new_mask = subset_masks[s_idx2] | (1 << local_idx[s_idx2][e_idx])
                            #   # Remove f_idx from new_mask if f_idx is in s_idx2
                            #   if local_idx[s_idx2][f_idx] != -1:
                            #       new_mask &= ~(1 << local_idx[s_idx2][f_idx])
                            #   if contains_q3(new_mask):
                            #       break
                            
                            valid = True
                            for s_idx2 in edge_to_subsets[e_idx]:
                                new_mask = subset_masks[s_idx2] | (1 << local_idx[s_idx2][e_idx])
                                if local_idx[s_idx2][f_idx] != -1:
                                    new_mask &= ~(1 << local_idx[s_idx2][f_idx])
                                if contains_q3(new_mask):
                                    valid = False
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
