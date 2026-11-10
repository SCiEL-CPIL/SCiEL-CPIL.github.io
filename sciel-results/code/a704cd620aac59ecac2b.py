import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    random.seed(42)
    
    # Precompute all 8-subsets of vertices
    all_subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(all_subsets)
    
    # Map global edges to indices 0..119
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
    
    # Local edge indices for K8 (28 edges)
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1
    
    # Generate Q3 masks for K8 using bipartition + perfect matching
    seen_masks = set()
    q3_masks = []
    for A in itertools.combinations(range(8), 4):
        A_set = set(A)
        B = tuple(v for v in range(8) if v not in A_set)
        for perm in itertools.permutations(B):
            mask = 0
            for i in range(4):
                a = A[i]
                matched = perm[i]
                for j in range(4):
                    b = B[j]
                    if b != matched:
                        e_idx = local_edge_idx[a][b]
                        mask |= (1 << e_idx)
            if mask not in seen_masks:
                seen_masks.add(mask)
                q3_masks.append(mask)
    
    # Precompute subset data
    subset_data = []
    for s in all_subsets:
        local_to_global = []
        for i in range(8):
            for j in range(i+1, 8):
                u_glob, v_glob = s[i], s[j]
                g_idx = edge_index[u_glob][v_glob]
                local_to_global.append(g_idx)
        
        global_to_local = [-1] * total_edges
        for l, g in enumerate(local_to_global):
            global_to_local[g] = l
        
        subset_data.append({
            'local_to_global': local_to_global,
            'global_to_local': global_to_local
        })
    
    # For each global edge, which subsets contain it
    edge_to_subsets = [[] for _ in range(total_edges)]
    for i, data in enumerate(subset_data):
        for g_idx in data['local_to_global']:
            edge_to_subsets[g_idx].append(i)
    
    # Precompute for each subset and each local edge, which Q3 masks contain that local edge
    q3_by_local_edge_per_subset = []
    for _ in range(num_subsets):
        q3_by_local_edge = [[] for _ in range(28)]
        for q_mask in q3_masks:
            for l_e in range(28):
                if (q_mask >> l_e) & 1:
                    q3_by_local_edge[l_e].append(q_mask)
        q3_by_local_edge_per_subset.append(q3_by_local_edge)
    
    # Start with K16: all edges present
    G = (1 << total_edges) - 1
    edge_count = total_edges
    
    # Helper: get 28-bit mask of G on subset i
    def get_subset_mask(G, i):
        data = subset_data[i]
        l2g = data['local_to_global']
        mask = 0
        for l in range(28):
            g_idx = l2g[l]
            if (G >> g_idx) & 1:
                mask |= (1 << l)
        return mask
    
    # Helper: check if adding edge e_idx to G creates a Q3
    def is_q3_free_addition(G, e_idx):
        for sub_idx in edge_to_subsets[e_idx]:
            data = subset_data[sub_idx]
            l2g = data['local_to_global']
            g2l = data['global_to_local']
            l_idx = g2l[e_idx]
            if l_idx == -1:
                continue
            
            # Get current mask of G on this subset
            mask = 0
            for l in range(28):
                g = l2g[l]
                if (G >> g) & 1:
                    mask |= (1 << l)
            
            # New mask with e_idx added
            new_mask = mask | (1 << l_idx)
            
            # Check Q3 masks that contain local edge l_idx
            q3_list = q3_by_local_edge_per_subset[sub_idx][l_idx]
            for q_mask in q3_list:
                if (new_mask & q_mask) == q_mask:
                    return False
        return True
    
    # Main loop: find bad subsets and remove edges
    # Track which subsets are bad
    bad_subsets = set(range(num_subsets))
    
    while bad_subsets and time.time() < deadline:
        # Pick a random bad subset
        idx = random.choice(list(bad_subsets))
        S = all_subsets[idx]
        
        # Get current mask
        mask = get_subset_mask(G, idx)
        
        # Check if still bad
        still_bad = False
        violated_pattern = None
        for q_mask in q3_masks:
            if (mask & q_mask) == q_mask:
                still_bad = True
                violated_pattern = q_mask
                break
        
        if not still_bad:
            bad_subsets.discard(idx)
            continue
        
        # Find edges in the violated pattern that are in G
        pattern_edges_local = []
        for l in range(28):
            if (violated_pattern >> l) & 1:
                g_idx = subset_data[idx]['local_to_global'][l]
                if (G >> g_idx) & 1:
                    pattern_edges_local.append(g_idx)
        
        if not pattern_edges_local:
            bad_subsets.discard(idx)
            continue
        
        # Remove a random edge from the pattern
        e_to_remove = random.choice(pattern_edges_local)
        G &= ~(1 << e_to_remove)
        edge_count -= 1
        
        # Update bad_subsets: only subsets containing e_to_remove might have changed
        affected = set(edge_to_subsets[e_to_remove])
        
        new_bad = set()
        for sidx in bad_subsets:
            if sidx in affected:
                # Recheck
                m = get_subset_mask(G, sidx)
                is_bad = False
                for q_mask in q3_masks:
                    if (m & q_mask) == q_mask:
                        is_bad = True
                        break
                if is_bad:
                    new_bad.add(sidx)
            else:
                new_bad.add(sidx)
        
        bad_subsets = new_bad
    
    # Try to add back edges if possible
    if time.time() < deadline - 5.0:
        for e_idx in range(total_edges):
            if time.time() >= deadline - 5.0:
                break
            if (G >> e_idx) & 1:
                continue
            if is_q3_free_addition(G, e_idx):
                G |= (1 << e_idx)
                edge_count += 1
    
    # Convert to edge list
    result = []
    for u, v in edge_list:
        e_idx = edge_index[u][v]
        if (G >> e_idx) & 1:
            result.append([u, v])
    
    return result
