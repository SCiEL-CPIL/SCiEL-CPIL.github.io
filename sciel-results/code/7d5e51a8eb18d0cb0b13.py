import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n = 16
    vertices = list(range(n))
    
    # Map edges to indices
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            idx_to_edge.append((u, v))
            idx += 1
    total_edges = idx
    
    # Precompute valid Q3 patterns (28-bit)
    local_edges = []
    local_edge_to_local_idx = {}
    l_idx = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_local_idx[(u, v)] = l_idx
            local_edges.append((u, v))
            l_idx += 1
            
    # Generate valid Q3 patterns
    q3_adj = {
        0: [1, 2, 4], 1: [0, 3, 5], 2: [0, 3, 6], 3: [1, 2, 7],
        4: [0, 5, 6], 5: [1, 4, 7], 6: [2, 4, 7], 7: [3, 5, 6]
    }
    
    valid_q3_patterns = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for i in range(8):
            u_local = perm[i]
            for v_q3 in q3_adj[i]:
                v_local = perm[v_q3]
                if u_local < v_local:
                    u, v = u_local, v_local
                else:
                    u, v = v_local, u_local
                mask |= (1 << local_edge_to_local_idx[(u, v)])
        valid_q3_patterns.add(mask)
        
    # Precompute subset mappings
    subsets_8 = list(itertools.combinations(vertices, 8))
    subset_mappings = []
    subset_vertex_masks = []
    
    for subset in subsets_8:
        subset_set = set(subset)
        global_indices = []
        v_mask = 0
        for v in subset:
            v_mask |= (1 << v)
        for u, v in idx_to_edge:
            if u in subset_set and v in subset_set:
                global_indices.append(edge_to_idx[(u, v)])
        if len(global_indices) == 28:
            subset_mappings.append(global_indices)
            subset_vertex_masks.append(v_mask)
        else:
            subset_mappings.append(None)
            subset_vertex_masks.append(0)
            
    # Map edge to subsets containing it
    edge_to_subsets = [[] for _ in range(total_edges)]
    for s_idx, v_mask in enumerate(subset_vertex_masks):
        if v_mask == 0:
            continue
        # For each edge, check if it's in this subset
        # This is O(total_edges * num_subsets) = 120 * 12870 = 1.5M ops. Fast.
        for e_idx in range(total_edges):
            u, v = idx_to_edge[e_idx]
            if (v_mask >> u) & 1 and (v_mask >> v) & 1:
                edge_to_subsets[e_idx].append(s_idx)
                
    best_mask = 0
    best_count = 0
    
    all_edges_indices = list(range(total_edges))
    
    while time.time() < deadline:
        current_mask = 0
        random.shuffle(all_edges_indices)
        
        for e_idx in all_edges_indices:
            bit = 1 << e_idx
            current_mask |= bit
            
            found_q3 = False
            # Check subsets containing this edge
            for s_idx in edge_to_subsets[e_idx]:
                mapping = subset_mappings[s_idx]
                if mapping is None:
                    continue
                pattern = 0
                # Extract 28-bit pattern
                for k, g_idx in enumerate(mapping):
                    if (current_mask >> g_idx) & 1:
                        pattern |= (1 << k)
                if pattern in valid_q3_patterns:
                    found_q3 = True
                    break
                    
            if found_q3:
                current_mask &= ~bit
            else:
                pass
                
        count = bin(current_mask).count('1')
        if count > best_count:
            best_count = count
            best_mask = current_mask
            
    result = []
    for i in range(total_edges):
        if (best_mask >> i) & 1:
            u, v = idx_to_edge[i]
            result.append([u, v])
            
    return result
