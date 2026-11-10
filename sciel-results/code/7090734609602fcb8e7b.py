import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    total_edges = 120
    
    # Edge indexing
    edge_index = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
            
    # Precompute all Q3 subgraphs in K16
    # Standard Q3 edges (vertices 0..7)
    standard_q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Local edge index for K8
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1
            
    # Generate all distinct Q3 masks in K8
    q3_masks_k8_set = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in standard_q3_edges:
            pu = perm[u]
            pv = perm[v]
            e_idx = local_edge_idx[pu][pv]
            mask |= (1 << e_idx)
        q3_masks_k8_set.add(mask)
    q3_masks_k8 = list(q3_masks_k8_set)
    
    # Map to K16
    all_q3_masks = []
    seen_masks = set()
    for subset in itertools.combinations(range(n), 8):
        for mask in q3_masks_k8:
            global_mask = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if (mask >> local_edge_idx[i][j]) & 1:
                        u_glob = subset[i]
                        v_glob = subset[j]
                        g_idx = edge_index[u_glob][v_glob]
                        global_mask |= (1 << g_idx)
            if global_mask not in seen_masks:
                seen_masks.add(global_mask)
                all_q3_masks.append(global_mask)
                
    # Precompute q3s_with_edge
    q3s_with_edge = [[] for _ in range(total_edges)]
    for q3_mask in all_q3_masks:
        m = q3_mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_with_edge[e_idx].append(q3_mask)
            m ^= lsb
            
    def creates_q3(G, e_idx):
        new_G = G | (1 << e_idx)
        for q3 in q3s_with_edge[e_idx]:
            if (new_G & q3) == q3:
                return True
        return False

    best_G = 0
    best_count = 0
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        G = (1 << total_edges) - 1
        removed_edges = []
        
        # Phase 1: Remove edges to eliminate Q3s
        while True:
            if time.time() > deadline:
                break
            found = False
            start_idx = random.randint(0, len(all_q3_masks) - 1)
            for i in range(len(all_q3_masks)):
                idx_q3 = (start_idx + i) % len(all_q3_masks)
                q3 = all_q3_masks[idx_q3]
                if (G & q3) == q3:
                    m = q3
                    edges_in_q3 = []
                    while m:
                        lsb = m & -m
                        e_idx = lsb.bit_length() - 1
                        edges_in_q3.append(e_idx)
                        m ^= lsb
                    e_to_remove = random.choice(edges_in_q3)
                    G &= ~(1 << e_to_remove)
                    found = True
                    break
            if not found:
                break
                
        # Phase 2: Add edges back greedily
        for e_idx in range(total_edges):
            if time.time() > deadline:
                break
            if not (G & (1 << e_idx)):
                if not creates_q3(G, e_idx):
                    G |= (1 << e_idx)
                    
        # Phase 3: Local Search
        missing_edges = [e for e in range(total_edges) if not (G & (1 << e))]
        present_edges = [e for e in range(total_edges) if (G & (1 << e))]
        
        while time.time() < deadline - 5:
            progress = False
            random.shuffle(missing_edges)
            for e_idx in missing_edges:
                if not (G & (1 << e_idx)):
                    if not creates_q3(G, e_idx):
                        G |= (1 << e_idx)
                        progress = True
                        missing_edges.remove(e_idx)
                        present_edges.append(e_idx)
            
            if not progress:
                if missing_edges and present_edges:
                    e_add = random.choice(missing_edges)
                    e_rem = random.choice(present_edges)
                    if e_add != e_rem:
                        new_G = (G & ~(1 << e_rem)) | (1 << e_add)
                        if not creates_q3(new_G, e_add):
                            G = new_G
                            present_edges.remove(e_rem)
                            missing_edges.remove(e_add)
                            missing_edges.append(e_rem)
                            progress = True
            if not progress:
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
