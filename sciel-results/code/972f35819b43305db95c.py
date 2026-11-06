import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    # Precompute all edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Local edge mapping for 8 vertices
    local_edges_list = []
    local_edge_to_idx = {}
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_idx[(i, j)] = len(local_edges_list)
            local_edges_list.append((i, j))
            
    # Build base Q3 mask
    base_q3_mask = 0
    for k, (u, v) in enumerate(local_edges_list):
        if bin(u ^ v).count('1') == 1:
            base_q3_mask |= (1 << k)
            
    # Generate all unique Q3 masks on 8 vertices
    q3_masks = set()
    for p in permutations(range(8)):
        m = 0
        for k, (u, v) in enumerate(local_edges_list):
            if (base_q3_mask >> k) & 1:
                pu, pv = p[u], p[v]
                if pu > pv: pu, pv = pv, pu
                idx = local_edge_to_idx[(pu, pv)]
                m |= (1 << idx)
        q3_masks.add(m)
        
    q3_masks_list = list(q3_masks)
    
    # Precompute bad_local table
    # Size 2^28
    bad_local = bytearray(1 << 28)
    all_ones_28 = (1 << 28) - 1
    
    for m in q3_masks_list:
        comp = all_ones_28 ^ m
        sub = comp
        while True:
            bad_local[m | sub] = 1
            if sub == 0: break
            sub = (sub - 1) & comp
            
    # Precompute subsets and their bit mappings
    subsets = list(combinations(range(n), 8))
    subset_bit_map = {}
    for S in subsets:
        mapping = []
        for k, (u, v) in enumerate(local_edges_list):
            su, sv = S[u], S[v]
            if su > sv: su, sv = sv, su
            ge_idx = edge_index[(su, sv)]
            mapping.append((ge_idx, k))
        subset_bit_map[S] = mapping

    def check_q3(G):
        """
        Returns (True, subset) if Q3 found, else (False, None).
        """
        for S in subsets:
            local_mask = 0
            for g_idx, l_idx in subset_bit_map[S]:
                if (G >> g_idx) & 1:
                    local_mask |= (1 << l_idx)
            if bin(local_mask).count('1') >= 12:
                if bad_local[local_mask]:
                    return True, S
        return False, None

    def is_q3_free(G):
        return check_q3(G)[0] is False

    best_mask = 0
    best_count = 0
    
    # Initial baseline: K3,13 (39 edges) is Q3-free?
    # Q3 is 3-regular. Bipartite.
    # K3,13 has parts 3 and 13.
    # Any Q3 in K3,13 would need 4 vertices in each part?
    # Q3 is 4+4 bipartite.
    # K3,13 has min part size 3. So no Q3.
    # So K3,13 is Q3-free.
    # Edges: 3*13 = 39.
    
    random.seed(42)
    
    for u in range(3):
        for v in range(3, 16):
            idx = edge_index[(u,v)]
            best_mask |= (1 << idx)
    best_count = bin(best_mask).count('1')
    
    for trial in range(500):
        if time.time() > deadline - 5:
            break
            
        # Random start
        p = random.uniform(0.6, 0.8)
        current_mask = 0
        for i in range(num_edges):
            if random.random() < p:
                current_mask |= (1 << i)
                
        # Repair
        while True:
            if time.time() > deadline - 5:
                break
            has_q3, S = check_q3(current_mask)
            if not has_q3:
                break
            # Remove a random edge from S that is in current_mask
            # Get edges in S
            s_edges = []
            for g_idx, l_idx in subset_bit_map[S]:
                if (current_mask >> g_idx) & 1:
                    s_edges.append(g_idx)
            if s_edges:
                rem_idx = random.choice(s_edges)
                current_mask &= ~(1 << rem_idx)
            else:
                break # Should not happen if check_q3 said True

        if check_q3(current_mask)[0]:
            continue

        # Greedy Add
        missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
        
        improved = True
        while improved:
            improved = False
            random.shuffle(missing_edges)
            for idx in missing_edges:
                if time.time() > deadline - 5:
                    break
                test_mask = current_mask | (1 << idx)
                if is_q3_free(test_mask):
                    current_mask = test_mask
                    missing_edges.remove(idx)
                    improved = True
                    c = bin(current_mask).count('1')
                    if c > best_count:
                        best_count = c
                        best_mask = current_mask
            
        # Perturbation
        if time.time() > deadline - 10:
            continue
            
        current_edges = [i for i in range(num_edges) if (current_mask >> i) & 1]
        num_remove = random.randint(3, 8)
        if len(current_edges) > num_remove:
            to_remove = random.sample(current_edges, num_remove)
            for idx in to_remove:
                current_mask &= ~(1 << idx)
                
        # Re-improve
        missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
        improved = True
        while improved:
            improved = False
            random.shuffle(missing_edges)
            for idx in missing_edges:
                if time.time() > deadline - 5:
                    break
                test_mask = current_mask | (1 << idx)
                if is_q3_free(test_mask):
                    current_mask = test_mask
                    missing_edges.remove(idx)
                    improved = True
                    c = bin(current_mask).count('1')
                    if c > best_count:
                        best_count = c
                        best_mask = current_mask

    # Final polish
    if time.time() < deadline - 2:
        missing_edges = [i for i in range(num_edges) if not (best_mask >> i) & 1]
        random.shuffle(missing_edges)
        for idx in missing_edges:
            if time.time() > deadline - 1:
                break
            test_mask = best_mask | (1 << idx)
            if is_q3_free(test_mask):
                best_mask = test_mask
                best_count = bin(best_mask).count('1')

    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result
