import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute Q3 edge masks in K8 (local coordinates 0..7)
    # Q3: vertices 0..7, edges between i,j if i^j is a power of 2
    q3_edges_local = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges_local.append((i, j))
    
    # Map local edge (i,j) to bit position in 28-bit mask
    local_edge_to_bit = {}
    k8_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i, j)] = len(k8_edges)
            k8_edges.append((i, j))
    
    # Generate all unique Q3 masks by permuting vertices
    from itertools import permutations
    q3_masks = set()
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            a, b = p[u], p[v]
            if a > b:
                a, b = b, a
            mask |= (1 << local_edge_to_bit[(a, b)])
        q3_masks.add(mask)
    q3_masks_list = list(q3_masks)
    
    # Precompute for each 8-subset: mapping from local edge bit to global edge index
    subset_edge_maps = {}
    for subset in combinations(range(n), 8):
        mapping = {}
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v:
                    u, v = v, u
                ge_idx = edge_index[(u, v)]
                le_bit = local_edge_to_bit[(i, j)]
                mapping[le_bit] = ge_idx
        subset_edge_maps[subset] = mapping
    
    # Fast Q3 check: for a given graph mask, check all 8-subsets
    def is_q3_free(graph_mask):
        for subset, mapping in subset_edge_maps.items():
            local_mask = 0
            for le_bit, ge_idx in mapping.items():
                if (graph_mask >> ge_idx) & 1:
                    local_mask |= (1 << le_bit)
            for qm in q3_masks_list:
                if (local_mask & qm) == qm:
                    return False
        return True
    
    # Find a violating Q3 and return the global edge indices of one Q3
    def find_violating_q3_edges(mask):
        for subset, mapping in subset_edge_maps.items():
            local_mask = 0
            for le_bit, ge_idx in mapping.items():
                if (mask >> ge_idx) & 1:
                    local_mask |= (1 << le_bit)
            for qm in q3_masks_list:
                if (local_mask & qm) == qm:
                    # Return global edge indices for this Q3
                    edges = []
                    for le_bit in range(28):
                        if (qm >> le_bit) & 1:
                            edges.append(mapping[le_bit])
                    return edges
        return None
    
    # Initial good graph: start with a known Q3-free structure
    # Try K_{3,13}: vertices 0,1,2 connected to 3..15
    # That's 3*13 = 39 edges
    best_mask = 0
    for u in range(3):
        for v in range(3, 16):
            idx = edge_index[(u, v)]
            best_mask |= (1 << idx)
    best_count = bin(best_mask).count('1')
    
    # Also try other structures
    # Try: partition into 4 groups of 4, connect all between groups except some
    # Or use a random dense start and repair
    
    random.seed(42)
    
    # Strategy: multiple restarts with different initial graphs
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Create a random dense initial graph
        p = random.uniform(0.6, 0.85)
        current_mask = 0
        for i in range(num_edges):
            if random.random() < p:
                current_mask |= (1 << i)
        
        # Repair: remove edges to eliminate Q3s
        max_repair = 200
        for _ in range(max_repair):
            if time.time() > deadline - 5:
                break
            viol_edges = find_violating_q3_edges(current_mask)
            if viol_edges is None:
                break
            # Remove a random edge from the violation
            rem_idx = random.choice(viol_edges)
            current_mask &= ~(1 << rem_idx)
        
        # Check if repair succeeded
        if find_violating_q3_edges(current_mask) is not None:
            continue
        
        cur_count = bin(current_mask).count('1')
        
        # Local search: try to add missing edges
        missing = [i for i in range(num_edges) if not (current_mask >> i) & 1]
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(missing)
            for idx in missing:
                if time.time() > deadline - 5:
                    break
                test = current_mask | (1 << idx)
                if is_q3_free(test):
                    current_mask = test
                    missing.remove(idx)
                    improved = True
                    if bin(current_mask).count('1') > best_count:
                        best_count = bin(current_mask).count('1')
                        best_mask = current_mask
        
        # If we found something better, do more local search from best
        if best_mask != current_mask:
            current_mask = best_mask
    
    # Final polish: try all possible single-edge additions from best
    if time.time() < deadline - 2:
        missing = [i for i in range(num_edges) if not (best_mask >> i) & 1]
        random.shuffle(missing)
        for idx in missing:
            if time.time() > deadline - 2:
                break
            test = best_mask | (1 << idx)
            if is_q3_free(test):
                best_mask = test
                best_count = bin(best_mask).count('1')
    
    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
    
    return result
