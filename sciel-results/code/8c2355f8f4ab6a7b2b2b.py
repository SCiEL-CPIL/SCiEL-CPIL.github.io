import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n = 16
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # 1. Generate all Q3 masks in K16
    # First, get the 840 unique Q3 edge patterns in K8
    q3_patterns = set()
    # Permutations of 0..7
    # To reduce duplicates, we can fix vertex 0 to be 0? No, Q3 is vertex-transitive.
    # We can fix vertex 0 to map to 0, and vertex 1 to map to 1?
    # Simpler: just generate all 40320 perms and add to set.
    from itertools import permutations
    for p in permutations(range(8)):
        mask = 0
        # Q3 edges in local 0..7: (u,v) if u^v is power of 2
        for u in range(8):
            for v in range(u+1, 8):
                if bin(u ^ v).count('1') == 1:
                    a, b = p[u], p[v]
                    if a > b: a, b = b, a
                    # Map local edge (a,b) to a global bit position?
                    # We need a consistent mapping for K8 edges to bits.
                    # Let's create a local edge index for K8
                    # There are 28 edges in K8.
                    pass
        # This part is messy. Let's use the previous program's local_edge_to_bit logic but cleaner.
        
    # Let's restart the Q3 generation cleanly.
    # Local K8 edges
    k8_edges = []
    local_edge_to_bit = {}
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i,j)] = len(k8_edges)
            k8_edges.append((i,j))
        
    q3_local_masks = set()
    for p in permutations(range(8)):
        mask = 0
        for u in range(8):
            for v in range(u+1, 8):
                if bin(u ^ v).count('1') == 1:
                    a, b = p[u], p[v]
                    if a > b: a, b = b, a
                    bit = local_edge_to_bit[(a,b)]
                    mask |= (1 << bit)
        q3_local_masks.add(mask)
        
    q3_local_list = list(q3_local_masks)
    # There should be 840 unique masks.
    
    # Now map to global K16
    all_q3_masks = []
    q3s_with_edge = [[] for _ in range(num_edges)]
    
    # Precompute subset mappings
    subsets = list(combinations(range(n), 8))
    
    for subset in subsets:
        # Map local vertex i to global vertex subset[i]
        # Map local edge (i,j) to global edge (subset[i], subset[j])
        # We need to map local bit (0..27) to global edge index
        local_to_global_edge = [0]*28
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u,v)]
                le_bit = local_edge_to_bit[(i,j)]
                local_to_global_edge[le_bit] = ge_idx
                
        for lm in q3_local_list:
            global_mask = 0
            for le_bit in range(28):
                if (lm >> le_bit) & 1:
                    ge_idx = local_to_global_edge[le_bit]
                    global_mask |= (1 << ge_idx)
            all_q3_masks.append(global_mask)
            
    # Deduplicate? Different subsets can yield the same global Q3?
    # No, because the vertex set is different, the edges are different.
    # Wait, if two different subsets of vertices produce the same set of edges, that's impossible for simple graphs unless the subsets are the same.
    # So all_q3_masks are unique.
    
    num_q3s = len(all_q3_masks)
    
    # Build q3s_with_edge
    for q_idx, q_mask in enumerate(all_q3_masks):
        for e in range(num_edges):
            if (q_mask >> e) & 1:
                q3s_with_edge[e].append(q_idx)
                
    # Search
    best_mask = 0
    best_count = 0
    random.seed(42)
    
    for trial in range(100):
        if time.time() > deadline:
            break
            
        # Random start
        current_mask = 0
        for i in range(num_edges):
            if random.random() < 0.7:
                current_mask |= (1 << i)
                
        # Repair
        # Find a Q3, remove an edge, repeat.
        # To speed up finding a Q3, we can just scan all_q3_masks.
        # 1.3M scans might be slow if done many times.
        # Heuristic: Pick a random edge present in graph. Check Q3s containing it.
        
        max_repair_steps = 200
        for _ in range(max_repair_steps):
            if time.time() > deadline:
                break
            
            # Try to find a Q3
            found_q3 = -1
            # Strategy: Pick a few random present edges and check their Q3s
            present_edges = [i for i in range(num_edges) if (current_mask >> i) & 1]
            if not present_edges:
                break
                
            # Shuffle to get random order
            random.shuffle(present_edges)
            
            for e in present_edges[:10]: # Check first 10 random edges
                for q_idx in q3s_with_edge[e]:
                    q_mask = all_q3_masks[q_idx]
                    if (current_mask & q_mask) == q_mask:
                        found_q3 = q_idx
                        break
                if found_q3 != -1:
                    break
            
            if found_q3 == -1:
                # It's possible we missed it if we only checked 10 edges.
                # If graph is dense, we likely found it.
                # If not, assume it's clean? No, we must be sure.
                # Fallback: Full scan if heuristic fails?
                # Given density 0.7, heuristics should work.
                # But for safety, if we don't find one, do a full scan.
                for q_idx in range(num_q3s):
                    q_mask = all_q3_masks[q_idx]
                    if (current_mask & q_mask) == q_mask:
                        found_q3 = q_idx
                        break
                        
            if found_q3 == -1:
                break # Clean
                
            # Remove a random edge from the found Q3
            q_mask = all_q3_masks[found_q3]
            q3_edges = [i for i in range(num_edges) if (q_mask >> i) & 1]
            edge_to_remove = random.choice(q3_edges)
            current_mask &= ~(1 << edge_to_remove)
            
        # Improve
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
            random.shuffle(missing_edges)
            
            for e in missing_edges:
                if time.time() > deadline:
                    break
                
                # Check if adding e creates a Q3
                can_add = True
                for q_idx in q3s_with_edge[e]:
                    q_mask = all_q3_masks[q_idx]
                    # Check if all other edges of q_mask are present
                    # q_mask without e
                    q_mask_without_e = q_mask & ~(1 << e)
                    if (current_mask & q_mask_without_e) == q_mask_without_e:
                        can_add = False
                        break
                        
                if can_add:
                    current_mask |= (1 << e)
                    improved = True
                    
        count = bin(current_mask).count('1')
        if count > best_count:
            best_count = count
            best_mask = current_mask
            
    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result
