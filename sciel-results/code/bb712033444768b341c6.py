import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    edges = []
    edge_idx = {}
    for u in range(16):
        for v in range(u+1, 16):
            edge_idx[(u,v)] = len(edges)
            edges.append((u,v))
            
    n_edges_total = len(edges)
    
    # Start with K16
    mask = (1 << n_edges_total) - 1
    edges_count = n_edges_total
    
    # Precompute Q3 check for a specific edge?
    # No, just find any Q3.
    
    def find_q3(mask):
        # Iterate over all 8-vertex subsets? 12870.
        # For each, check if it contains Q3.
        # This is slow.
        # Instead, iterate over all pairs (A,B) of size 4.
        # 900,000 pairs.
        # For each, check if subgraph contains Q3.
        # This is also slow.
        
        # Let's just pick random edges and check if they are in a Q3?
        # No.
        
        # Let's use the random sampling idea but in reverse.
        # Pick random A, B. Check if Q3 exists.
        # If yes, return the edges of that Q3.
        
        for _ in range(1000):
            a_set = random.sample(range(16), 4)
            b_set = random.sample(range(16), 4)
            if set(a_set).intersection(b_set):
                continue
                
            # Check if subgraph on A U B contains Q3
            # Extract 16-bit mask
            sub_mask = 0
            for i, a in enumerate(a_set):
                for j, b in enumerate(b_set):
                    if a < b:
                        ei = edge_idx[(a,b)]
                    else:
                        ei = edge_idx[(b,a)]
                    if (mask >> ei) & 1:
                        sub_mask |= (1 << (i * 4 + j))
                        
            # Check against all 840 patterns?
            # We only have 70 for (0,0).
            # We need all 840.
            # Let's generate all 840 valid masks.
            
            pass
        
        return None

    # This is getting complicated.
    
    # Let's stick to the SA but optimize the check.
    # We can precompute the 16 bit-indices for each of the 100,100 pairs for a *fixed* edge?
    # No.
    
    # Let's just return the seed if we can't do better quickly.
    # But the seed is 39.
    # We can try a simple greedy:
    # Start with K16.
    # While Q3 exists:
    #   Find Q3.
    #   Remove random edge from Q3.
    
    # To find Q3 fast:
    # We can precompute all 12870 8-vertex subsets.
    # For each, precompute the 120-bit mask of edges *within* that subset?
    # No, the mask depends on G.
    
    # Let's try the SA with sample size 10.
    
    VALID_MASKS_UV = generate_valid_q3_masks()
    
    def is_q3_created(mask, edge_i):
        u, v = edges[edge_i]
        candidates_a = [x for x in range(16) if x != u]
        candidates_b = [x for x in range(16) if x != u and x != v]
        
        for _ in range(10):
            a_set = [u] + random.sample(candidates_a, 3)
            b_set = [v] + random.sample(candidates_b, 3)
            
            sub_mask = 0
            for i, a in enumerate(a_set):
                for j, b in enumerate(b_set):
                    if i == 0 and j == 0:
                        sub_mask |= 1
                    else:
                        if a < b:
                            ei = edge_idx[(a,b)]
                        else:
                            ei = edge_idx[(b,a)]
                        if (mask >> ei) & 1:
                            sub_mask |= (1 << (i * 4 + j))
            
            for pat in VALID_MASKS_UV:
                if (sub_mask & pat) == pat:
                    return True
        return False

    mask = 0
    edges_count = 0
    best_mask = 0
    best_edges = 0
    
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    while time.time() < deadline:
        i = random.randint(0, n_edges_total - 1)
        
        if (mask >> i) & 1 == 0:
            if not is_q3_created(mask, i):
                mask |= (1 << i)
                edges_count += 1
                if edges_count > best_edges:
                    best_edges = edges_count
                    best_mask = mask
        else:
            if random.random() < np.exp(1/T):
                mask &= ~(1 << i)
                edges_count -= 1
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    result = []
    for i in range(n_edges_total):
        if (best_mask >> i) & 1:
            result.append(list(edges[i]))
            
    return result
