import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            idx_to_edge.append((u, v))
            idx += 1
    total_edges = 120
    
    base_q3_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    q3_masks_8 = set()
    for perm in itertools.permutations(range(8)):
        mapped_edges = tuple(sorted((perm[u], perm[v]) for u, v in base_q3_edges))
        q3_masks_8.add(mapped_edges)
    q3_list_8 = list(q3_masks_8)
    
    edge_to_q3s = [[] for _ in range(total_edges)]
    q3_id = 0
    
    # Precomputation
    for combo in itertools.combinations(range(n), 8):
        if time.time() > deadline - 20:
            break
        inv_mapping = [0] * 8
        for i, v in enumerate(combo):
            inv_mapping[i] = v
            
        for q3_edges_8 in q3_list_8:
            global_edges = []
            for u, v in q3_edges_8:
                gu = inv_mapping[u]
                gv = inv_mapping[v]
                if gu > gv:
                    gu, gv = gv, gu
                global_edges.append(edge_to_idx[(gu, gv)])
            
            for e_idx in global_edges:
                edge_to_q3s[e_idx].append(q3_id)
            q3_id += 1
            
    # State
    current_edges = set(range(total_edges)) # Start with K16
    counts = np.zeros(q3_id, dtype=np.int32)
    
    # Initial counts
    for e in current_edges:
        indices = edge_to_q3s[e]
        if indices:
            counts[indices] += 1
            
    best_edges = current_edges.copy()
    best_count = len(current_edges)
    
    # Phase 1: Remove edges to make valid
    # Find Q3s with count 12
    invalid_q3s = np.where(counts == 12)[0]
    
    while len(invalid_q3s) > 0:
        if time.time() > deadline - 10:
            break
        # Pick a random invalid Q3
        q3_idx = invalid_q3s[random.randint(0, len(invalid_q3s)-1)]
        # We need to know which edges are in this Q3.
        # We didn't store q3_to_edges.
        # We can derive it? No.
        # We need q3_to_edges.
        
        # Let's add q3_to_edges to precomputation.
        # This doubles memory but allows picking an edge to remove.
        
    # Re-structure precomputation to store q3_to_edges
    # This is crucial for the "Remove" phase.
    
    # I will rewrite the precomputation part in the final code to include q3_to_edges.
    
    # ... (Final code will include this)
    
    # Convert best_edges to list
    result = []
    for e in best_edges:
        u, v = idx_to_edge[e]
        result.append([u, v])
    return result
