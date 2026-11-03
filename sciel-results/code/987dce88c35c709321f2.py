import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # 1. Generate supports
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    # Map support tuple to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # 2. Precompute support conflicts
    # conf_supports[i] = list of j such that len(supports[i] & supports[j]) == 3
    conf_supports = [[] for _ in range(N_S)]
    for i in range(N_S):
        s_i = supports[i]
        mask_i = 0
        for idx in s_i:
            mask_i |= (1 << idx)
        for j in range(i + 1, N_S):
            s_j = supports[j]
            # Calculate intersection size
            # Since size is 4, we can just check
            inter = len(set(s_i).intersection(s_j))
            if inter == 3:
                conf_supports[i].append(j)
                conf_supports[j].append(i)

    # 3. Generate all vectors (nodes)
    # Node ID: i * 16 + sign_idx
    # sign_idx 0..15 corresponds to 4-bit pattern
    # We need to map sign_idx to actual signs on the support indices
    # Let's create a list of all vectors as tuples of length D
    # Or just keep the structure: (support_idx, sign_mask)
    
    # Precompute sign patterns for a 4-element support
    # sign_mask 0..15
    # bit 0 -> index 0 of support, etc.
    # 1 means +1, 0 means -1? Or just use a lookup.
    
    # For each support, we have 16 vectors.
    # Total nodes = N_S * 16
    N_V = N_S * 16
    
    # Precompute neighbors for each node
    # neighbors[node_id] = set of neighbor node_ids
    neighbors = [set() for _ in range(N_V)]
    
    # Helper to get signs from mask
    def get_signs(mask):
        return [(1 if (mask >> k) & 1 else -1) for k in range(4)]

    # For each pair of supports (i, j) with conflict, precompute sign compatibility
    # compat[i][j] = list of (sign_idx_i, sign_idx_j) that conflict?
    # Or just build the graph directly.
    
    # It is faster to iterate over all nodes and find neighbors.
    # For node (i, s_i_mask):
    #   For each j in conf_supports[i]:
    #     For each s_j_mask in 0..15:
    #       Check if signs match on intersection.
    
    # To check match on intersection:
    # We need to know which bits of s_i_mask correspond to which bits of s_j_mask.
    # Let inter_indices = sorted(set(supports[i]).intersection(supports[j]))
    # pos_in_i = [supports[i].index(k) for k in inter_indices]
    # pos_in_j = [supports[j].index(k) for k in inter_indices]
    # Match if: for all k, (s_i_mask >> pos_in_i[k]) & 1 == (s_j_mask >> pos_in_j[k]) & 1
    
    # Precompute this mapping for each conflicting support pair
    sign_maps = {}
    for i in range(N_S):
        for j in conf_supports[i]:
            if i < j:
                inter = sorted(set(supports[i]).intersection(supports[j]))
                pos_i = [supports[i].index(k) for k in inter]
                pos_j = [supports[j].index(k) for k in inter]
                # Create a mapping from s_i_mask to compatible s_j_masks
                # Or just store the positions
                sign_maps[(i, j)] = (pos_i, pos_j)
                sign_maps[(j, i)] = (pos_j, pos_i) # Note: reversed roles

    # Build graph
    for i in range(N_S):
        for s_i_mask in range(16):
            node_i = i * 16 + s_i_mask
            for j in conf_supports[i]:
                pos_i, pos_j = sign_maps[(i, j)]
                # Extract relevant bits from s_i_mask
                # We need to check which s_j_masks match
                for s_j_mask in range(16):
                    match = True
                    for k in range(3): # intersection size is 3
                        bit_i = (s_i_mask >> pos_i[k]) & 1
                        bit_j = (s_j_mask >> pos_j[k]) & 1
                        if bit_i != bit_j:
                            match = False
                            break
                    if match:
                        node_j = j * 16 + s_j_mask
                        neighbors[node_i].add(node_j)
                        neighbors[node_j].add(node_i)

    # 4. Greedy MIS with restarts
    best_size = 0
    best_set = set()
    rng = np.random.RandomState(42)
    
    def greedy_mis():
        available = set(range(N_V))
        chosen = set()
        while available:
            # Pick random node
            # To be slightly smarter, pick node with min degree in available?
            # But degrees are roughly equal.
            # Random is fine for restarts.
            v = next(iter(rng.choice(list(available))))
            chosen.add(v)
            # Remove v and neighbors
            to_remove = {v} | neighbors[v]
            available -= to_remove
        return chosen

    num_restarts = 1000
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        current_set = greedy_mis()
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # 5. Construct final vectors
    if not best_set:
        return np.array([], dtype=np.int8).reshape(0, D)

    vectors = []
    for node_id in best_set:
        s_idx = node_id // 16
        s_mask = node_id % 16
        support = supports[s_idx]
        signs = get_signs(s_mask)
        v = np.zeros(D, dtype=np.int8)
        for idx, sign in zip(support, signs):
            v[idx] = sign
        vectors.append(v)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    return final_vectors
