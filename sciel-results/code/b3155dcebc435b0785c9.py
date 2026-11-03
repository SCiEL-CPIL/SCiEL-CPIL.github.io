import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Type D: Vectors with four non-zero entries +/- 1
    # Generate all supports (4-subsets)
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Build conflict graph for supports
    # Conflict if intersection size >= 3
    # Use bitmasks for efficiency
    neighbors = [0] * N_S
    ALL_NODES_MASK = (1 << N_S) - 1
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter_len = len(set(supports[i]) & set(supports[j]))
            if inter_len >= 3:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
                
    def greedy_fill_bitset(candidates):
        selected = 0
        c = candidates
        while c:
            # Find node with min degree in c
            min_deg = float('inf')
            best_idx = -1
            temp = c
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                deg = (c & neighbors[idx]).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_idx = idx
                temp ^= lsb
            
            if best_idx == -1: break
            
            selected |= (1 << best_idx)
            c &= ~(1 << best_idx)
            c &= ~neighbors[best_idx]
            
        return selected

    def local_search(s_mask):
        best_mask = s_mask
        best_size = best_mask.bit_count()
        improved = True
        while improved:
            improved = False
            # Get list of nodes in best_mask
            nodes = [i for i in range(N_S) if (best_mask >> i) & 1]
            for u in nodes:
                if time.time() > deadline - 10:
                    return best_mask
                # Remove u
                s_without_u = best_mask & ~(1 << u)
                
                # Calculate forbidden mask: neighbors of all nodes in s_without_u
                forbidden = 0
                temp = s_without_u
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    forbidden |= neighbors[idx]
                    temp ^= lsb
                
                # Candidates: All nodes not in s_without_u and not forbidden
                candidates = ALL_NODES_MASK & ~s_without_u & ~forbidden
                
                # Try to fill
                # Run greedy fill a few times with randomization?
                # Min-degree is deterministic. 
                # To add randomness, we can break ties randomly or use a different heuristic.
                # For now, just run it once. If we have time, we can add restarts.
                
                best_fill = greedy_fill_bitset(candidates)
                
                candidate_mask = s_without_u | best_fill
                candidate_size = candidate_mask.bit_count()
                
                if candidate_size > best_size:
                    best_size = candidate_size
                    best_mask = candidate_mask
                    improved = True
                    break # Restart the while loop with the new best
                    
        return best_mask

    best_support_mask = 0
    best_size = 0
    
    # Initial Greedy with random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 60: # Leave time for local search
            break
            
        # Randomized Greedy Initialization
        # Pick random node, then min-degree?
        # Or just shuffle and pick min-degree from remaining?
        
        # Let's use a simple randomized greedy:
        # At each step, pick a random node from candidates with min degree?
        # Or just pick a random node?
        
        # Let's try: Pick random node from available, add to set, remove neighbors.
        available = ALL_NODES_MASK
        current_mask = 0
        
        while available:
            # Pick a random set bit
            # Count bits
            count = available.bit_count()
            # Pick random index in [0, count)
            r_idx = np.random.randint(0, count)
            
            # Find r_idx-th set bit
            # This is slow in Python.
            # Alternative: 
            # Convert to list? 
            # Or use a trick:
            # 
            # Let's stick to Min-Degree but with random tie-breaking?
            # Or just use the deterministic Min-Degree greedy, but start with a random node?
            
            # Let's use the same greedy_fill_bitset but on the whole graph?
            # greedy_fill_bitset expects a mask of candidates.
            # It picks min-degree.
            
            # To randomize, we can shuffle the neighbors? No.
            
            # Let's just use the deterministic greedy_fill_bitset for initialization too.
            # It might get stuck in the same local optimum.
            
            # Better: Randomly remove a few nodes from the full set, then greedy fill?
            
            # Let's try:
            # Start with empty.
            # Pick a random node v.
            # Add v.
            # Then use greedy_fill_bitset on the remaining candidates.
            
            if available == 0: break
            
            count = available.bit_count()
            if count == 0: break
            
            # Pick random node
            # Efficient random bit pick:
            # r = np.random.randint(0, 1 << N_S)
            # while not (available & (1 << (r % N_S))): ...
            
            # Simpler:
            # Get list of indices?
            # indices_list = [i for i in range(N_S) if (available >> i) & 1]
            # v = np.random.choice(indices_list)
            
            # This is O(N_S).
            indices_list = [i for i in range(N_S) if (available >> i) & 1]
            if not indices_list: break
            v = indices_list[np.random.randint(len(indices_list))]
            
            current_mask |= (1 << v)
            available &= ~(1 << v)
            available &= ~neighbors[v]
            
            # Now fill the rest using greedy
            rest_fill = greedy_fill_bitset(available)
            current_mask |= rest_fill
            
        if current_mask.bit_count() > best_size:
            best_size = current_mask.bit_count()
            best_support_mask = current_mask
            
    # Local Search on the best found
    if time.time() < deadline - 20:
        best_support_mask = local_search(best_support_mask)
        best_size = best_support_mask.bit_count()
        
    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    # Iterate through set bits in best_support_mask
    temp = best_support_mask
    while temp:
        lsb = temp & -temp
        s_idx = lsb.bit_length() - 1
        support = supports[s_idx]
        # Generate all 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
        temp ^= lsb
            
    if selected_vectors_D:
        selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
    else:
        selected_vectors_D = np.array([], dtype=np.int64).reshape(0, D)
        
    # Combine Type A and Type D
    final_vectors = np.vstack([vectors_A, selected_vectors_D])
    
    return final_vectors
