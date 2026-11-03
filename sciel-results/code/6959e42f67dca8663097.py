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
    # We will try to assign signs to ALL supports such that dot products are <= 2.
    
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports) # 330
    
    # Precompute intersection info
    # For each pair (i, j), if intersection size is 3, we need to check signs.
    # If intersection size <= 2, they are always compatible (max dot 2).
    
    # We represent each support's sign configuration as an integer 0-15.
    # bits 0-3 correspond to the 4 positions in the support.
    # 0 = 1, 1 = -1.
    
    # Precompute sign values for each support index and sign config
    # sign_values[s_idx][config] = np.array of length 4 (1 or -1)
    sign_configs = np.array([list(itertools.product([1, -1], repeat=4))], dtype=np.int8) # shape (16, 4)
    
    # Map support index to its 4 indices
    support_indices = np.array(supports, dtype=np.int32) # shape (330, 4)
    
    # We need to check compatibility between any two supports i, j with intersection 3.
    # Let's identify all pairs with intersection 3.
    # There are not too many.
    
    conflict_pairs = [] # List of (i, j, intersection_indices)
    # To make local search fast, we can precompute for each pair (i,j) the mapping of sign bits.
    # If i and j share 3 indices, let the shared indices be at positions p1, p2, p3 in support i
    # and q1, q2, q3 in support j.
    # The dot product is sum(sign_i[pk] * sign_j[qk]).
    # We want this sum <= 2.
    # Since signs are +/-1, the sum is 3 if all products are 1.
    # So we want to avoid (sign_i[p1]==sign_j[q1]) AND (sign_i[p2]==sign_j[q2]) AND (sign_i[p3]==sign_j[q3]).
    
    # For local search, we maintain an array `current_signs` of length N_S, values 0-15.
    # And a boolean array `active` of length N_S.
    
    # Initialize
    rng = np.random.RandomState(42)
    current_signs = rng.randint(0, 16, size=N_S)
    active = np.ones(N_S, dtype=bool)
    
    # Function to calculate violations for a given sign config of support i
    # Violation with j if active[j] and dot > 2.
    
    # To speed up, we can precompute for each support i, the list of neighbors j with intersection 3.
    neighbors = [[] for _ in range(N_S)]
    # neighbor_info[i][j] = (pos_in_i, pos_in_j) for the 3 shared indices
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = set(supports[i]) & set(supports[j])
            if len(inter) == 3:
                # Find positions
                pos_i = [supports[i].index(x) for x in inter]
                pos_j = [supports[j].index(x) for x in inter]
                neighbors[i].append((j, pos_i, pos_j))
                neighbors[j].append((i, pos_j, pos_i))
                
    # Local Search
    # We want to minimize the number of "bad" pairs (active pairs with dot=3).
    # If we can reach 0, we have a valid set.
    # If not, we might remove some supports.
    
    # Let's track the number of violations.
    # A simple local search:
    # 1. Start with all active.
    # 2. While time < deadline:
    #    a. If no violations, break.
    #    b. Pick a random active support i that is involved in a violation.
    #    c. Try all 16 sign configs for i.
    #    d. Pick the one that minimizes the number of violations involving i.
    #    e. If the min is still > 0, and we are stuck (no improvement for many steps), 
    #       remove i from active set.
    #    f. Update current_signs[i].
    
    # To efficiently find violations, we can just recalculate for the neighbors of i.
    
    best_valid_count = 0
    best_signs = current_signs.copy()
    best_active = active.copy()
    
    # Helper to get sign value from config and position
    # config is 0-15. bit k is 0 for +1, 1 for -1?
    # Let's define: bit k of config is 0 -> +1, 1 -> -1.
    # sign_val = 1 - 2 * bit
    
    def get_sign(config, pos):
        return 1 - 2 * ((config >> pos) & 1)
    
    # Precompute sign table for speed
    sign_table = np.zeros((16, 4), dtype=np.int8)
    for c in range(16):
        for p in range(4):
            sign_table[c, p] = 1 - 2 * ((c >> p) & 1)
            
    violations_count = 0
    # Initial violation count is hard to compute fast without iterating all pairs.
    # But we can just run the search and let it find violations.
    
    # To know if a support is "bad", we check its neighbors.
    
    # We will run multiple restarts of this local search if we get stuck.
    
    for restart in range(10):
        if time.time() > deadline - 20:
            break
            
        current_signs = rng.randint(0, 16, size=N_S)
        active = np.ones(N_S, dtype=bool)
        
        # We'll use a simple hill climbing.
        # We need to track which supports are currently "in conflict".
        # A support is in conflict if it has at least one active neighbor with dot=3.
        
        in_conflict = np.zeros(N_S, dtype=bool)
        
        # Initial pass to mark conflicts?
        # It's expensive. Let's just pick random supports and try to fix them.
        
        steps = 0
        max_steps = 50000 # Arbitrary limit per restart
        
        while steps < max_steps and time.time() < deadline - 20:
            steps += 1
            
            # Find a random support in conflict?
            # We don't have a list of conflicts.
            # We can pick a random active support.
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            i = active_indices[rng.randint(len(active_indices))]
            
            # Check if i is in conflict
            has_conflict = False
            for j, pos_i, pos_j in neighbors[i]:
                if not active[j]:
                    continue
                # Calculate dot product
                # dot = sum(sign_table[current_signs[i], pos_i[k]] * sign_table[current_signs[j], pos_j[k]] for k in range(3))
                # Since pos_i and pos_j are lists of 3 indices
                s_i = sign_table[current_signs[i], pos_i]
                s_j = sign_table[current_signs[j], pos_j]
                dot = np.sum(s_i * s_j)
                if dot == 3:
                    has_conflict = True
                    break
                    
            if not has_conflict:
                # This support is fine. 
                # We could try to add a previously removed support?
                # For now, skip.
                continue
                
            # Try to find a better sign for i
            best_config = current_signs[i]
            min_violations = float('inf')
            
            for new_config in range(16):
                v_count = 0
                for j, pos_i, pos_j in neighbors[i]:
                    if not active[j]:
                        continue
                    s_i = sign_table[new_config, pos_i]
                    s_j = sign_table[current_signs[j], pos_j]
                    dot = np.sum(s_i * s_j)
                    if dot == 3:
                        v_count += 1
                        
                if v_count < min_violations:
                    min_violations = v_count
                    best_config = new_config
                    
            if min_violations == 0:
                current_signs[i] = best_config
            else:
                # If we can't fix it, remove i
                active[i] = False
                # We might want to re-add it later with a different sign?
                # For now, removal is a structural simplification.
                
            # Optimization: If we removed i, we might have resolved conflicts for others.
            # We don't track global conflict count, so we just continue.
            
        # End of local search for this restart
        
        # Count how many active supports we have
        count = np.sum(active)
        if count > best_valid_count:
            best_valid_count = count
            best_signs = current_signs.copy()
            best_active = active.copy()
            
    # Generate vectors from best_active supports
    selected_vectors_D = []
    for s_idx in range(N_S):
        if best_active[s_idx]:
            support = supports[s_idx]
            config = best_signs[s_idx]
            v = [0] * D
            for idx, pos in zip(support, range(4)):
                v[idx] = sign_table[config, pos]
            selected_vectors_D.append(v)
            
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)
    
    # Combine Type A and Type D
    if len(selected_vectors_D) > 0:
        final_vectors = np.vstack([vectors_A, selected_vectors_D])
    else:
        final_vectors = vectors_A
        
    return final_vectors
