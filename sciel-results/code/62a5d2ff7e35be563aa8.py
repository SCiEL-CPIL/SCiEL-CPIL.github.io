import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = 12
    m = 40
    num_candidates = 1 << n
    
    # Generate all masks
    masks = np.array(list(range(num_candidates)), dtype=np.uint16)
    
    # Precompute comparability matrix
    # Comp[i, j] = 1 if masks[i] subset of masks[j] or masks[j] subset of masks[i]
    # Note: masks are distinct, so no equality.
    # Vectorized check: (a & b) == a  OR  (a & b) == b
    # Since we need a matrix, we do:
    # a = masks[:, None], b = masks[None, :]
    # subset_ab = (a & b) == a
    # subset_ba = (a & b) == b
    # Comp = subset_ab | subset_ba
    
    a = masks[:, None]
    b = masks[None, :]
    # To save memory and time, we can just compute one direction and symmetrize?
    # No, just compute both.
    # 4096x4096 is 16M elements. uint8 is 16MB. Fine.
    
    # Optimization: 
    # subset_ab = (a & b) == a
    # This is 16M operations.
    
    try:
        subset_ab = (a & b) == a
        subset_ba = (a & b) == b
        Comp = (subset_ab | subset_ba).astype(np.uint8)
    except MemoryError:
        # Fallback to smaller chunks if needed, but 16MB is safe.
        Comp = np.zeros((num_candidates, num_candidates), dtype=np.uint8)
        for i in range(num_candidates):
            m_i = masks[i]
            # j is subset of i
            Comp[i, :] = ((masks & m_i) == masks).astype(np.uint8)
            # i is subset of j
            Comp[i, :] |= ((masks & m_i) == m_i).astype(np.uint8)
            # Diagonal is 1, but we don't pick same set twice, so it doesn't matter for sum, 
            # but we should set it to 0 for correctness if we ever sum including self?
            # In our swap logic, we sum over 'current' which does not contain 'j' (candidate) 
            # and does contain 'i' (to be removed).
            # So Comp[j, i] and Comp[i, j] are used.
            # Comp[i, i] is not used in the delta calculation for swapping i with j (j!=i).
            # Wait, scores_in[i] = sum_{k in current} Comp[i, k].
            # If k=i, Comp[i, i]=1.
            # scores_in[j] = sum_{k in current} Comp[j, k].
            # Delta = scores_in[j] - scores_in[i].
            # If we remove i and add j:
            # New score contribution from i's neighbors: sum_{k in current\{i}} Comp[j, k]
            # Old score contribution from i's neighbors: sum_{k in current\{i}} Comp[i, k]
            # Delta = (sum_{k in current} Comp[j, k] - Comp[j, i]) - (sum_{k in current} Comp[i, k] - Comp[i, i])
            # Note Comp[j, i] = Comp[i, j].
            # So Delta = scores_in[j] - Comp[i, j] - (scores_in[i] - 1)
            # Delta = scores_in[j] - scores_in[i] + 1 - Comp[i, j]
            
            # My previous simplified formula Delta = scores_in[j] - scores_in[i] was slightly off
            # because it didn't account for the interaction between i and j?
            # No, i and j are not both in the set simultaneously.
            # Old set: C. i in C, j not in C.
            # New set: C' = C - {i} + {j}.
            # Score(C) = Score(C \ {i}) + sum_{k in C\{i}} Comp[i, k] + (0 for i,i)
            # Score(C') = Score(C \ {i}) + sum_{k in C\{i}} Comp[j, k] + (0 for j,j)
            # Delta = sum_{k in C\{i}} (Comp[j, k] - Comp[i, k])
            # scores_in[j] = sum_{k in C} Comp[j, k] = sum_{k in C\{i}} Comp[j, k] + Comp[j, i]
            # scores_in[i] = sum_{k in C} Comp[i, k] = sum_{k in C\{i}} Comp[i, k] + Comp[i, i]
            # Comp[i, i] = 1 (since i subset i).
            # Comp[j, i] = Comp[i, j].
            # sum_{k in C\{i}} Comp[j, k] = scores_in[j] - Comp[i, j]
            # sum_{k in C\{i}} Comp[i, k] = scores_in[i] - 1
            # Delta = (scores_in[j] - Comp[i, j]) - (scores_in[i] - 1)
            # Delta = scores_in[j] - scores_in[i] - Comp[i, j] + 1
            
            pass # This logic applies to the fallback, but let's verify for the vectorized case too.

    # Correct Delta Formula:
    # Delta(i, j) = scores_in[j] - scores_in[i] - Comp[i, j] + 1
    
    # Greedy Initialization
    current = []
    in_set = np.zeros(num_candidates, dtype=bool)
    
    # Start with a chain
    chain_masks = [1 << i - 1 for i in range(1, n + 1)] # 1, 2, 4... 
    # Wait, chain of subsets:
    # S_0 = {}
    # S_1 = {0}
    # S_2 = {0, 1}
    # ...
    # S_k = {0, ..., k-1}
    # Mask for S_k is (1 << k) - 1
    for k in range(n + 1):
        mask_val = (1 << k) - 1
        idx = np.where(masks == mask_val)[0][0]
        current.append(idx)
        in_set[idx] = True
    
    # Add more greedily
    while len(current) < m:
        if time.time() > deadline:
            break
        best_idx = -1
        best_score = -1
        # Calculate score for all candidates not in set
        # scores_in = Comp[current, :].sum(axis=0)
        scores_in = Comp[current, :].sum(axis=0)
        # Mask out current
        scores_in[in_set] = -1
        best_idx = np.argmax(scores_in)
        if best_score < scores_in[best_idx]:
            best_score = scores_in[best_idx]
        current.append(best_idx)
        in_set[best_idx] = True
        
    # Local Search
    best_score_total = 0
    # Calculate initial total score
    for i in range(len(current)):
        for j in range(i+1, len(current)):
            if Comp[current[i], current[j]]:
                best_score_total += 1
                
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Calculate scores_in for all 4096
        scores_in = Comp[current, :].sum(axis=0)
        
        # Find best swap
        best_i = -1
        best_j = -1
        best_delta = 0
        
        # We want to maximize Delta = scores_in[j] - scores_in[i] - Comp[i, j] + 1
        # subject to i in current, j not in current.
        
        # Iterate over i in current
        for i_idx in current:
            # We need max over j not in current of:
            # scores_in[j] - Comp[i_idx, j]
            # Let val_j = scores_in[j] - Comp[i_idx, j]
            # Mask out current
            val = scores_in.copy()
            val[in_set] = -np.inf
            # Subtract Comp row
            val -= Comp[i_idx, :]
            
            j_idx = np.argmax(val)
            delta = val[j_idx] - scores_in[i_idx] + 1 # Wait, formula: scores_in[j] - scores_in[i] - Comp[i, j] + 1
            # val[j_idx] is scores_in[j_idx] - Comp[i_idx, j_idx]
            # So delta = val[j_idx] - scores_in[i_idx] + 1
            
            if delta > best_delta:
                best_delta = delta
                best_i = i_idx
                best_j = j_idx
                
        if best_delta > 0:
            # Perform swap
            in_set[best_i] = False
            in_set[best_j] = True
            current[current.index(best_i)] = best_j
            best_score_total += best_delta
            improved = True
        else:
            # Try a random kick if stuck?
            # For now, just break.
            # To escape local optima, we could do a random restart from scratch occasionally.
            # Given time constraints, simple local search from greedy start is a good baseline.
            # Let's add a simple random restart if we have time.
            if time.time() < deadline - 10:
                # Random restart
                current = list(np.random.choice(num_candidates, m, replace=False))
                in_set = np.zeros(num_candidates, dtype=bool)
                in_set[current] = True
                # Recalculate score
                best_score_total = 0
                for i in range(m):
                    for j in range(i+1, m):
                        if Comp[current[i], current[j]]:
                            best_score_total += 1
                improved = True # Restart loop
            else:
                break

    # Convert to lists of ints
    result = []
    for idx in current:
        mask = masks[idx]
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result

