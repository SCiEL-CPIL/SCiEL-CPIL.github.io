"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = P['n']
    m = P['m']
    num_masks = 1 << n  # 4096
    
    # 1. Precompute comparability matrix
    # comp[i, j] = 1 if i and j are comparable (one strict subset of other)
    masks = np.arange(num_masks, dtype=np.int32)
    
    # Efficient construction of comp using broadcasting
    mask_i = masks[:, None]
    mask_j = masks[None, :]
    subset_ij = (mask_i | mask_j) == mask_j
    subset_ji = (mask_i | mask_j) == mask_i
    not_equal = mask_i != mask_j
    comp = ((subset_ij | subset_ji) & not_equal).astype(np.int8)
    
    # 2. Greedy Initialization
    selected = np.zeros(num_masks, dtype=bool)
    selected_indices = []
    
    # Greedy: pick set with max degree, then next max degree to selected, etc.
    scores = comp.sum(axis=1).astype(np.int32)
    
    for _ in range(m):
        if time.time() > deadline - 20:
            break
        # Pick argmax of scores, among not selected
        if len(selected_indices) == 0:
            best_idx = int(np.argmax(scores))
        else:
            # Marginal gain for x = sum(comp[x, k] for k in selected_indices)
            gains = comp[:, selected_indices].sum(axis=1).astype(np.int32)
            gains[selected] = -1
            best_idx = int(np.argmax(gains))
            
        selected_indices.append(best_idx)
        selected[best_idx] = True
        
    # 3. Local Search with Single Swaps
    current_set = set(selected_indices)
    in_set = np.zeros(num_masks, dtype=bool)
    for idx in selected_indices:
        in_set[idx] = True
        
    # score_in_set[x] = sum(comp[x, k] for k in current_set)
    score_in_set = comp[:, list(current_set)].sum(axis=1).astype(np.int32)
    
    # Calculate initial score
    best_score = 0
    for idx in selected_indices:
        best_score += score_in_set[idx]
    best_score //= 2
    
    current_score = best_score
    best_indices = list(selected_indices)
    
    # Precompute for faster swap search
    # We want to find the best (out, in) pair where out is in set, in is not in set
    # delta = score_in_set[in] - score_in_set[out] + comp[in, out]
    # Wait, the change in score when we swap out->in:
    # Old contribution of out: score_in_set[out] - comp[out, out] (but comp[out,out]=0)
    # Actually, score_in_set[x] includes comp[x, y] for all y in set.
    # If we remove out and add in:
    # New score = Old score - (contribution of out) + (contribution of in)
    # Contribution of out to total score = sum_{y in set, y!=out} comp[out, y] = score_in_set[out]
    # But wait, the total score is sum_{x in set} score_in_set[x] / 2.
    # Let S be the set. Score(S) = sum_{x in S} (sum_{y in S, y!=x} comp[x,y]) / 2 = sum_{x in S} score_in_set[x] / 2.
    # If we replace out with in:
    # New score_in_set'[x] = score_in_set[x] - comp[x, out] + comp[x, in] for all x.
    # New total = sum_{x in S \ {out}} score_in_set'[x] / 2 + score_in_set'[in] / 2
    # This is getting complex. Let's use the direct delta formula.
    # Delta = (score_in_set[in] - comp[in, out]) - (score_in_set[out] - comp[out, in])
    # Since comp is symmetric, comp[in, out] = comp[out, in].
    # Delta = score_in_set[in] - score_in_set[out]
    # Wait, is that right?
    # Let's re-derive.
    # Score(S) = 1/2 * sum_{x in S} sum_{y in S, y!=x} comp[x,y]
    # Let S' = (S \ {out}) U {in}
    # Score(S') - Score(S) = 1/2 * [ sum_{x in S'} sum_{y in S', y!=x} comp[x,y] - sum_{x in S} sum_{y in S, y!=x} comp[x,y] ]
    # Terms involving neither in nor out cancel out.
    # Terms involving out: - sum_{y in S, y!=out} comp[out, y] - sum_{y in S, y!=out} comp[y, out] = -2 * score_in_set[out]
    # Wait, score_in_set[out] = sum_{y in S, y!=out} comp[out, y].
    # So we subtract 2 * score_in_set[out] from the numerator? No.
    # The term for out in the sum is sum_{y in S, y!=out} comp[out, y] + sum_{y in S, y!=out} comp[y, out] = 2 * score_in_set[out].
    # So we lose 2 * score_in_set[out] from the numerator.
    # Terms involving in: + sum_{y in S', y!=in} comp[in, y] + sum_{y in S', y!=in} comp[y, in]
    # S' \ {in} = S \ {out}.
    # So we gain 2 * sum_{y in S \ {out}} comp[in, y] = 2 * (score_in_set[in] - comp[in, out])
    # So Delta * 2 = 2 * (score_in_set[in] - comp[in, out]) - 2 * score_in_set[out]
    # Delta = score_in_set[in] - comp[in, out] - score_in_set[out]
    
    # Let's verify with a small example.
    # S = {1, 2}. comp[1,2]=1, comp[2,1]=1, comp[1,1]=0, comp[2,2]=0.
    # score_in_set[1] = 1, score_in_set[2] = 1.
    # Score(S) = (1+1)/2 = 1.
    # Swap 1 with 3. Assume comp[3,2]=1, comp[3,1]=0.
    # S' = {2, 3}.
    # score_in_set'[2] = comp[2,3] = 1.
    # score_in_set'[3] = comp[3,2] = 1.
    # Score(S') = (1+1)/2 = 1.
    # Delta = 0.
    # Formula: Delta = score_in_set[3] - comp[3,1] - score_in_set[1]
    # score_in_set[3] (in old S) = comp[3,1] + comp[3,2] = 0 + 1 = 1.
    # Delta = 1 - 0 - 1 = 0. Correct.
    
    # Another example: S = {1, 2}. comp[1,2]=1.
    # Swap 1 with 3. comp[3,2]=0, comp[3,1]=1.
    # S' = {2, 3}. comp[2,3]=0.
    # Score(S) = 1. Score(S') = 0. Delta = -1.
    # Formula: Delta = score_in_set[3] - comp[3,1] - score_in_set[1]
    # score_in_set[3] (in old S) = comp[3,1] + comp[3,2] = 1 + 0 = 1.
    # Delta = 1 - 1 - 1 = -1. Correct.
    
    # So Delta = score_in_set[in] - comp[in, out] - score_in_set[out]
    # To maximize Delta, for a fixed out, we want to maximize score_in_set[in] - comp[in, out] among in not in S.
    
    while time.time() < deadline - 5:
        best_delta = 0
        best_out = -1
        best_in = -1
        
        # Iterate over all A in set
        for A in selected_indices:
            # We want max_{B not in set} (score_in_set[B] - comp[B, A])
            # Let val[B] = score_in_set[B] - comp[B, A]
            # We need max val[B] for B not in set.
            
            # This is O(N) per A. Total O(m*N) = 40*4096 = 160k ops.
            # We can vectorize this.
            
            # val = score_in_set - comp[:, A]
            # val[in_set] = -1
            # max_val = max(val)
            # max_idx = argmax(val)
            
            # But creating val array of size N for each A is O(m*N) memory allocation.
            # Let's try to do it in one go if possible, or just loop.
            # Since m is small (40), looping is fine.
            
            vals = score_in_set - comp[:, A]
            vals[in_set] = -1
            max_val = np.max(vals)
            max_idx = int(np.argmax(vals))
            
            delta = max_val - score_in_set[A]
            
            if delta > best_delta:
                best_delta = delta
                best_out = A
                best_in = max_idx
                
        if best_delta <= 0:
            break # No improvement possible
            
        # Apply swap
        # Update score_in_set
        # score_in_set[x] += comp[x, best_in] - comp[x, best_out]
        score_in_set += comp[:, best_in] - comp[:, best_out]
        
        # Update set structures
        idx_out = selected_indices.index(best_out)
        selected_indices[idx_out] = best_in
        current_set.remove(best_out)
        current_set.add(best_in)
        in_set[best_out] = False
        in_set[best_in] = True
        
        current_score += best_delta
        if current_score > best_score:
            best_score = current_score
            best_indices = list(selected_indices)
            
    # 4. Final Polish: Try to improve with 2-swaps?
    # Given time constraints, let's stick to single swaps but run multiple passes if time permits.
    # The loop above already runs until no improvement is found.
    
    # Convert to list of lists of Python ints
    result = []
    for mask in best_indices:
        mask_int = int(mask)
        subset = [b for b in range(n) if (mask_int >> b) & 1]
        result.append(subset)
        
    return result
# EVOLVE-BLOCK-END
