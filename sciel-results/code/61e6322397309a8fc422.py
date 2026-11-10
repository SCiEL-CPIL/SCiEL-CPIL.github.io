"""Program to find a large pairwise non-coprime subset of [1,255] containing 255."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b)>1 for all distinct pairs. Maximize cardinality.
    
    Strategy: 
    1. Identify all candidates in [1, 255] that share a factor with 255 (div by 3, 5, or 17).
    2. Construct the compatibility graph using bitmasks for efficiency.
    3. Use a greedy initialization combined with multi-restart local search (1-swap and 2-swap)
       to find the maximum clique containing 255.
    """
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    n = 255
    # Candidates must share a factor with 255
    candidates = [x for x in range(1, n + 1) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    num_cands = len(candidates)
    
    # Map candidate value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build compatibility bitmask
    # compat[i] is a bitmask where bit j is set if gcd(candidates[i], candidates[j]) > 1
    compat = [0] * num_cands
    for i in range(num_cands):
        for j in range(i + 1, num_cands):
            if math.gcd(candidates[i], candidates[j]) > 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    # Index of 255
    idx_255 = val_to_idx[255]
    
    best_size = 0
    best_set = set([255])
    
    # Greedy initialization function
    def greedy_init(order):
        """Build a clique greedily following the given order of indices."""
        S = set([idx_255])
        for idx in order:
            if idx == idx_255:
                continue
            # Check if idx is compatible with all in S
            # We can use bitmasks for this check
            # Let mask_S be the bitmask of all elements in S
            # We need (compat[idx] & mask_S) == mask_S
            mask_S = 0
            for s in S:
                mask_S |= (1 << s)
            if (compat[idx] & mask_S) == mask_S:
                S.add(idx)
        return S

    # Local search: 1-swap
    def local_search_1_swap(S):
        """Try to improve set S by swapping one element out and one in."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            for i, u in enumerate(S_list):
                if time.time() > deadline:
                    break
                # Try removing u
                S_minus_u = S - {u}
                # Find candidates not in S that are compatible with all in S_minus_u
                # and prefer those that were incompatible with u (to fix the block)
                mask_S_minus_u = 0
                for s in S_minus_u:
                    mask_S_minus_u |= (1 << s)
                
                for w in range(num_cands):
                    if w in S:
                        continue
                    if w == idx_255:
                        continue
                    # Check if w is compatible with all in S_minus_u
                    if (compat[w] & mask_S_minus_u) == mask_S_minus_u:
                        # Check if w is actually better (e.g., connects to more outside nodes, or just valid)
                        # Since we just want max size, any valid swap that increases size is good.
                        # But 1-swap keeps size same. We need 1-swap to enable future 1-adds?
                        # Or we can do 1-swap to rearrange, then 1-add.
                        # Actually, if we remove u and add w, size is same.
                        # This is only useful if w has higher degree to allow further additions.
                        # For now, let's just do 1-add and 2-swap (remove 2, add 1? No, remove 1, add 2? No.)
                        # Standard Max Clique local search: 
                        # 1. Try to add a node.
                        # 2. If stuck, try to swap: remove a node, add a node (size same), then try to add again.
                        pass
        return S

    # Better Local Search: 1-Add and 1-Swap (size preserving to unlock adds)
    def improve(S):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 1. Try to add a node
            mask_S = 0
            for s in S:
                mask_S |= (1 << s)
            
            added = False
            # Sort candidates by degree for better adds
            # Precompute degrees
            if not hasattr(improve, 'degrees'):
                improve.degrees = [0] * num_cands
                for i in range(num_cands):
                    improve.degrees[i] = bin(compat[i]).count('1')
            
            candidates_to_add = [i for i in range(num_cands) if i not in S and i != idx_255]
            candidates_to_add.sort(key=lambda x: -improve.degrees[x])
            
            for w in candidates_to_add:
                if (compat[w] & mask_S) == mask_S:
                    S.add(w)
                    mask_S |= (1 << w)
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # 2. Try to swap: remove u, add w (size same), then try to add more
            S_list = list(S)
            for u in S_list:
                if time.time() > deadline:
                    break
                if u == idx_255:
                    continue
                S_minus_u = S - {u}
                mask_S_minus_u = 0
                for s in S_minus_u:
                    mask_S_minus_u |= (1 << s)
                
                # Find w not in S, compatible with S_minus_u
                # Prefer w that was incompatible with u?
                for w in range(num_cands):
                    if w in S:
                        continue
                    if w == idx_255:
                        continue
                    if (compat[w] & mask_S_minus_u) == mask_S_minus_u:
                        # Check if this swap is beneficial
                        # Simple heuristic: if w has higher degree than u, or if w connects to more potential adds
                        # For now, just do the swap and see if we can add more
                        S_trial = S_minus_u | {w}
                        # Try to add to S_trial
                        mask_trial = 0
                        for s in S_trial:
                            mask_trial |= (1 << s)
                        
                        added_more = False
                        for v in range(num_cands):
                            if v in S_trial:
                                continue
                            if (compat[v] & mask_trial) == mask_trial:
                                S_trial.add(v)
                                added_more = True
                                break
                        
                        if added_more:
                            S = S_trial
                            improved = True
                            break
                
                if improved:
                    break
        return S

    # Multiple restarts
    for trial in range(100):
        if time.time() > deadline:
            break
            
        # Create a random order biased towards high degree
        indices = list(range(num_cands))
        random.shuffle(indices)
        
        # Sort by degree descending, but keep some randomness
        # Actually, just shuffle and greedy
        S = greedy_init(indices)
        S = improve(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
        # Early stop if we find a very large clique (unlikely to beat 85 by much)
        if best_size > 90:
            break

    # Convert indices back to values
    result = [candidates[i] for i in best_set]
    return result
# EVOLVE-BLOCK-END
