import numpy as np
import itertools
import time

D = 11

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # Precompute: for each pair of subsets, whether they are compatible
    # Compatible if |S ∩ T| <= 2
    masks = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(subsets):
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks[i] = m
    
    # Compute pairwise intersection sizes using vectorized approach
    m1 = masks[:, np.newaxis]
    m2 = masks[np.newaxis, :]
    and_matrix = m1 & m2
    
    # Popcount table for values 0 to 2^11-1
    popcount_table = np.zeros(2048, dtype=np.int32)
    for i in range(2048):
        popcount_table[i] = bin(i).count('1')
    
    popcount_matrix = popcount_table[and_matrix]
    
    # Compatible if popcount <= 2
    compatible = (popcount_matrix <= 2)
    
    # Conflict graph: edge if NOT compatible
    conflict = ~compatible
    # Make symmetric (should already be, but ensure)
    conflict = conflict | conflict.T
    # Diagonal is True (self-conflict), which is fine
    
    # Adjacency list for fast local search
    adj_list = [np.where(conflict[i])[0] for i in range(n)]
    
    def greedy_randomized(rng):
        """Randomized greedy: repeatedly pick a random low-conflict vertex."""
        remaining = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)
        
        while remaining.any():
            # Count conflicts with remaining for each candidate
            conflicts_with_remaining = np.sum(conflict[:, remaining], axis=1)
            candidates = np.where(remaining)[0]
            if len(candidates) == 0:
                break
            # Pick randomly among those with minimum conflicts
            min_conf = conflicts_with_remaining[candidates].min()
            best_cands = candidates[conflicts_with_remaining[candidates] == min_conf]
            idx = rng.choice(best_cands)
            selected[idx] = True
            remaining[idx] = False
            remaining[conflict[idx, :]] = False
        
        return selected
    
    def local_search(start_selected, rng, max_iters=200):
        """Local search: try to add vertices, or swap out one for two."""
        current = start_selected.copy()
        best = start_selected.copy()
        best_size = np.sum(best)
        
        # Precompute blocked: for each vertex, which vertices it blocks
        # blocked_by[v] = all vertices that conflict with v
        # A vertex can be added if it's not selected and doesn't conflict with any selected
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Find candidates that can be added
            sel_idx = np.where(current)[0]
            if len(sel_idx) > 0:
                blocked = np.any(conflict[sel_idx, :], axis=0)
            else:
                blocked = np.zeros(n, dtype=bool)
            
            candidates = np.where(~current & ~blocked)[0]
            
            if len(candidates) > 0:
                # Try adding a random candidate
                idx = rng.choice(candidates)
                new_sel = current.copy()
                new_sel[idx] = True
                new_size = np.sum(new_sel)
                if new_size > best_size:
                    best = new_sel.copy()
                    best_size = new_size
                current = new_sel
                continue
            
            # Stuck: try a swap - remove one, see if we can add more
            if len(sel_idx) == 0:
                break
            
            # Try removing a random selected vertex and see what becomes available
            out_idx = rng.choice(sel_idx)
            
            # Temporarily remove
            temp_current = current.copy()
            temp_current[out_idx] = False
            
            temp_sel = np.where(temp_current)[0]
            if len(temp_sel) > 0:
                temp_blocked = np.any(conflict[temp_sel, :], axis=0)
            else:
                temp_blocked = np.zeros(n, dtype=bool)
            
            temp_candidates = np.where(~temp_current & ~temp_blocked)[0]
            
            if len(temp_candidates) > 1:
                # We can add more than one back - great!
                # Add up to 2 random candidates
                add_candidates = temp_candidates[rng.choice(len(temp_candidates), size=min(2, len(temp_candidates)), replace=False)]
                for add_idx in add_candidates:
                    temp_current[add_idx] = True
                if np.sum(temp_current) > best_size:
                    best = temp_current.copy()
                    best_size = np.sum(best)
                current = temp_current
            elif len(temp_candidates) == 1:
                # Neutral swap
                temp_current[temp_candidates[0]] = True
                current = temp_current
            else:
                # Can't improve, restore
                current[out_idx] = True
        
        return best
    
    # Run multiple restarts
    rng = np.random.default_rng(42)
    best_selected = None
    best_size = 0
    
    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Randomized greedy initialization
        init = greedy_randomized(rng)
        
        # Local search refinement
        refined = local_search(init, rng, max_iters=150)
        
        size = np.sum(refined)
        if size > best_size:
            best_size = size
            best_selected = refined.copy()
        
        # Also try with different random seeds in greedy
        if r % 10 == 0:
            rng2 = np.random.default_rng(r * 12345)
            init2 = greedy_randomized(rng2)
            refined2 = local_search(init2, rng2, max_iters=100)
            size2 = np.sum(refined2)
            if size2 > best_size:
                best_size = size2
                best_selected = refined2.copy()
    
    if best_selected is None:
        # Fallback: just pick one subset
        best_selected = np.zeros(n, dtype=bool)
        best_selected[0] = True
        best_size = 1
    
    sel_indices = np.where(best_selected)[0]
    
    # Build vectors: for each selected subset, generate all 2^4 = 16 sign combinations
    vectors = []
    for idx in sel_indices:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
    
    if len(vectors) == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    return np.array(vectors, dtype=np.int64)
