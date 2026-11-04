<<<<<<< SEARCH
def _basic_greedy(v, k, t):
    """Original greedy: pick uncovered t-subset, extend to k-subset maximizing coverage."""
    all_tsubsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        # Pick the first uncovered t-subset as seed
        seed_subset = next(iter(uncovered))
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(combo + (x,)))
                    if new_tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove all t-subsets of this block from uncovered
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
    
    return blocks


def _greedy_cover(v, k, t, num_restarts=3, seed=42):
    """
    Randomized greedy set-cover:
    - Precompute all possible k-subsets? No, too many.
    - Instead, use a priority-based greedy: at each step, pick the k-subset
      that covers the most uncovered t-subsets.
    - To speed up, use local search: start from a random or heuristic block,
      then improve by swapping points.
    
    For C(19,7,4): v=19, k=7, t=4.
    Number of 4-subsets: C(19,4) = 3876.
    Number of 7-subsets: C(19,7) = 50388.
    
    We can precompute for each 7-subset, the list of 4-subsets it contains.
    Then greedy set cover: repeatedly pick the 7-subset covering most uncovered 4-subsets.
    """
    rng = random.Random(seed)
    
    # Precompute all k-subsets and their contained t-subsets
    all_ksubsets = list(itertools.combinations(range(v), k))
    # Map each t-subset to index
    tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    # For each k-subset, compute the set of t-subset indices it covers
    ksub_tsubsets = []
    for ksub in all_ksubsets:
        ts_idxs = set()
        for tsub in itertools.combinations(ksub, t):
            ts_idxs.add(tsub_to_idx[tsub])
        ksub_tsubsets.append(ts_idxs)
    
    # Greedy set cover
    uncovered = set(range(len(tsubsets)))
    blocks = []
    
    while uncovered:
        best_idx = -1
        best_covered = 0
        best_new_covered = set()
        
        for i, ts_set in enumerate(ksub_tsubsets):
            covered = ts_set & uncovered
            if len(covered) > best_covered:
                best_covered = len(covered)
                best_idx = i
                best_new_covered = covered
        
        if best_idx == -1:
            break
        
        blocks.append(list(all_ksubsets[best_idx]))
        uncovered -= best_new_covered
    
    return blocks


def _greedy_cover_optimized(v, k, t, num_restarts=3, seed=42):
    """
    Optimized greedy with numpy for speed.
    """
    import numpy as np
    
    rng = random.Random(seed)
    
    all_ksubsets = list(itertools.combinations(range(v), k))
    tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubsets)
    n_ksubs = len(all_ksubsets)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    # Build incidence matrix: n_ksubs x n_tsubs
    # incidence[i, j] = 1 if k-subset i contains t-subset j
    incidence = np.zeros((n_ksubs, n_tsubs), dtype=np.uint8)
    for i, ksub in enumerate(all_ksubsets):
        for tsub in itertools.combinations(ksub, t):
            j = tsub_to_idx[tsub]
            incidence[i, j] = 1
    
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    blocks = []
    
    while uncovered.any():
        # For each k-subset, count how many uncovered t-subsets it covers
        covered_counts = incidence @ uncovered
        best_idx = int(np.argmax(covered_counts))
        if covered_counts[best_idx] == 0:
            break
        blocks.append(list(all_ksubsets[best_idx]))
        uncovered[incidence[best_idx] == 1] = 0
    
    return blocks
=======
def _basic_greedy(v, k, t):
    """Fallback greedy for larger instances."""
    all_tsubsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        seed_subset = next(iter(uncovered))
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(combo + (x,)))
                    if new_tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
    
    return blocks

def _local_search_cover(v, k, t, time_limit=110):
    """
    1. Greedy initialization.
    2. Local search:
       - Try to remove blocks (if redundant).
       - Try to swap blocks (if removing creates gaps, find a block covering the gaps).
    """
    start_time = time.time()
    
    # Precomputation
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubsets)
    
    # Map t-subsets to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubsets)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Compute bitmask for each k-subset
    block_masks = [0] * n_ksubs
    blocks_covering_tsub = [[] for _ in range(n_tsubs)]
    
    for i, ksub in enumerate(all_ksubsets):
        mask = 0
        for tsub in itertools.combinations(ksub, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
            blocks_covering_tsub[idx].append(i)
        block_masks[i] = mask
    
    # 1. Greedy Initialization
    uncovered_mask = (1 << n_tsubs) - 1
    selected_indices = []
    
    while uncovered_mask:
        best_idx = -1
        best_cover = 0
        for i in range(n_ksubs):
            cover = block_masks[i] & uncovered_mask
            cnt = bin(cover).count('1')
            if cnt > best_cover:
                best_cover = cnt
                best_idx = i
        
        if best_idx == -1:
            break
        
        selected_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
        if time.time() - start_time > time_limit * 0.2:
            break

    # 2. Local Search
    selected_set = set(selected_indices)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        indices_to_check = list(selected_indices)
        random.shuffle(indices_to_check)
        
        for i in indices_to_check:
            if i not in selected_set:
                continue
            
            union_others = 0
            for j in selected_indices:
                if j != i:
                    union_others |= block_masks[j]
            
            unique_to_i = block_masks[i] & ~union_others
            
            if unique_to_i == 0:
                selected_indices.remove(i)
                selected_set.remove(i)
                improved = True
                break
            
            u_mask = unique_to_i
            best_candidate = -1
            
            while u_mask:
                lsb = u_mask & -u_mask
                t_idx = lsb.bit_length() - 1
                u_mask ^= lsb
                
                candidates = blocks_covering_tsub[t_idx]
                for c in candidates:
                    if c in selected_set:
                        continue
                    if (block_masks[c] & unique_to_i) == unique_to_i:
                        best_candidate = c
                        break
                
                if best_candidate != -1:
                    break

            if best_candidate != -1:
                selected_indices.remove(i)
                selected_indices.append(best_candidate)
                selected_set.remove(i)
                selected_set.add(best_candidate)
                improved = True
                break
    
    return [list(all_ksubsets[i]) for i in selected_indices]
>>>>>>> REPLACE
