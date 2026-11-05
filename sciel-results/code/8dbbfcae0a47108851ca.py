import itertools
import numpy as np
import time
import random
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 120s limit, use 115s
    deadline = start_time + 115.0

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-blocks and their t-subset indices
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_choose_t = len(list(itertools.combinations(range(k), t)))

    # Map block to its t-subset indices
    block_t_indices = np.empty((n_blocks, k_choose_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, subset in enumerate(itertools.combinations(block, t)):
            block_t_indices[i, j] = t_sub_to_idx[subset]

    # Build sparse coverage matrix: rows=blocks, cols=t-subsets
    row_indices = np.repeat(np.arange(n_blocks), k_choose_t)
    col_indices = block_t_indices.flatten()
    data = np.ones(n_blocks * k_choose_t, dtype=np.int8)
    cover_matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t))
    
    # Precompute for each t-subset, the list of blocks that cover it
    # This is useful for finding replacements
    t_sub_blocks = {}
    for i in range(n_blocks):
        for idx in block_t_indices[i]:
            if idx not in t_sub_blocks:
                t_sub_blocks[idx] = []
            t_sub_blocks[idx].append(i)

    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        used = np.zeros(n_blocks, dtype=bool)

        while uncovered.any():
            if time.time() > deadline - 5.0:
                break

            # Calculate gains: number of uncovered subsets covered by each block
            # cover_matrix is (n_blocks, n_t), uncovered is (n_t,)
            gains = (cover_matrix @ uncovered.astype(np.int8)).astype(np.int32)
            gains[used] = -1

            max_gain = gains.max()
            if max_gain <= 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if seed == 0:
                chosen = candidates[0]
            else:
                chosen = candidates[rng.randint(0, len(candidates) - 1)]

            selected.append(chosen)
            used[chosen] = True
            uncovered[block_t_indices[chosen]] = False

        return selected

    def remove_redundant(selected):
        """Remove blocks that are not essential for coverage."""
        if not selected:
            return selected
        selected = list(selected)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1
        
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            for b in selected:
                sub_idx = block_t_indices[b]
                if np.min(cover_count[sub_idx]) > 1:
                    cover_count[sub_idx] -= 1
                    changed = True
                else:
                    new_selected.append(b)
            selected = new_selected
        return selected

    def local_search(selected, max_rounds=10):
        """
        Local search with 1-opt and 2-opt swaps.
        """
        if not selected:
            return selected
        selected = list(selected)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1
        
        improved = True
        rnd = 0
        while improved and rnd < max_rounds and time.time() < deadline - 2.0:
            improved = False
            rnd += 1
            
            # Pass 1: Remove redundant
            new_selected = []
            for b in selected:
                sub_idx = block_t_indices[b]
                if np.min(cover_count[sub_idx]) > 1:
                    cover_count[sub_idx] -= 1
                    improved = True
                else:
                    new_selected.append(b)
            selected = new_selected

            # Pass 2: 1-Opt Replacement
            # For each block, try to find a replacement that covers essential subsets
            # and has higher "efficiency" (covers more non-essential or same essential with less redundancy)
            # Actually, if we replace B with B', we must cover all subsets that ONLY B covered.
            # Let E(B) be essential subsets of B.
            # B' must cover E(B).
            # Gain = (Subsets covered by B' but not B) - (Subsets covered by B but not B')
            # Since B' covers E(B), it covers all subsets B uniquely covered.
            # So Gain = |B' \ B|.
            # We want to maximize Gain.
            
            for i in range(len(selected)):
                if time.time() > deadline - 2.0:
                    break
                b = selected[i]
                sub_idx = block_t_indices[b]
                essential = sub_idx[cover_count[sub_idx] == 1]
                
                if len(essential) == 0:
                    # Can remove
                    cover_count[sub_idx] -= 1
                    selected.pop(i)
                    improved = True
                    break
                else:
                    # Find best replacement
                    best_replace = -1
                    best_gain = -1
                    first_ess = essential[0]
                    candidates = t_sub_blocks.get(first_ess, [])
                    
                    # Limit candidates for speed
                    if len(candidates) > 100:
                        rng_local = random.Random(i * 7919 + rnd * 31 + 1)
                        candidates = rng_local.sample(candidates, 100)
                    
                    essential_set = set(essential)
                    for cand in candidates:
                        if cand == b or cand in selected:
                            continue
                        cand_sub = block_t_indices[cand]
                        # Check if cand covers all essential
                        # Use numpy for speed
                        if not np.all(np.isin(essential, cand_sub)):
                            continue
                        
                        # Calculate gain: number of subsets covered by cand that are not covered by b
                        # Or simply: new coverage added to the system
                        # Since cand covers essential, it maintains validity.
                        # The "gain" in terms of potential future removals is how much it covers.
                        # Actually, we just want to swap.
                        # Let's score by: count of subsets in cand_sub where cover_count is 0 (newly covered)
                        # This is a heuristic for "quality".
                        score = np.sum(cover_count[cand_sub] == 0)
                        if score > best_gain:
                            best_gain = score
                            best_replace = cand

                    if best_replace != -1:
                        old_sub = block_t_indices[b]
                        new_sub = block_t_indices[best_replace]
                        cover_count[old_sub] -= 1
                        cover_count[new_sub] += 1
                        selected[i] = best_replace
                        improved = True
                        break
            
            # Pass 3: 2-Opt Replacement (Simplified)
            # Try to swap 2 blocks with 2 blocks that cover all essential subsets of the pair
            # and have better total new coverage.
            if time.time() < deadline - 2.0 and len(selected) >= 2:
                # Sample pairs
                n_pairs_to_try = min(20, len(selected) * (len(selected) - 1) // 2)
                pair_indices = []
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        pair_indices.append((i, j))
                        if len(pair_indices) >= n_pairs_to_try:
                            break
                    if len(pair_indices) >= n_pairs_to_try:
                        break
                
                rng_pair = random.Random(rnd * 104729 + 13)
                if len(pair_indices) > n_pairs_to_try:
                    pair_indices = rng_pair.sample(pair_indices, n_pairs_to_try)

                for (i, j) in pair_indices:
                    if time.time() > deadline - 2.0:
                        break
                    b1 = selected[i]
                    b2 = selected[j]
                    sub1 = block_t_indices[b1]
                    sub2 = block_t_indices[b2]
                    
                    essential1 = sub1[cover_count[sub1] == 1]
                    essential2 = sub2[cover_count[sub2] == 1]
                    
                    if len(essential1) == 0 or len(essential2) == 0:
                        continue

                    # Find candidates for b1 and b2
                    cand1_list = t_sub_blocks.get(essential1[0], [])
                    cand2_list = t_sub_blocks.get(essential2[0], [])
                    
                    if len(cand1_list) > 30:
                        cand1_list = random.Random(i*100+j).sample(cand1_list, 30)
                    if len(cand2_list) > 30:
                        cand2_list = random.Random(i*100+j+1).sample(cand2_list, 30)
                    
                    best_pair = None
                    best_score = -1
                    ess1_set = set(essential1)
                    ess2_set = set(essential2)
                    
                    for c1 in cand1_list:
                        if c1 == b1 or c1 == b2 or c1 in selected: continue
                        s1 = block_t_indices[c1]
                        if not np.all(np.isin(essential1, s1)): continue
                        
                        for c2 in cand2_list:
                            if c2 == b1 or c2 == b2 or c2 in selected or c2 == c1: continue
                            s2 = block_t_indices[c2]
                            if not np.all(np.isin(essential2, s2)): continue
                            
                            # Score: new coverage
                            score = np.sum(cover_count[s1] == 0) + np.sum(cover_count[s2] == 0)
                            if score > best_score:
                                best_score = score
                                best_pair = (c1, c2)
                    
                    if best_pair:
                        c1, c2 = best_pair
                        cover_count[block_t_indices[b1]] -= 1
                        cover_count[block_t_indices[b2]] -= 1
                        cover_count[block_t_indices[c1]] += 1
                        cover_count[block_t_indices[c2]] += 1
                        selected[i] = c1
                        selected[j] = c2
                        improved = True
                        break
                        
        return selected

    def repair(selected):
        """
        Greedily repair coverage for selected blocks.
        Returns a new list of blocks.
        """
        selected = list(selected)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            cover_count[block_t_indices[b]] += 1
        
        uncovered = (cover_count == 0)
        
        while uncovered.any():
            if time.time() > deadline - 5.0:
                break
            gains = (cover_matrix @ uncovered.astype(np.int8)).astype(np.int32)
            # Exclude already selected blocks? 
            # We can add duplicates in repair if we want, but better to avoid.
            # Actually, if a block is already selected, its gain is 0 for uncovered subsets 
            # because cover_count > 0 implies it's not uncovered? 
            # No, cover_count > 0 means it's covered. 
            # So gains for selected blocks will be 0 for the uncovered part?
            # Yes, because if a subset is uncovered, cover_count is 0.
            # If block b is selected, it covers some subsets. If those subsets are uncovered, 
            # then cover_count for them was 0, but b is selected, so cover_count should be >=1.
            # Contradiction. So selected blocks cannot cover uncovered subsets.
            # So we don't need to mask selected blocks, their gain will be 0.
            
            max_gain = gains.max()
            if max_gain <= 0:
                break
            
            chosen = np.argmax(gains)
            selected.append(chosen)
            cover_count[block_t_indices[chosen]] += 1
            uncovered = (cover_count == 0)
            
        return selected

    best_blocks = None
    best_len = float('inf')

    # 1. Initial Greedy
    selected = greedy_cover(0)
    selected = remove_redundant(selected)
    selected = local_search(selected, max_rounds=5)
    best_blocks = selected
    best_len = len(selected)

    # 2. Iterated Local Search
    # Run multiple restarts or perturbations
    n_restarts = 15
    for seed in range(1, n_restarts + 1):
        if time.time() > deadline - 20.0:
            break
        
        # Option A: New Greedy Start
        if seed % 3 == 0:
            selected = greedy_cover(seed)
            selected = remove_redundant(selected)
        else:
            # Option B: Perturb Best
            selected = list(best_blocks)
            # Remove 5-10 random blocks
            n_remove = min(len(selected), random.Random(seed).randint(5, 10))
            indices_to_remove = random.Random(seed).sample(range(len(selected)), n_remove)
            for idx in sorted(indices_to_remove, reverse=True):
                selected.pop(idx)
            selected = repair(selected)
            selected = remove_redundant(selected)
        
        # Local Search
        selected = local_search(selected, max_rounds=3)
        
        if len(selected) < best_len:
            best_blocks = selected
            best_len = len(selected)

    # 3. Final Polish
    if best_blocks and time.time() < deadline - 5.0:
        best_blocks = local_search(best_blocks, max_rounds=10)
        best_blocks = remove_redundant(best_blocks)

    result_blocks = [list(all_blocks[i]) for i in best_blocks]
    return result_blocks
