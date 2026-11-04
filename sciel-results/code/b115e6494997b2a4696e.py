import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the bitmask of t-subsets it covers
    # and the list of t-subset indices
    k_sub_t_idxs = []
    block_masks = []
    for k_sub in all_k_subs:
        t_idxs = []
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_to_idx[t_sub]
            t_idxs.append(idx)
            mask |= (1 << idx)
        k_sub_t_idxs.append(t_idxs)
        block_masks.append(mask)

    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed, randomness=0.0):
        """
        Greedy set cover with bitmask acceleration.
        randomness: 0.0 is pure greedy, 1.0 is random selection from candidates.
        """
        rng = random.Random(seed)
        uncovered = (1 << n_t) - 1
        blocks_idx = []
        used_blocks = set()
        
        while uncovered:
            if time.time() >= deadline - 5.0:
                break
                
            # Pick an uncovered t-subset
            # We pick the one with the fewest remaining candidates to force progress
            # But for speed, we often just pick the first uncovered one or a random one.
            # Let's try: pick a random uncovered one, then find best block for it.
            # This is faster than finding the most constrained one every time.
            
            # Find an uncovered t-subset
            # Optimization: Instead of scanning all, we can just pick a random index 
            # and check if it's uncovered. Since n_t is ~11628, this is fast enough.
            s_idx = -1
            while s_idx == -1:
                s_idx = rng.randint(0, n_t - 1)
                if not ((uncovered >> s_idx) & 1):
                    s_idx = -1
                    continue
                break
            
            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1
            best_gain_second = -1 # For diversity
            
            # Calculate gains for all candidates covering this specific s_idx
            gains = []
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                gains.append((gain, b_idx))
            
            if not gains:
                break
                
            # Sort by gain
            gains.sort(reverse=True)
            
            if randomness == 0.0:
                best_gain, best_block_idx = gains[0]
            else:
                # Pick from top candidates with some probability
                # Probability distribution: P(rank i) proportional to gain
                # Simple approach: Pick top 1 with prob (1-randomness), 
                # top 2-5 with prob randomness/4, etc.
                # Let's just pick randomly from the top 5 if randomness > 0.5
                if randomness > 0.5 and len(gains) > 1:
                    top_k = min(5, len(gains))
                    _, best_block_idx = rng.choice(gains[:top_k])
                else:
                    best_gain, best_block_idx = gains[0]

            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)
            
            # Update uncovered
            uncovered &= ~block_masks[best_block_idx]

        return blocks_idx

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            cover_count[k_sub_t_idxs[b_idx]] += 1

        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                # Check if any t-subset in this block has cover_count == 1
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if np.any(counts_for_block == 1):
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    cover_count[k_sub_t_idxs[b_idx]] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def local_search_1_for_1(blocks_idx):
        """
        Try to replace blocks with better ones.
        For each block B, find its essential 5-subsets (covered only by B).
        Find a block B' that covers all essential subsets.
        If B' covers more total subsets than B, or if swapping allows further reduction, swap.
        """
        if not blocks_idx:
            return blocks_idx
        
        blocks_idx = list(blocks_idx)
        improved = True
        max_iter = 50
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < deadline - 2.0:
            improved = False
            iter_count += 1
            
            # Rebuild cover count
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in blocks_idx:
                cover_count[k_sub_t_idxs[b_idx]] += 1
            
            # Identify essential subsets for each block
            # essential_mask[b_idx] will be a bitmask of t-subsets covered only by b_idx
            essential_masks = [0] * len(blocks_idx)
            for i, b_idx in enumerate(blocks_idx):
                mask = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        mask |= (1 << t_idx)
                essential_masks[i] = mask
                
            # Try to swap
            for i, b_idx in enumerate(blocks_idx):
                ess_mask = essential_masks[i]
                if ess_mask == 0:
                    continue
                
                # Find candidates that cover ess_mask
                # We can sample candidates from the blocks covering the first essential t-subset
                first_ess_t_idx = (ess_mask & -ess_mask).bit_length() - 1
                candidates = t_sub_to_k_idxs[first_ess_t_idx]
                
                best_cand = -1
                best_score = -1
                
                # Current score of b_idx: total coverage
                current_score = block_masks[b_idx].bit_count()
                
                for cand_idx in candidates:
                    if cand_idx == b_idx:
                        continue
                    if cand_idx in blocks_idx:
                        continue # Avoid duplicates
                    
                    # Check if cand covers all essential
                    if (block_masks[cand_idx] & ess_mask) != ess_mask:
                        continue
                    
                    # Score: Does it cover more?
                    # A better score: number of new subsets it covers that are NOT covered by others?
                    # Let's use total bit count as a proxy for "utility"
                    cand_score = block_masks[cand_idx].bit_count()
                    
                    if cand_score > best_score:
                        best_score = cand_score
                        best_cand = cand_idx
                
                if best_cand != -1:
                    # Swap
                    blocks_idx[i] = best_cand
                    improved = True
                    # Restart loop after a swap to update cover counts
                    break
            
        return blocks_idx

    def local_search_rebuild(blocks_idx):
        """
        Take the most essential blocks as a core, and re-greedy the rest.
        """
        if not blocks_idx:
            return blocks_idx
            
        blocks_idx = list(blocks_idx)
        
        # Calculate cover counts to find essential blocks
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            cover_count[k_sub_t_idxs[b_idx]] += 1
            
        # Score each block by how many essential subsets it has
        block_scores = []
        for b_idx in blocks_idx:
            score = 0
            for t_idx in k_sub_t_idxs[b_idx]:
                if cover_count[t_idx] == 1:
                    score += 1
            block_scores.append((score, b_idx))
            
        # Sort by score descending
        block_scores.sort(reverse=True)
        
        # Keep the top 50% as core
        core_size = max(1, len(blocks_idx) // 2)
        core_blocks = [b for _, b in block_scores[:core_size]]
        core_set = set(core_blocks)
        
        # Determine what is covered by core
        covered_by_core = 0
        for b_idx in core_blocks:
            covered_by_core |= block_masks[b_idx]
            
        # What is still uncovered?
        # We need to cover ALL t-subsets. 
        # The core covers some. We need to add blocks to cover the rest.
        # However, the core might not cover everything. 
        # We treat the core as fixed, and we must cover the remaining uncovered t-subsets.
        
        # Uncovered mask relative to the full space? No, relative to the target.
        # The target is all t-subsets.
        # Uncovered = all_t_mask ^ covered_by_core
        all_t_mask = (1 << n_t) - 1
        to_cover = all_t_mask & ~covered_by_core
        
        if to_cover == 0:
            return core_blocks
            
        # Greedy fill for the rest
        # We cannot use blocks in core_set again? 
        # Actually, we can use ANY block, but if we use a core block again, it's redundant.
        # So we exclude core_set from candidates.
        
        new_blocks = list(core_blocks)
        used = set(core_blocks)
        current_uncovered = to_cover
        
        while current_uncovered:
            if time.time() >= deadline - 5.0:
                break
                
            # Pick an uncovered t-subset
            s_idx = -1
            while s_idx == -1:
                s_idx = random.randint(0, n_t - 1)
                if not ((current_uncovered >> s_idx) & 1):
                    s_idx = -1
                    continue
                break
            
            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                if b_idx in used:
                    continue
                gain = (block_masks[b_idx] & current_uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    
            if best_block_idx == -1:
                # No block can cover this? Should not happen if core is valid part of a cover
                # But if we removed blocks, maybe we isolated a subset?
                # If so, we must keep the original block.
                # This logic is tricky. If we can't cover it, the rebuild failed.
                # Let's just return the original blocks if this happens.
                return blocks_idx
                
            new_blocks.append(best_block_idx)
            used.add(best_block_idx)
            current_uncovered &= ~block_masks[best_block_idx]
            
        return new_blocks

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic Greedy
    blocks_idx = greedy_cover(0, randomness=0.0)
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 2. Randomized Greedy Restarts
    # We have ~115 seconds. 
    # Greedy is fast. Local search is slower.
    # Let's do 50 greedy runs, then local search on the best ones.
    
    candidates_for_ls = []
    
    for seed in range(1, 100):
        if time.time() >= deadline - 30.0:
            break
        
        # Vary randomness
        rand_val = random.random()
        blocks_idx = greedy_cover(seed, randomness=rand_val)
        blocks_idx = reduce_blocks(blocks_idx)
        
        if blocks_idx:
            candidates_for_ls.append(blocks_idx)
            if len(blocks_idx) < best_len:
                best_blocks = blocks_idx
                best_len = len(blocks_idx)

    # 3. Local Search on Top Candidates
    # Sort candidates by length
    candidates_for_ls.sort(key=len)
    
    # Take top 10 unique-ish candidates
    processed = set()
    for blocks_idx in candidates_for_ls[:10]:
        if time.time() >= deadline - 10.0:
            break
        
        # Hash to avoid processing identical sets
        h = hash(tuple(sorted(blocks_idx)))
        if h in processed:
            continue
        processed.add(h)
        
        # Apply 1-for-1 local search
        ls_blocks = local_search_1_for_1(blocks_idx)
        ls_blocks = reduce_blocks(ls_blocks)
        
        if ls_blocks and len(ls_blocks) < best_len:
            best_blocks = ls_blocks
            best_len = len(ls_blocks)
            
        # Apply Rebuild Local Search
        rb_blocks = local_search_rebuild(blocks_idx)
        rb_blocks = reduce_blocks(rb_blocks)
        rb_blocks = local_search_1_for_1(rb_blocks) # Polish after rebuild
        rb_blocks = reduce_blocks(rb_blocks)
        
        if rb_blocks and len(rb_blocks) < best_len:
            best_blocks = rb_blocks
            best_len = len(rb_blocks)

    # 4. Final Polish on Best
    if best_blocks and time.time() < deadline - 5.0:
        best_blocks = local_search_1_for_1(best_blocks)
        best_blocks = reduce_blocks(best_blocks)
        
        # Try one more rebuild on the very best
        if time.time() < deadline - 2.0:
            rb_best = local_search_rebuild(best_blocks)
            rb_best = reduce_blocks(rb_best)
            rb_best = local_search_1_for_1(rb_best)
            rb_best = reduce_blocks(rb_best)
            if rb_best and len(rb_best) < len(best_blocks):
                best_blocks = rb_best

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
