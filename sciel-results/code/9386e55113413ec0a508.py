import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the bitmask of t-subsets it covers
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

    # Build conflict graph: two blocks are "conflicting" if their intersection
    # is less than (k - t) points. In the dual hypergraph, we want a minimal
    # transversal (hitting set) that covers all t-subsets.
    # A minimal covering design corresponds to a minimal hitting set in the dual.
    # We'll use a greedy approach that prefers blocks with high coverage but
    # also considers structural properties.

    # For each block, precompute its intersection size with all other blocks
    # This is expensive, so we'll compute on-demand or use a different strategy.

    def greedy_cover(seed, use_structural=False):
        """Greedy set cover with structural awareness."""
        rng = random.Random(seed)
        uncovered = (1 << n_t) - 1
        blocks_idx = []
        used_blocks = set()
        
        while uncovered:
            if time.time() >= deadline - 5.0:
                break
                
            # Pick an uncovered t-subset
            if use_structural:
                # Find uncovered t-subset with fewest candidate blocks
                # that haven't been used yet
                best_s_idx = -1
                min_candidates = float('inf')
                for _ in range(30):
                    while True:
                        s_idx = rng.randint(0, n_t - 1)
                        if (uncovered >> s_idx) & 1:
                            break
                    candidates = [b for b in t_sub_to_k_idxs[s_idx] if b not in used_blocks]
                    if len(candidates) < min_candidates:
                        min_candidates = len(candidates)
                        best_s_idx = s_idx
                    if min_candidates == 1:
                        break
                s_idx = best_s_idx
            else:
                # Pick a random uncovered t-subset
                while True:
                    s_idx = rng.randint(0, n_t - 1)
                    if (uncovered >> s_idx) & 1:
                        break

            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1
            best_overlap_penalty = 0

            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                
                if use_structural:
                    # Penalize blocks that overlap too much with already chosen blocks
                    # Compute overlap with existing blocks
                    overlap_penalty = 0
                    for existing in used_blocks:
                        # Count common points
                        common = len(set(all_k_subs[b_idx]) & set(all_k_subs[existing]))
                        if common < (k - t):
                            overlap_penalty += (k - t) - common
                else:
                    overlap_penalty = 0
                
                # Adjusted score: gain minus penalty
                score = gain - overlap_penalty * 0.1
                
                if score > best_gain or (score == best_gain and gain > best_gain):
                    best_gain = score
                    best_block_idx = b_idx

            if best_block_idx == -1:
                break

            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]

        return blocks_idx

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            cover_count[k_sub_t_idxs[b_idx]] += 1

        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if np.any(counts_for_block == 1):
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    cover_count[k_sub_t_idxs[b_idx]] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def local_search(blocks_idx, max_iter=80):
        """
        Local search with swap operations to improve coverage efficiency.
        """
        if not blocks_idx:
            return blocks_idx
        
        blocks_idx = list(blocks_idx)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline - 2.0:
            improved = False
            iter_count += 1
            
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in blocks_idx:
                cover_count[k_sub_t_idxs[b_idx]] += 1
            
            # Try removing blocks
            for i in range(len(blocks_idx)):
                if time.time() >= deadline - 3.0:
                    break
                b_idx = blocks_idx[i]
                
                cover_count[k_sub_t_idxs[b_idx]] -= 1
                
                gap_mask = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 0:
                        gap_mask |= (1 << t_idx)
                
                if gap_mask == 0:
                    blocks_idx.pop(i)
                    improved = True
                    break
                else:
                    # Try to find a replacement
                    used_set = set(blocks_idx)
                    best_replace = -1
                    best_score = -1
                    
                    candidates = set()
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 0:
                            candidates.update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(b_idx)
                    candidates -= used_set
                    
                    if len(candidates) > 300:
                        rng = random.Random(iter_count * 1000 + i)
                        candidates = set(rng.sample(list(candidates), 300))
                    
                    for cand_idx in candidates:
                        if (block_masks[cand_idx] & gap_mask) != gap_mask:
                            continue
                        score = (block_masks[cand_idx] & ~gap_mask).bit_count()
                        if score > best_score:
                            best_score = score
                            best_replace = cand_idx
                    
                    if best_replace != -1:
                        blocks_idx[i] = best_replace
                        improved = True
                        break
                    else:
                        cover_count[k_sub_t_idxs[b_idx]] += 1
            
            # Try swapping blocks for better coverage
            if not improved and time.time() < deadline - 3.0:
                for i in range(len(blocks_idx)):
                    if time.time() >= deadline - 3.0:
                        break
                    b_idx = blocks_idx[i]
                    
                    # Find blocks that could replace this one with better coverage
                    used_set = set(blocks_idx) - {b_idx}
                    candidates = set()
                    for t_idx in k_sub_t_idxs[b_idx]:
                        candidates.update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(b_idx)
                    candidates -= used_set
                    
                    if len(candidates) > 200:
                        rng = random.Random(iter_count * 2000 + i)
                        candidates = set(rng.sample(list(candidates), 200))
                    
                    current_mask = block_masks[b_idx]
                    current_unique = (current_mask & ~(np.bitwise_or.reduce([block_masks[j] for j in used_set]) if used_set else 0)).bit_count()
                    
                    best_swap = -1
                    best_gain = current_unique - 1
                    
                    for cand_idx in candidates:
                        new_mask = block_masks[cand_idx]
                        # Check if removing b_idx and adding cand_idx still covers everything
                        # This is complex, so we use a heuristic: prefer blocks with higher unique coverage
                        unique_coverage = 0
                        for t_idx in k_sub_t_idxs[cand_idx]:
                            # Count how many other blocks (excluding b_idx) cover this t-subset
                            count = 0
                            for other in used_set:
                                if t_idx in k_sub_t_idxs[other]:
                                    count += 1
                            if count == 0:
                                unique_coverage += 1
                        
                        if unique_coverage > best_gain:
                            best_gain = unique_coverage
                            best_swap = cand_idx
                    
                    if best_swap != -1 and best_gain > current_unique:
                        blocks_idx[i] = best_swap
                        improved = True
                        break

        return blocks_idx

    def hypergraph_independent_set_approach(seed):
        """
        Alternative approach: Use the dual hypergraph perspective.
        In the dual, vertices are t-subsets and hyperedges are k-blocks.
        We want a minimal hitting set (transversal) that intersects every hyperedge.
        This is equivalent to the original problem.
        
        We'll use a different greedy strategy that considers the structure
        of the dual hypergraph more carefully.
        """
        rng = random.Random(seed)
        uncovered = (1 << n_t) - 1
        blocks_idx = []
        used_blocks = set()
        
        # Precompute for each t-subset, the list of blocks covering it,
        # sorted by some structural property
        t_sub_candidates = {}
        for s_idx in range(n_t):
            cands = t_sub_to_k_idxs[s_idx]
            # Sort by gain (number of t-subsets covered)
            t_sub_candidates[s_idx] = sorted(cands, key=lambda b: block_masks[b].bit_count(), reverse=True)
        
        while uncovered:
            if time.time() >= deadline - 5.0:
                break
            
            # Find the most constrained uncovered t-subset
            best_s_idx = -1
            min_candidates = float('inf')
            for _ in range(50):
                while True:
                    s_idx = rng.randint(0, n_t - 1)
                    if (uncovered >> s_idx) & 1:
                        break
                candidates = [b for b in t_sub_candidates[s_idx] if b not in used_blocks]
                if len(candidates) < min_candidates:
                    min_candidates = len(candidates)
                    best_s_idx = s_idx
                if min_candidates == 1:
                    break
            
            if best_s_idx == -1:
                break
            
            s_idx = best_s_idx
            candidates = [b for b in t_sub_candidates[s_idx] if b not in used_blocks]
            
            if not candidates:
                break
            
            # Pick the best candidate
            best_block_idx = candidates[0]
            best_gain = (block_masks[best_block_idx] & uncovered).bit_count()
            
            for b_idx in candidates[1:]:
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]

        return blocks_idx

    best_blocks = None
    best_len = float('inf')

    # 1. Deterministic Greedy
    blocks_idx = greedy_cover(0, use_structural=False)
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # 2. Structural Greedy
    if time.time() < deadline - 40.0:
        blocks_idx = greedy_cover(1, use_structural=True)
        blocks_idx = reduce_blocks(blocks_idx)
        if blocks_idx and len(blocks_idx) < best_len:
            best_blocks = blocks_idx
            best_len = len(blocks_idx)

    # 3. Hypergraph-based greedy
    if time.time() < deadline - 40.0:
        blocks_idx = hypergraph_independent_set_approach(42)
        blocks_idx = reduce_blocks(blocks_idx)
        if blocks_idx and len(blocks_idx) < best_len:
            best_blocks = blocks_idx
            best_len = len(blocks_idx)

    # 4. Randomized restarts
    seeds = list(range(2, 62))
    for seed in seeds:
        if time.time() >= deadline - 35.0:
            break
        use_structural = (seed % 4 == 0)
        blocks_idx = greedy_cover(seed, use_structural=use_structural)
        blocks_idx = reduce_blocks(blocks_idx)
        if blocks_idx and len(blocks_idx) < best_len:
            best_blocks = blocks_idx
            best_len = len(blocks_idx)

    # 5. Local search on best solution
    if best_blocks and time.time() < deadline - 10.0:
        improved_blocks = local_search(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)

    # 6. Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
