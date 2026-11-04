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

    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_to_idx[t_sub]
            t_idxs.append(idx)
        k_sub_t_idxs.append(t_idxs)

    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def reverse_greedy_prune(seed, max_removals=None):
        """
        Start with all blocks, iteratively remove a block if no t-subset is uniquely covered by it.
        Uses NumPy for fast count updates.
        """
        rng = random.Random(seed)
        # Start with all blocks
        blocks_idx = list(range(n_k))
        used_set = set(blocks_idx)
        
        # Build initial cover count using NumPy
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            cover_count[k_sub_t_idxs[b_idx]] += 1
        
        # Find blocks that can be removed (no t-subset has count == 1 for them)
        # Vectorized approach: for each block, check if any of its t-subsets has count==1
        
        if max_removals is None:
            max_removals = len(blocks_idx)
        
        removals_done = 0
        changed = True
        while changed and removals_done < max_removals and time.time() < deadline - 10.0:
            changed = False
            
            # Find all blocks where removing them would leave some t-subset uncovered
            # i.e., blocks where all their t-subsets have count > 1
            removable = []
            for b_idx in blocks_idx:
                # Check if any t-subset in this block has count == 1
                counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                if not np.any(counts_for_block == 1):
                    removable.append(b_idx)
            
            if not removable:
                break
            
            # Randomly pick a removable block to remove
            if len(removable) > 50:
                # Sample from removable for speed
                sample_size = min(50, len(removable))
                candidates = rng.sample(removable, sample_size)
            else:
                candidates = removable
            
            # Pick one to remove
            b_idx = rng.choice(candidates)
            
            # Verify it's still removable (counts may have changed)
            counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
            if np.any(counts_for_block == 1):
                # No longer removable, continue
                continue
            
            # Remove it
            blocks_idx.remove(b_idx)
            used_set.discard(b_idx)
            cover_count[k_sub_t_idxs[b_idx]] -= 1
            removals_done += 1
            changed = True
            
            # Periodic check for more removable blocks
            if removals_done % 100 == 0:
                # Rebuild removable list
                removable = []
                for b_idx in blocks_idx:
                    counts_for_block = cover_count[k_sub_t_idxs[b_idx]]
                    if not np.any(counts_for_block == 1):
                        removable.append(b_idx)
                if not removable:
                    break
        
        return blocks_idx

    def forward_greedy(seed, use_constrained=False):
        """Forward greedy set cover as backup."""
        rng = random.Random(seed)
        uncovered = (1 << n_t) - 1
        blocks_idx = []
        used_blocks = set()
        
        # Precompute block masks
        block_masks = []
        for k_sub in all_k_subs:
            mask = 0
            for t_sub in itertools.combinations(k_sub, t):
                idx = t_sub_to_idx[t_sub]
                mask |= (1 << idx)
            block_masks.append(mask)
        
        while uncovered:
            if time.time() >= deadline - 5.0:
                break
                
            # Pick an uncovered t-subset
            if use_constrained:
                best_s_idx = -1
                min_candidates = float('inf')
                for _ in range(20):
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
                lsb = uncovered & -uncovered
                s_idx = lsb.bit_length() - 1

            candidates = t_sub_to_k_idxs[s_idx]
            best_block_idx = -1
            best_gain = -1

            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
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

    def local_search(blocks_idx, max_iter=30):
        """
        Local search: Try to remove blocks or swap them for better coverage.
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
                    used_set = set(blocks_idx)
                    best_replace = -1
                    best_score = -1
                    
                    candidates = set()
                    for t_idx in k_sub_t_idxs[b_idx]:
                        if cover_count[t_idx] == 0:
                            candidates.update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(b_idx)
                    candidates -= used_set
                    
                    if len(candidates) > 100:
                        rng = random.Random(iter_count * 1000 + i)
                        candidates = set(rng.sample(list(candidates), 100))
                    
                    for cand_idx in candidates:
                        mask = 0
                        for t_idx in k_sub_t_idxs[cand_idx]:
                            mask |= (1 << t_idx)
                        if (mask & gap_mask) != gap_mask:
                            continue
                        score = mask.bit_count()
                        if score > best_score:
                            best_score = score
                            best_replace = cand_idx
                    
                    if best_replace != -1:
                        blocks_idx[i] = best_replace
                        improved = True
                        break
                    else:
                        cover_count[k_sub_t_idxs[b_idx]] += 1
            
        return blocks_idx

    best_blocks = None
    best_len = float('inf')

    # Strategy 1: Reverse greedy pruning (primary)
    # Multiple restarts with different random seeds
    for seed in range(50):
        if time.time() >= deadline - 30.0:
            break
        
        blocks_idx = reverse_greedy_prune(seed)
        blocks_idx = reduce_blocks(blocks_idx)
        
        if blocks_idx and len(blocks_idx) < best_len:
            best_blocks = blocks_idx
            best_len = len(blocks_idx)
            
            # Early stop if we've found something very good
            if best_len <= 270:
                break

    # Strategy 2: If reverse greedy didn't find anything good, try forward greedy
    if best_len > 300:
        for seed in range(20):
            if time.time() >= deadline - 20.0:
                break
            
            blocks_idx = forward_greedy(seed)
            blocks_idx = reduce_blocks(blocks_idx)
            
            if blocks_idx and len(blocks_idx) < best_len:
                best_blocks = blocks_idx
                best_len = len(blocks_idx)

    # Strategy 3: Local search on best solution
    if best_blocks and time.time() < deadline - 5.0:
        improved_blocks = local_search(best_blocks)
        improved_blocks = reduce_blocks(improved_blocks)
        if improved_blocks and len(improved_blocks) < best_len:
            best_blocks = improved_blocks
            best_len = len(improved_blocks)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
