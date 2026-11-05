import itertools
import time
import heapq
import random

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin
    
    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute coverage masks for all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    block_masks = [0] * n_k_blocks
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 3. Greedy Construction with lazy priority queue
    uncovered_mask = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    # Priority queue: (-gain, block_idx)
    # We use lazy deletion: when we pop a block, we recompute its gain
    # and skip if it's already selected or if the gain is stale.
    
    # Initial: push all blocks with their initial gain
    # All blocks have the same initial gain (number of t-subsets in a k-set)
    # But we can just push them all with gain = popcount(mask)
    
    # Actually, for efficiency, let's just iterate greedily but use a smarter approach:
    # Since all blocks start with the same coverage, we can randomize the order
    # to break ties and potentially get different local optima.
    
    # Better: Use a priority queue with lazy evaluation
    pq = []
    for i in range(n_k_blocks):
        gain = block_masks[i].bit_count()
        heapq.heappush(pq, (-gain, i))
    
    # Track which blocks are already selected
    selected_set = set()
    
    while uncovered_mask != 0:
        if time.time() > deadline - 30:
            break
            
        best_idx = -1
        best_gain = -1
        
        # Pop from priority queue until we find a valid candidate
        while pq:
            neg_gain, idx = heapq.heappop(pq)
            
            # Skip if already selected
            if idx in selected_set:
                continue
                
            # Recompute actual gain
            gain = (block_masks[idx] & uncovered_mask).bit_count()
            
            # If gain is 0, skip
            if gain == 0:
                continue
                
            # If the gain matches what we expected (or is better), use it
            # Due to lazy evaluation, the actual gain may be less than -neg_gain
            # We just take the first valid one we pop (which has the highest priority)
            best_idx = idx
            best_gain = gain
            break
            
        if best_idx == -1:
            # No more valid blocks in the queue, rebuild
            # This shouldn't happen if uncovered_mask != 0, but just in case
            for i in range(n_k_blocks):
                if i not in selected_set:
                    gain = (block_masks[i] & uncovered_mask).bit_count()
                    if gain > 0:
                        heapq.heappush(pq, (-gain, i))
                        continue
            # Try again
            while pq:
                neg_gain, idx = heapq.heappop(pq)
                if idx in selected_set:
                    continue
                gain = (block_masks[idx] & uncovered_mask).bit_count()
                if gain > 0:
                    best_idx = idx
                    best_gain = gain
                    break
            
            if best_idx == -1:
                break
        
        selected_block_indices.append(best_idx)
        selected_set.add(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
        # Push updated gains for remaining blocks? No, we use lazy evaluation.
        # The next time we pop a block, we'll recompute its gain.
        
    # 4. Local Search: Redundancy Removal + Swap Improvement
    current_blocks = selected_block_indices[:]
    current_masks = [block_masks[i] for i in current_blocks]
    
    # Compute cover counts
    def compute_cover_counts(masks):
        cover_count = [0] * n_t_subs
        for mask in masks:
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
        return cover_count
    
    # Pass 1: Remove non-critical blocks
    cover_count = compute_cover_counts(current_masks)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        new_current_blocks = []
        new_current_masks = []
        
        for i in range(len(current_blocks)):
            mask = current_masks[i]
            is_critical = False
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    is_critical = True
                    break
                m ^= lsb
                
            if is_critical:
                new_current_blocks.append(current_blocks[i])
                new_current_masks.append(mask)
            else:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                improved = True
                
        current_blocks = new_current_blocks
        current_masks = new_current_masks
        
    # Pass 2: Try to replace pairs of blocks with a single block
    # If two blocks together cover a set of t-subsets that is a subset of
    # the t-subsets covered by some single block, we can replace them.
    
    if time.time() < deadline - 5:
        # Build a map from covered t-subsets to blocks
        # For each pair of blocks, compute their union of covered t-subsets
        # Then check if any single block covers all of them.
        
        n_current = len(current_blocks)
        
        # Precompute: for each t-subset, which current blocks cover it
        # t_sub_to_blocks: dict from t-subset index to list of block indices
        t_sub_to_blocks = [[] for _ in range(n_t_subs)]
        for bi, mask in enumerate(current_masks):
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                t_sub_to_blocks[idx].append(bi)
                m ^= lsb
        
        # Try to find blocks that can be removed by replacing with existing blocks
        # Or try to find a block that covers the union of two blocks
        
        # Simpler: Try removing each block and see if coverage is maintained
        # by the remaining blocks. If not, try to find a replacement.
        
        # Actually, let's try a different approach: 
        # For each block, check if its "unique" t-subsets (those only it covers)
        # can be covered by other blocks. If yes, remove it.
        
        # This is already done in Pass 1. Let's try swaps.
        
        # Try to replace one block with another that covers more of the unique subsets
        # while allowing removal of another block.
        
        # Given time constraints, let's just do multiple rounds of redundancy removal
        # with different random orderings to escape local optima.
        
        best_blocks = current_blocks[:]
        best_masks = current_masks[:]
        
        for restart in range(3):
            if time.time() > deadline - 3:
                break
            
            # Shuffle and re-add blocks in random order, then do redundancy removal
            shuffled = current_blocks[:]
            random.shuffle(shuffled)
            
            # Rebuild from scratch with greedy + redundancy
            # This is expensive, so let's just try a simpler local search
            
            # Try to remove blocks in random order
            trial_blocks = current_blocks[:]
            trial_masks = current_masks[:]
            trial_cover = compute_cover_counts(trial_masks)
            
            # Try removing each block in random order
            order = list(range(len(trial_blocks)))
            random.shuffle(order)
            
            for bi in order:
                if time.time() > deadline - 3:
                    break
                if bi >= len(trial_blocks):
                    continue
                mask = trial_masks[bi]
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if trial_cover[idx] == 1:
                        is_critical = True
                        break
                    m ^= lsb
                
                if not is_critical:
                    # Remove this block
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        trial_cover[idx] -= 1
                        m ^= lsb
                    trial_blocks.pop(bi)
                    trial_masks.pop(bi)
                    # Adjust indices
                    order = [x for x in order if x != bi]
                    order = [x - 1 if x > bi else x for x in order]
            
            if len(trial_blocks) < len(best_blocks):
                best_blocks = trial_blocks
                best_masks = trial_masks
        
        current_blocks = best_blocks
        current_masks = best_masks
    
    # Final blocks
    blocks = [list(all_k_blocks[i]) for i in current_blocks]
    
    return blocks
