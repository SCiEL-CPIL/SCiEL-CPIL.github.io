import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    
    # Map t-subset tuple to index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Full uncovered mask: all t-subsets uncovered
    full_uncovered_mask = (1 << num_t) - 1
    
    # Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    
    # block_masks[i] = bitmask of t-subsets covered by k-block i
    block_masks = [0] * num_k
    # sub_to_blocks[j] = list of block indices that cover t-subset j
    sub_to_blocks = [[] for _ in range(num_t)]
    
    for i, b in enumerate(all_k_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
            sub_to_blocks[idx].append(i)
        block_masks[i] = mask

    best_blocks = None
    best_count = float('inf')

    def greedy_cover(seed):
        """Run one randomized greedy set cover."""
        rng = random.Random(seed)
        uncovered = full_uncovered_mask
        blocks = []
        
        while uncovered:
            best_block_idx = -1
            best_gain = 0
            best_noise = 0.0
            
            # Optimization: If time is tight, we might skip some checks, 
            # but 75k blocks * 335 steps is fast enough in Py if optimized.
            # We use bit_count which is fast in C.
            
            for i in range(num_k):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    best_noise = rng.random()
                elif gain == best_gain and gain > 0:
                    # Random tie-breaking
                    if rng.random() < 0.5:
                        best_block_idx = i
                        best_noise = rng.random()
            
            if best_block_idx == -1:
                break  # Should not happen if v, k, t are valid
            
            blocks.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
            if time.time() > deadline:
                return None  # Timeout
        
        return blocks

    def prune_blocks(indices):
        """Iteratively remove redundant blocks. Returns list of indices."""
        if not indices:
            return indices
        
        # Count coverage of each t-subset
        count_cover = [0] * num_t
        for idx in indices:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & (-m)
                t_idx = lsb.bit_length() - 1
                count_cover[t_idx] += 1
                m ^= lsb
        
        # Iteratively remove non-essential blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_indices = []
            for idx in indices:
                # Check if this block has any uniquely covered t-subset
                mask = block_masks[idx]
                m = mask
                essential = False
                while m:
                    lsb = m & (-m)
                    t_idx = lsb.bit_length() - 1
                    if count_cover[t_idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                
                if essential:
                    new_indices.append(idx)
                else:
                    # Remove this block: decrement counts
                    m = mask
                    while m:
                        lsb = m & (-m)
                        t_idx = lsb.bit_length() - 1
                        count_cover[t_idx] -= 1
                        m ^= lsb
                    changed = True
            
            indices = new_indices
            
        return indices

    def local_search(indices):
        """Perform 1-swap local search to minimize block count."""
        if not indices:
            return indices
        
        indices = list(indices)
        count_cover = [0] * num_t
        for idx in indices:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & (-m)
                t_idx = lsb.bit_length() - 1
                count_cover[t_idx] += 1
                m ^= lsb
        
        improved = True
        stagnant_count = 0
        max_stagnant = 50 # Limit neutral swaps to avoid infinite loops
        
        while improved and time.time() < deadline:
            improved = False
            stagnant_count = 0
            
            # We iterate over a copy to allow modifications
            current_indices = list(indices)
            
            for i in range(len(current_indices)):
                if time.time() > deadline:
                    return indices
                
                idx = current_indices[i]
                mask = block_masks[idx]
                
                # Find essential subsets for this block
                essential_subs = []
                m = mask
                while m:
                    lsb = m & (-m)
                    t_idx = lsb.bit_length() - 1
                    if count_cover[t_idx] == 1:
                        essential_subs.append(t_idx)
                    m ^= lsb
                
                if not essential_subs:
                    # Block is redundant, remove it
                    # Decrement counts
                    m = mask
                    while m:
                        lsb = m & (-m)
                        t_idx = lsb.bit_length() - 1
                        count_cover[t_idx] -= 1
                        m ^= lsb
                    current_indices.pop(i)
                    improved = True
                    break # Restart loop with new state
                
                # Try to find a replacement block
                # Pick the first essential subset to filter candidates
                first_sub = essential_subs[0]
                candidates = sub_to_blocks[first_sub]
                
                for cand_idx in candidates:
                    if cand_idx in current_indices:
                        continue
                    
                    cand_mask = block_masks[cand_idx]
                    
                    # Check if cand covers all essential subsets of idx
                    covers_all = True
                    for sub_idx in essential_subs:
                        if not (cand_mask & (1 << sub_idx)):
                            covers_all = False
                            break
                    
                    if covers_all:
                        # Perform swap
                        # Remove old block idx
                        m = mask
                        while m:
                            lsb = m & (-m)
                            t_idx = lsb.bit_length() - 1
                            count_cover[t_idx] -= 1
                            m ^= lsb
                        
                        # Add new block cand_idx
                        m = cand_mask
                        while m:
                            lsb = m & (-m)
                            t_idx = lsb.bit_length() - 1
                            count_cover[t_idx] += 1
                            m ^= lsb
                        
                        current_indices[i] = cand_idx
                        improved = True
                        stagnant_count += 1
                        break # Restart loop with new state after swap
                
                if improved:
                    indices = current_indices
                    break
                if stagnant_count > max_stagnant:
                    break
            
        # Final prune
        return prune_blocks(indices)

    # Run multiple greedy instances with different seeds
    num_restarts = 100  # Increased restarts
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        seed = restart * 7919 + 12345
        blocks = greedy_cover(seed)
        
        if blocks is None:
            continue
        
        # Prune redundant blocks
        pruned = prune_blocks(blocks)
        
        # Local search
        searched = local_search(pruned)
        
        if len(searched) < best_count:
            best_count = len(searched)
            best_blocks = searched
            
            # Early termination if we've hit a very good number
            if best_count <= 320: # Best known is 335, so 320 is a big jump
                break

    # Fallback: if no valid result, do a simple deterministic greedy
    if best_blocks is None:
        uncovered = full_uncovered_mask
        blocks = []
        while uncovered:
            best_block_idx = -1
            best_gain = 0
            for i in range(num_k):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            if best_block_idx == -1:
                break
            blocks.append(all_k_blocks[best_block_idx])
            uncovered &= ~block_masks[best_block_idx]
        best_blocks = blocks
    else:
        # Convert indices back to tuples
        best_blocks = [all_k_blocks[idx] for idx in best_blocks]

    return best_blocks
