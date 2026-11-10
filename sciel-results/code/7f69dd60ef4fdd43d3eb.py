import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit
    
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
    for i, b in enumerate(all_k_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
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
            
            for i in range(num_k):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    # Add small noise to break ties randomly
                    gain += rng.random() * 0.01
                    best_gain = gain
                    best_block_idx = i
                elif gain == best_gain and gain > 0:
                    # Random tie-breaking
                    if rng.random() < 0.5:
                        best_block_idx = i
            
            if best_block_idx == -1:
                break  # Should not happen
            
            blocks.append(all_k_blocks[best_block_idx])
            uncovered &= ~block_masks[best_block_idx]
            
            if time.time() > deadline:
                return None  # Timeout
        
        return blocks
    
    def prune_blocks(blocks):
        """Iteratively remove redundant blocks."""
        if not blocks:
            return blocks
        
        # Work with indices into all_k_blocks for speed
        block_indices = []
        for b in blocks:
            # Find index in all_k_blocks
            # Since blocks are tuples from all_k_blocks, we can use a lookup
            pass
        
        # Actually, let's just work with the tuples directly and use block_masks lookup
        # Build a set for O(1) lookup
        block_set = set(blocks)
        
        # Get indices
        idx_map = {b: i for i, b in enumerate(all_k_blocks)}
        indices = [idx_map[b] for b in blocks]
        
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
        
        return [all_k_blocks[idx] for idx in indices]
    
    # Run multiple greedy instances with different seeds
    num_restarts = 30  # Adjust based on timing
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        seed = restart * 7919 + 12345  # Deterministic but varied seeds
        blocks = greedy_cover(seed)
        
        if blocks is None:
            continue
        
        # Prune redundant blocks
        pruned = prune_blocks(blocks)
        
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_blocks = pruned
        
        # Early termination if we've hit a very good number
        # Lower bound is 264, best known is 335
        if best_count <= 300:
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
    
    return best_blocks
