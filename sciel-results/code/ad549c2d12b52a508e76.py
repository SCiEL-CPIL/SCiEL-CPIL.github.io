import itertools
import time
import random

def _remove_redundant(blocks_idx, block_masks, n_t_subs, deadline):
    """Remove redundant blocks (1-exchange)."""
    if not blocks_idx:
        return blocks_idx
    
    current = list(blocks_idx)
    n = len(current)
    
    # Build cover count
    cover_count = [0] * n_t_subs
    for bi in current:
        m = block_masks[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        new_current = []
        for i in range(len(current)):
            bi = current[i]
            mask = block_masks[bi]
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
                new_current.append(bi)
            else:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                improved = True
        current = new_current
    
    return current

def _two_exchange(blocks_idx, block_masks, all_k_blocks, n_k_blocks, n_t_subs, deadline):
    """Try 2-exchanges: remove one block, try to add one different block that covers more."""
    if not blocks_idx:
        return blocks_idx
    
    current = list(blocks_idx)
    
    # Build coverage mask for current set
    coverage = 0
    for bi in current:
        coverage |= block_masks[bi]
    
    # Build cover count
    cover_count = [0] * n_t_subs
    for bi in current:
        m = block_masks[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        n = len(current)
        for i in range(n):
            if time.time() > deadline - 10:
                break
            bi = current[i]
            removed_mask = block_masks[bi]
            
            # Temporarily remove
            new_cover_count = cover_count[:]
            m = removed_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                new_cover_count[idx] -= 1
                m ^= lsb
            
            # Check if coverage is still complete
            new_coverage = 0
            for j in range(n):
                if j != i:
                    new_coverage |= block_masks[current[j]]
            
            # Find uncovered subsets
            uncovered = ~new_coverage & ((1 << n_t_subs) - 1)
            if uncovered == 0:
                # Can simply remove this block
                new_current = [current[j] for j in range(n) if j != i]
                current = new_current
                # Rebuild cover_count
                cover_count = [0] * n_t_subs
                for bi2 in current:
                    m = block_masks[bi2]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        m ^= lsb
                improved = True
                break
            else:
                # Try to find a block that covers the most uncovered subsets
                best_gain = 0
                best_block = -1
                for bi2 in range(n_k_blocks):
                    if bi2 in current:
                        continue
                    gain = (block_masks[bi2] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = bi2
                    if best_gain >= uncovered.bit_count():
                        break
                
                if best_block != -1 and best_gain > 0:
                    # Check: after adding best_block, is everything covered?
                    # New coverage = new_coverage | block_masks[best_block]
                    # We need new_coverage | block_masks[best_block] == full
                    # Equivalently: (uncovered & ~block_masks[best_block]) == 0
                    if (uncovered & ~block_masks[best_block]) == 0:
                        # Replace bi with best_block
                        new_current = list(current)
                        new_current[i] = best_block
                        current = new_current
                        # Rebuild cover_count
                        cover_count = [0] * n_t_subs
                        for bi2 in current:
                            m = block_masks[bi2]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_count[idx] += 1
                                m ^= lsb
                        improved = True
                        break
    
    return current

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
    
    full_mask = (1 << n_t_subs) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different greedy orderings
    max_restarts = 5
    restart_time = (deadline - time.time() - 30) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30:
            break
        
        # Greedy construction with random tie-breaking
        uncovered_mask = full_mask
        selected = []
        
        while uncovered_mask != 0:
            if time.time() > deadline - 30:
                break
            
            best_idx = -1
            best_gain = -1
            best_gain_list = []
            
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_gain_list = [i]
                elif gain == best_gain:
                    best_gain_list.append(i)
            
            if best_idx == -1 or best_gain == 0:
                break
            
            # Random tie-breaking for diversity
            if len(best_gain_list) > 1:
                best_idx = random.choice(best_gain_list)
            
            selected.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        
        # Local search: redundancy removal
        selected = _remove_redundant(selected, block_masks, n_t_subs, deadline)
        
        # Local search: 2-exchange
        selected = _two_exchange(selected, block_masks, all_k_blocks, n_k_blocks, n_t_subs, deadline)
        
        # Verify coverage
        cov = 0
        for bi in selected:
            cov |= block_masks[bi]
        if cov == full_mask and len(selected) < best_count:
            best_count = len(selected)
            best_blocks = list(selected)
        
        if time.time() > deadline - 15:
            break
    
    if best_blocks is None:
        # Fallback: greedy without restarts
        uncovered_mask = full_mask
        selected = []
        while uncovered_mask != 0:
            if time.time() > deadline - 5:
                break
            best_idx = -1
            best_gain = -1
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        selected = _remove_redundant(selected, block_masks, n_t_subs, deadline)
        best_blocks = selected
    
    return [list(all_k_blocks[i]) for i in best_blocks]
