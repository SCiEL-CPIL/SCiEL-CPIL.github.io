"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets as bitmasks
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    block_to_mask = [0] * num_blocks
    for bi, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_to_mask[bi] = mask
    
    full_uncovered_mask = (1 << num_t) - 1
    
    # Precompute frequency of each t-subset across all blocks (for rarity-based greedy)
    # freq[i] = number of blocks covering t-subset i
    freq = [0] * num_t
    for bi in range(num_blocks):
        m = block_to_mask[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            freq[idx] += 1
            m ^= lsb
    
    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_uncovered_mask
        blocks = []
        
        while uncovered:
            best_block_idx = -1
            best_score = -1
            
            for bi in range(num_blocks):
                gain = (uncovered & block_to_mask[bi]).bit_count()
                if gain == 0:
                    continue
                
                # Score: primary is gain, secondary is rarity (lower freq = more valuable)
                # Use a combined score that rewards covering rare subsets
                # For each t-subset in the gain, add 1/freq as bonus
                score = gain * 1.0
                
                # Add rarity bonus: for subsets with low freq, they're harder to cover
                # This encourages choosing blocks that cover rare subsets
                m = uncovered & block_to_mask[bi]
                rarity_bonus = 0.0
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    rarity_bonus += 1.0 / max(freq[idx], 1)
                    m ^= lsb
                
                score += rarity_bonus * 0.1
                
                # Add random noise for diversification
                if seed is not None:
                    score += random.random() * 0.05
                
                if score > best_score:
                    best_score = score
                    best_block_idx = bi
            
            if best_block_idx == -1:
                break
                
            blocks.append(best_block_idx)
            uncovered &= ~block_to_mask[best_block_idx]
            
            if time.time() > deadline:
                return None
                
        return blocks

    def prune_blocks(block_indices):
        """Remove redundant blocks using iterative essential-block removal."""
        if not block_indices:
            return block_indices
        
        current = list(block_indices)
        
        while time.time() < deadline:
            # Count how many blocks cover each t-subset
            count_cover = [0] * num_t
            for bi in current:
                m = block_to_mask[bi]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    count_cover[idx] += 1
                    m ^= lsb
            
            # Identify essential blocks (those that are the only cover for some t-subset)
            essential = [False] * len(current)
            for i, bi in enumerate(current):
                m = block_to_mask[bi]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if count_cover[idx] == 1:
                        essential[i] = True
                        break
                    m ^= lsb
            
            old_len = len(current)
            current = [current[i] for i in range(old_len) if essential[i]]
            
            if len(current) < old_len:
                continue
            else:
                break
        
        return current

    def local_search(block_indices, max_swaps=50):
        """Try to improve by swapping blocks or removing redundant ones."""
        if not block_indices:
            return block_indices
        
        current = list(block_indices)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # First, try to remove any block whose coverage is fully redundant
            for i in range(len(current)):
                bi = current[i]
                # Check if all t-subsets covered by bi are also covered by other blocks
                m = block_to_mask[bi]
                redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    # Check if any other block covers this t-subset
                    covered_by_other = False
                    for j in range(len(current)):
                        if j == i:
                            continue
                        if block_to_mask[current[j]] & lsb:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    current.pop(i)
                    improved = True
                    break
            
            if time.time() > deadline:
                break
        
        return current

    # Run multiple restarts with different strategies
    best_blocks = None
    best_count = float('inf')
    
    # Strategy 1: Pure greedy with rarity bonus
    for seed in range(20):
        if time.time() > deadline:
            break
        blocks = greedy_cover(seed=seed)
        if blocks is None:
            continue
        blocks = prune_blocks(blocks)
        blocks = local_search(blocks)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback
        blocks = greedy_cover(seed=0)
        if blocks:
            return [all_blocks[bi] for bi in blocks]
        return []
    
    return [all_blocks[bi] for bi in best_blocks]

# EVOLVE-BLOCK-END
