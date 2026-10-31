"""Rewritten program for covering design C(v,k,t) with a stronger greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t): a set of k-subsets (blocks) of {0..v-1}
    such that every t-subset is contained in at least one block.
    Minimize the number of blocks.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # 1. Precompute all t-subsets and assign an index to each
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # 2. Precompute the bitmask for every possible k-subset (block)
    # A bit i is set if the t-subset at index i is contained in the block.
    all_k_subsets = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in all_k_subsets:
        mask = 0
        # A block covers a t-subset if the t-subset is a subset of the block.
        # Check only the t-subsets formed by the block's elements.
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Helper to count bits safely
    def popcount(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    def run_greedy(seed):
        rng = random.Random(seed)
        unc = (1 << n_t) - 1
        sel = []
        sel_idx_set = set()
        
        while unc:
            if time.time() > deadline:
                return None
            
            best_gain = -1
            best_indices = []
            
            for i, mask in enumerate(block_masks):
                if i in sel_idx_set:
                    continue
                x = unc & mask
                if x == 0:
                    continue
                gain = popcount(x)
                
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain and gain > 0:
                    best_indices.append(i)
            
            if not best_indices:
                break
            
            chosen_idx = rng.choice(best_indices)
            
            sel.append(all_k_subsets[chosen_idx])
            sel_idx_set.add(chosen_idx)
            unc &= ~block_masks[chosen_idx]
            
            if time.time() > deadline - 5.0:
                return None
        
        return sel

    best_blocks = None
    best_count = float('inf')
    
    # First run: deterministic (seed=0)
    res = run_greedy(0)
    if res:
        best_blocks = res
        best_count = len(res)
    
    # Try a few more seeds if time allows
    for seed in range(1, 20):
        if time.time() > deadline - 10.0:
            break
        res = run_greedy(seed)
        if res and len(res) < best_count:
            best_blocks = res
            best_count = len(res)
            
    if best_blocks is None:
        # Fallback: should not happen if first run succeeds
        return []

    return best_blocks
# EVOLVE-BLOCK-END
