"""Rewritten program for covering design C(v,k,t) with a stronger greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

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
    n_blocks = len(all_k_subsets)
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

    def run_greedy(seed, top_k=1):
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
            
            if top_k > 1 and len(best_indices) > top_k:
                # Randomly sample from the top candidates
                chosen_idx = rng.choice(best_indices[:top_k])
            else:
                chosen_idx = rng.choice(best_indices)
            
            sel.append(all_k_subsets[chosen_idx])
            sel_idx_set.add(chosen_idx)
            unc &= ~block_masks[chosen_idx]
            
            if time.time() > deadline - 5.0:
                return None
        
        return sel

    def local_search(blocks, seed, max_iters=50):
        """Try to remove redundant blocks or swap to improve coverage."""
        rng = random.Random(seed)
        current = list(blocks)
        n = len(current)
        
        # Compute coverage count for each t-subset
        # Using a dict to count how many blocks cover each t-subset
        coverage = {}
        for block in current:
            for ts in itertools.combinations(block, t):
                coverage[ts] = coverage.get(ts, 0) + 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 10.0:
            improved = False
            iters += 1
            n = len(current)
            
            # Try removing each block if it's redundant
            for i in range(n):
                if time.time() > deadline - 10.0:
                    return current
                
                block = current[i]
                # Check if all t-subsets in this block are covered by other blocks
                redundant = True
                for ts in itertools.combinations(block, t):
                    if coverage.get(ts, 0) <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove the block
                    for ts in itertools.combinations(block, t):
                        coverage[ts] -= 1
                    current.pop(i)
                    improved = True
                    break  # Restart from beginning after a removal
        
        return current

    best_blocks = None
    best_count = float('inf')
    
    # First run: deterministic (seed=0)
    res = run_greedy(0)
    if res:
        res = local_search(res, 0)
        best_blocks = res
        best_count = len(res)
    
    # Try more seeds with different top_k values if time allows
    seed_range = list(range(1, 30))
    rng_master = random.Random(42)
    rng_master.shuffle(seed_range)
    
    for seed in seed_range:
        if time.time() > deadline - 15.0:
            break
        
        # Vary top_k to explore different regions
        top_k = rng_master.choice([1, 2, 3, 5])
        res = run_greedy(seed, top_k=top_k)
        if res:
            res = local_search(res, seed + 1000)
            if len(res) < best_count:
                best_blocks = res
                best_count = len(res)
                
    if best_blocks is None:
        # Fallback: should not happen if first run succeeds
        return []

    return best_blocks
# EVOLVE-BLOCK-END
