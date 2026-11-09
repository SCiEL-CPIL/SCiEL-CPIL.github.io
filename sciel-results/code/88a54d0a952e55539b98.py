"""Improved covering design C(19,7,4) using precomputed bitmasks and randomized greedy."""
import itertools
import random
import time

# Precomputation for v=19, k=7, t=4
V, K, T = 19, 7, 4

def _precompute_masks():
    """Precompute bitmasks for all k-subsets."""
    points = range(V)
    t_subsets = list(itertools.combinations(points, T))
    num_t = len(t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(points, K))
    num_k = len(k_subsets)
    
    masks = [0] * num_k
    # For each k-subset, set bits for all t-subsets it contains
    for i, k_sub in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(k_sub, T):
            idx = t_to_idx[t_sub]
            mask |= (1 << idx)
        masks[i] = mask
        
    return k_subsets, masks, num_t

# Global precomputation
_K_SUBSETS, _MASKS, _NUM_T = _precompute_masks()
_TOTAL_MASK = (1 << _NUM_T) - 1
_NUM_K = len(_K_SUBSETS)

def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy with bitmasks."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if v != V or k != K or t != T:
        # Fallback for other parameters
        return _simple_greedy(v, k, t, deadline)

    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple randomized greedy instances
    run = 0
    while True:
        if time.time() > deadline - 5.0:
            break
        
        covered_mask = 0
        blocks = []
        
        while covered_mask != _TOTAL_MASK:
            if time.time() > deadline - 2.0:
                break
            
            uncovered = _TOTAL_MASK & ~covered_mask
            
            max_gain = -1
            best_indices = []
            
            # Iterate all blocks to find max gain
            for i in range(_NUM_K):
                gain = (_MASKS[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_indices.clear()
                    best_indices.append(i)
                elif gain == max_gain:
                    best_indices.append(i)
            
            if max_gain == 0:
                # Stuck, break to avoid infinite loop
                break
            
            # Randomized selection from best candidates
            chosen_idx = rng.choice(best_indices)
            
            block = _K_SUBSETS[chosen_idx]
            blocks.append(block)
            covered_mask |= _MASKS[chosen_idx]
        
        if covered_mask == _TOTAL_MASK:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
        
        run += 1
    
    if best_blocks is None:
        # Fallback: deterministic greedy to ensure validity
        covered_mask = 0
        blocks = []
        while covered_mask != _TOTAL_MASK:
            uncovered = _TOTAL_MASK & ~covered_mask
            max_gain = -1
            best_idx = 0
            for i in range(_NUM_K):
                gain = (_MASKS[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            blocks.append(_K_SUBSETS[best_idx])
            covered_mask |= _MASKS[best_idx]
        best_blocks = blocks
        best_count = len(blocks)

    return best_blocks

def _simple_greedy(v, k, t, deadline):
    """Fallback greedy for non-standard parameters."""
    start_time = time.time()
    points = list(range(v))
    t_subsets = list(itertools.combinations(points, t))
    covered = set()
    blocks = []
    
    while len(covered) < len(t_subsets):
        if time.time() > deadline:
            break
        best_block = None
        max_new = 0
        for block in itertools.combinations(points, k):
            block_set = set(block)
            new_covered = 0
            for ts in itertools.combinations(block, t):
                if ts not in covered:
                    new_covered += 1
            if new_covered > max_new:
                max_new = new_covered
                best_block = block
        if best_block is None or max_new == 0:
            break
        blocks.append(list(best_block))
        for ts in itertools.combinations(best_block, t):
            covered.add(ts)
            
    return blocks
