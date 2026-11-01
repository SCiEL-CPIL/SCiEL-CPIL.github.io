import itertools
import random
import time
import sys

# Precomputation for specific v, k, t is not possible at import time if v,k,t vary,
# but the task says "evaluator passes this task's parameters v, k and t".
# However, C(19,8,5) is the specific instance.
# We will assume v=19, k=8, t=5 for optimization, but write generally enough or 
# just hardcode the logic for this specific case if it's safe.
# The prompt says "The evaluator passes this task's parameters v, k and t".
# It implies construct(v, k, t) is called.
# We will do the precomputation inside construct if not cached, but since 
# v,k,t are fixed for this task instance, we can cache it globally.

_cache = {}

def _precompute(v, k, t):
    if (v, k, t) in _cache:
        return _cache[(v, k, t)]
    
    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    t_sub_map = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    all_k_blocks = list(itertools.combinations(points, k))
    n_k = len(all_k_blocks)
    
    # block_to_subs: for each block, list of indices of t-subsets it covers
    block_to_subs = []
    # sub_to_blocks: for each t-subset, list of indices of blocks that cover it
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_k_blocks):
        subs_in_block = []
        for t_sub in itertools.combinations(block, t):
            s_idx = t_sub_map[t_sub]
            subs_in_block.append(s_idx)
            sub_to_blocks[s_idx].append(b_idx)
        block_to_subs.append(subs_in_block)
        
    _cache[(v, k, t)] = (all_k_blocks, block_to_subs, sub_to_blocks, n_t, n_k)
    return _cache[(v, k, t)]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    all_k_blocks, block_to_subs, sub_to_blocks, n_t, n_k = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # We will run multiple greedy attempts with different random seeds for tie-breaking
    # The greedy algorithm:
    # 1. gain[b] = number of uncovered t-subsets in block b
    # 2. uncovered[s] = 1 if t-subset s is not covered by any selected block, 0 otherwise
    # 3. Select block with max gain. Tie-break randomly.
    # 4. Update gains for all blocks sharing uncovered t-subsets with the selected block.
    
    # Initial state
    # gain[b] is initially binom(k, t)
    initial_gain = len(block_to_subs[0]) if n_k > 0 else 0
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts) # Deterministic but varied per attempt
        
        uncovered = [1] * n_t
        gain = [initial_gain] * n_k
        selected = []
        remaining_uncovered = n_t
        
        # Optimization: Keep track of blocks with max gain? 
        # Just scanning max is O(n_k). n_k ~ 75k. 300 steps * 75k = 22.5M ops.
        # This is fine.
        
        while remaining_uncovered > 0:
            # Find max gain
            max_gain = 0
            max_blocks = []
            # To avoid O(n_k) scan if we can, but we need to update gains anyway.
            # The update is sparse, but finding max is dense.
            # We can use a heap, but updates are frequent.
            # Let's just scan. It's fast enough.
            
            for b in range(n_k):
                if gain[b] > max_gain:
                    max_gain = gain[b]
                    max_blocks = [b]
                elif gain[b] == max_gain:
                    max_blocks.append(b)
            
            if max_gain == 0:
                # Should not happen if we are covering everything
                break
            
            # Pick randomly from max_blocks to break ties
            b_sel = random.choice(max_blocks)
            selected.append(all_k_blocks[b_sel])
            
            # Update
            for s in block_to_subs[b_sel]:
                if uncovered[s]:
                    uncovered[s] = 0
                    remaining_uncovered -= 1
                    # Update gain for all blocks containing s
                    for b_prime in sub_to_blocks[s]:
                        gain[b_prime] -= 1
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = selected
            
        # If we found a very good solution, we might stop early or continue to try for better
        # The lower bound is 264. If we get close, keep trying.
        
    if best_blocks is None:
        # Fallback: return empty or a trivial invalid result? 
        # Must return valid. 
        # If time ran out before first completion, we might have a partial.
        # But the loop above completes one full greedy run in < 1 second.
        # So best_blocks should be set.
        return []
        
    return best_blocks
