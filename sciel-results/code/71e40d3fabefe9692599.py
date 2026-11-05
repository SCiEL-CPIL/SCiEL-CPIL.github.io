import itertools
import random
import time

# Global cache to avoid recomputation if construct is called multiple times with same params
_cache = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _cache:
        return _cache[key]
    
    # 1. Generate all t-subsets and map to IDs
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    triple_to_id = {ts: i for i, ts in enumerate(all_t_subsets)}
    id_to_triple = {i: ts for i, ts in enumerate(all_t_subsets)}
    
    # 2. Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # 3. For each block, compute the set of t-subset IDs it covers
    blocks_triples = []
    # Also build reverse index: for each t-subset ID, which blocks cover it?
    # This is useful if we want to pick a random uncovered t-subset and look up candidates.
    blocks_containing_t = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_subsets):
        covered_ids = set()
        for ts in itertools.combinations(block, t):
            tid = triple_to_id[ts]
            covered_ids.add(tid)
            blocks_containing_t[tid].append(b_idx)
        blocks_triples.append(covered_ids)
        
    data = {
        'all_k_subsets': all_k_subsets,
        'blocks_triples': blocks_triples,
        'blocks_containing_t': blocks_containing_t,
        'num_t_subsets': num_t_subsets,
        'num_blocks': num_blocks
    }
    _cache[key] = data
    return data

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    data = _precompute(v, k, t)
    all_k_subsets = data['all_k_subsets']
    blocks_triples = data['blocks_triples']
    blocks_containing_t = data['blocks_containing_t']
    num_t_subsets = data['num_t_subsets']
    
    best_blocks = None
    best_len = float('inf')
    
    # Number of restarts to try
    # We can estimate how long one run takes and adjust, but fixed 50-100 is safe for this size
    # Let's run as many as possible until deadline
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts) # Deterministic sequence of random choices for reproducibility in testing, but varied across attempts
        
        uncovered = set(range(num_t_subsets))
        chosen_block_indices = []
        
        while uncovered:
            # Pick a random uncovered t-subset to focus on
            # Using next(iter(uncovered)) is faster but deterministic per state.
            # random.choice requires list conversion or sampling from set.
            # Set sampling is O(1) in Python 3.7+? No, sets don't have random.sample directly without list.
            # We can maintain a list of uncovered IDs? That's expensive to update.
            # Alternative: Just pick the smallest ID in uncovered? That's deterministic.
            # To randomize, we can shuffle the list of uncovered IDs? No, too slow.
            
            # Efficient random pick from set:
            # random.choice(list(uncovered)) is O(N). N starts at 2024, shrinks.
            # 2024 is small. This is fine.
            target_tid = random.choice(list(uncovered))
            
            # Get all blocks that cover this target_tid
            candidates = blocks_containing_t[target_tid]
            
            best_gain = -1
            best_b_idx = -1
            
            for b_idx in candidates:
                # Calculate gain: how many uncovered triples does this block cover?
                # Intersection of block's covered set and current uncovered set
                gain = len(blocks_triples[b_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
                # Optional early break if gain is max possible (k choose t)? 
                # Max gain is 35. If we find 35, we can break.
                if best_gain == len(blocks_triples[b_idx]):
                    break
            
            if best_b_idx == -1:
                # Should not happen if uncovered is not empty and candidates exist
                break
                
            chosen_block_indices.append(best_b_idx)
            uncovered.difference_update(blocks_triples[best_b_idx])
            
        if len(chosen_block_indices) < best_len:
            best_len = len(chosen_block_indices)
            best_blocks = [all_k_subsets[i] for i in chosen_block_indices]
            
            # If we reached a very good solution (e.g., <= 70), we can stop early?
            # The lower bound is 69. If we find 70, 71, etc., it's good.
            # But we don't know the exact optimum. We just minimize.
            # Let's keep searching unless we hit the theoretical minimum or time runs out.
            
    # Fallback if no blocks found (should not happen)
    if best_blocks is None:
        # Return a trivial valid covering (all k-subsets) - this is huge and bad score, but valid
        # Or just the first few? No, must be valid.
        # The greedy loop always produces a valid covering.
        # So this branch is only if precompute failed or similar.
        return []
        
    # Convert tuples to lists as required by interface
    return [list(b) for b in best_blocks]
