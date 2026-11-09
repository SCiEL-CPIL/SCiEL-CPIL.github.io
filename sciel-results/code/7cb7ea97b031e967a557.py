import itertools
import random
import time
import sys
from collections import defaultdict

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # block_masks[i] is a bitmask of triples covered by block i
    block_masks = [0] * n_blocks
    # blocks_containing[j] is a list of block indices containing triple j
    blocks_containing = [[] for _ in range(n_triples)]
    # block_triples[i] is a list of triple indices in block i (for fast iteration)
    block_triples = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        tris = []
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
            blocks_containing[idx].append(i)
            tris.append(idx)
        block_masks[i] = mask
        block_triples[i] = tris
        
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, block_triples, seed):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    used_mask = 0
    active_blocks = set(range(n_blocks))
    
    while uncovered:
        best_block_idx = -1
        max_gain = 0
        best_gain_ties = []
        
        # Iterate through active blocks
        # To allow for diversity, we can randomize the order of checking or handle ties
        # However, simple max is robust. To improve diversity, we can pick randomly among top-k gains?
        # For now, let's stick to max but maybe add a small random weight?
        # No, greedy is deterministic. Let's just do max.
        
        # Optimization: Pre-compute gains? No, uncovered changes.
        # We can iterate.
        
        # If active_blocks is large, this is slow.
        # But n_blocks is ~300k. Iterating 300k * 76 steps is ~22M operations.
        # Each operation is bit_count of 2024-bit int. This is fast in Python?
        # Python int bit_count is C-speed.
        # 22M * ~10ns = 0.2s? Maybe slower due to Python loop overhead.
        # Let's optimize the inner loop.
        
        # Alternative: Maintain a list of active blocks and update it?
        # Or just iterate over range(n_blocks) and check if in active?
        # active_blocks is a set. `idx in active_blocks` is O(1).
        
        # Let's try to speed up by not using a set for active, but just a boolean list?
        # Or just keep active_blocks as a list and remove? Removal from list is O(N).
        # Set is better.
        
        for idx in active_blocks:
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block_idx = idx
                best_gain_ties = [idx]
            elif gain == max_gain:
                best_gain_ties.append(idx)
            
            # Early exit if we cover all remaining
            if gain == uncovered.bit_count():
                # This is the best possible gain.
                # We can break, but we might have missed a tie?
                # If we break, we pick the first one.
                # To be random, we can collect all with max gain?
                # But if gain == total, it's unique maximum possible.
                # So we can break.
                break
        
        if best_block_idx == -1:
            break
            
        # If we didn't break early, we might have ties.
        # Pick randomly from ties to introduce diversity?
        # The current code just picks the first max.
        # Let's pick randomly from best_gain_ties if we didn't break early.
        # But if we broke early, best_gain_ties might be incomplete.
        # Let's just pick best_block_idx.
        
        blocks_used.append(best_block_idx)
        used_mask |= block_masks[best_block_idx]
        uncovered &= ~used_mask
        active_blocks.remove(best_block_idx)
        
    return blocks_used

def _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, deadline):
    rng = random.Random(seed)
    
    # State
    selected = set(initial_blocks)
    coverage_count = [0] * n_triples
    
    # Initialize counts
    for b in selected:
        for tri_idx in block_triples[b]:
            coverage_count[tri_idx] += 1
            
    # Helper to check if valid
    def is_valid():
        for c in coverage_count:
            if c == 0:
                return False
        return True

    if not is_valid():
        return list(initial_blocks)

    best_size = len(selected)
    best_selected = set(selected)
    
    # Pre-compute some stats?
    # Let's run the local search loop
    
    while time.time() < deadline:
        # 1. Try to remove a redundant block
        # Find a block that can be removed safely
        # A block b is redundant if for all tri in b, coverage_count[tri] > 1
        
        # To speed this up, we can maintain a count of "critical" blocks?
        # Or just iterate.
        
        # Iterate through selected blocks
        # To make it random, shuffle a copy?
        # Or pick a random index?
        
        removed = False
        items = list(selected)
        rng.shuffle(items)
        
        for b in items:
            redundant = True
            for tri_idx in block_triples[b]:
                if coverage_count[tri_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove b
                for tri_idx in block_triples[b]:
                    coverage_count[tri_idx] -= 1
                selected.remove(b)
                removed = True
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                break # Restart search
        
        if not removed:
            # 2. Try to swap
            # Find a block b in selected with unique triples
            # And a block b' not in selected that covers those unique triples
            
            # Pick a random block from selected
            b = rng.choice(list(selected))
            
            # Find unique triples for b
            unique_tris = []
            for tri_idx in block_triples[b]:
                if coverage_count[tri_idx] == 1:
                    unique_tris.append(tri_idx)
            
            if not unique_tris:
                # Should not happen if removal failed, but maybe due to race?
                # Or if we didn't check all?
                # If no unique tris, b is redundant.
                # Try removing again?
                # If we can't remove, it's not redundant.
                continue
            
            # We need a b' that covers all unique_tris
            # Pick one unique tri to find candidates
            # To maximize chance, pick the one with fewest candidates?
            # Or just the first one?
            # Let's pick the one with the fewest blocks containing it, to constrain search.
            
            min_candidates = None
            best_tri = unique_tris[0]
            for tri in unique_tris:
                cands = blocks_containing[tri]
                if min_candidates is None or len(cands) < len(min_candidates):
                    min_candidates = cands
                    best_tri = tri
            
            # Shuffle candidates
            cands = list(min_candidates)
            rng.shuffle(cands)
            
            swap_found = False
            for b_prime in cands:
                if b_prime in selected:
                    continue
                
                # Check if b_prime covers all unique_tris
                covers_all = True
                # We can check if unique_tris is a subset of block_triples[b_prime]
                # Since unique_tris is small, we can just check each
                # Or use set intersection?
                # Set of block_triples[b_prime] is not precomputed as a set.
                # Let's just check each tri in unique_tris
                # Is tri in block_triples[b_prime]?
                # block_triples[b_prime] is a list.
                # To make this faster, we could precompute a set for each block?
                # n_blocks * n_triples_per_block memory.
                # 300k * 35 = 10M ints. ~80MB. Acceptable.
                # But we didn't precompute it in _precompute.
                # Let's just do a linear scan or set conversion on the fly?
                # Linear scan of 35 items is fast.
                
                # Optimization: Create a set of block_triples[b_prime]?
                # Or just check membership.
                # `tri in block_triples[b_prime]` is O(k).
                
                for tri in unique_tris:
                    if tri not in block_triples[b_prime]:
                        covers_all = False
                        break
                
                if covers_all:
                    # Perform Swap
                    # Remove b
                    for tri_idx in block_triples[b]:
                        coverage_count[tri_idx] -= 1
                    selected.remove(b)
                    
                    # Add b_prime
                    for tri_idx in block_triples[b_prime]:
                        coverage_count[tri_idx] += 1
                    selected.add(b_prime)
                    
                    swap_found = True
                    break
            
            if not swap_found:
                # Try a different block?
                # The current loop picks one random block b.
                # If it fails, we loop back to the top.
                # This might be inefficient if many blocks fail.
                # We could iterate through all blocks and try to swap each?
                # But that's expensive.
                # Let's just rely on the random restart of the outer loop (picking new b).
                pass

    return list(best_selected)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110 # Safety margin for 120s limit
    
    # Precompute
    all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run greedy + local search multiple times
    # We need to manage time carefully.
    # Let's do a fixed number of iterations, or until deadline.
    # Each iteration: Greedy (fast) + Local Search (variable).
    
    # To ensure we don't get stuck in one local search for 100s,
    # we can limit the local search time per iteration.
    # E.g., 5 seconds per iteration.
    
    while time.time() < deadline - 10: # Leave 10s for final check
        # 1. Greedy
        initial_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, block_triples, seed)
        seed += 1
        
        if not initial_blocks:
            continue
            
        # 2. Local Search
        # Give it a sub-deadline
        sub_deadline = min(deadline - 10, time.time() + 5)
        
        blocks = _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, sub_deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # print(f"New best: {best_len} blocks at t={time.time()-start_time}")
            
    if best_blocks is None:
        # Fallback
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, 999)
        best_blocks = blocks
        
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
