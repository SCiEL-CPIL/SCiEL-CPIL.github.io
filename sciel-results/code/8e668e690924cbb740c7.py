"""
Randomized Rounding + Local Search for covering design C(24,7,3).

Strategy:
1. Generate a pool of diverse candidate blocks (Greedy + Random).
2. Use Randomized Rounding: assign a probability to each candidate block based on 
   a relaxed set cover solution (or simply heuristic weights), then sample a subset.
3. Repair the sampled subset to ensure validity (add missing blocks greedily).
4. Local Search: Iteratively try to remove blocks or swap blocks to reduce count 
   while maintaining validity.
"""
import itertools
import random
import time
import sys
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_tsets(v, t):
    """Precompute all t-sets and map them to indices."""
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    return all_tsets, tset_to_idx

def _block_to_tset_indices(block, tset_to_idx, t):
    """Get the indices of t-sets covered by a block."""
    return [tset_to_idx[ts] for ts in itertools.combinations(block, t)]

def _generate_candidate_pool(v, k, t, rng, num_candidates=2000, time_limit=5.0):
    """
    Generate a diverse pool of candidate blocks.
    Mix of greedy blocks and random blocks.
    """
    deadline = time.time() + time_limit
    candidates = []
    seen = set()
    
    # 1. Add some greedy blocks (high quality)
    # Simple greedy: pick block covering most uncovered t-sets
    uncovered = np.ones(C(v, t), dtype=np.int32) # Placeholder size, will use actual count
    # Actually, let's just generate a mix.
    
    # Generate random blocks
    while len(candidates) < num_candidates and time.time() < deadline:
        block = tuple(sorted(rng.sample(range(v), k)))
        if block not in seen:
            seen.add(block)
            candidates.append(block)
            
    # Add some "structured" blocks? 
    # For now, random is sufficient for the pool, but let's add a few greedy ones.
    # We'll do a quick greedy pass to find good blocks and add them to the pool.
    uncovered_mask = np.ones(C(v, t), dtype=np.int32)
    # This is getting complex to maintain in pool generation. 
    # Let's stick to random + a few specific patterns if needed.
    # Actually, a pool of 2000 random blocks out of 346k is sparse.
    # Let's ensure we have a valid covering in the pool first? No, the pool is for selection.
    
    return candidates

def _greedy_repair(selected_blocks, v, k, t, tset_to_idx, all_tsets, counts, rng, time_limit=10.0):
    """
    Given a set of selected blocks (and their counts), add more blocks greedily 
    to cover all t-sets.
    """
    deadline = time.time() + time_limit
    blocks = list(selected_blocks)
    
    # Initialize counts from selected blocks
    # (Assume counts is already updated for selected_blocks)
    
    while time.time() < deadline:
        # Find uncovered t-sets
        uncovered_indices = np.where(counts == 0)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Pick a random uncovered t-set to focus on? 
        # Or pick the one with fewest candidate blocks covering it?
        # Let's pick the first uncovered one and find a block that covers it.
        # To be faster, we can precompute which blocks cover which t-sets.
        # But we don't have that index here easily.
        
        # Alternative: Greedy set cover step.
        # Find block that covers most uncovered t-sets.
        # This is expensive if we scan all possible C(24,7) blocks.
        # Instead, we can sample candidate blocks.
        
        # Let's use a simpler repair: 
        # Pick a random uncovered t-set.
        # Generate a random block containing it.
        # Add it.
        
        target_tset = tuple(all_tsets[rng.choice(uncovered_indices)])
        
        # Generate a block containing target_tset
        remaining_points = [x for x in range(v) if x not in target_tset]
        needed = k - t
        if needed > 0:
            extra = rng.sample(remaining_points, needed)
        else:
            extra = []
            
        new_block = tuple(sorted(list(target_tset) + extra))
        
        # Add to blocks
        blocks.append(new_block)
        
        # Update counts
        for ts in itertools.combinations(new_block, t):
            idx = tset_to_idx[ts]
            counts[idx] += 1
            
    return blocks

def _local_search(blocks, v, k, t, tset_to_idx, all_tsets, counts, rng, time_limit=60.0):
    """
    Local search to minimize number of blocks.
    Moves:
    1. Remove a block if it doesn't break validity.
    2. Swap a block with another if it allows removal of another block later? 
       (Too complex). 
    3. Try to remove a block, if it fails, try to find a replacement that covers 
       the critical sets and allows removal of another block?
    
    Let's stick to:
    - Try to remove random blocks.
    - If removal fails, try to "fix" by adding a block that covers the critical sets, 
      then try to remove the original block AND another block?
      
    Simpler: 
    - Iterate through blocks. 
    - If block is "redundant" (all its t-sets covered by others), remove it.
    - If not redundant, try to replace it with a "better" block that covers its critical sets 
      and maybe allows other blocks to become redundant?
      
    For now, let's do:
    1. Pass 1: Remove all redundant blocks.
    2. Pass 2: For each block, try to replace it with a random block that covers its critical sets.
       If the new block makes another block redundant, accept.
    """
    deadline = time.time() + time_limit
    current_blocks = [list(b) for b in blocks]
    # Rebuild counts to be safe
    counts[:] = 0
    for b in current_blocks:
        for ts in itertools.combinations(b, t):
            counts[tset_to_idx[ts]] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current_blocks)
        
        # Pass 1: Remove redundant blocks
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            block = current_blocks[i]
            # Check if all t-sets in block are covered by other blocks
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = tset_to_idx[ts]
                if counts[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove
                for ts in itertools.combinations(block, t):
                    counts[tset_to_idx[ts]] -= 1
                current_blocks.pop(i)
                improved = True
                
        if not improved:
            # Pass 2: Try swaps
            # For each block, find its critical t-sets (count == 1)
            # Try to find a replacement block that covers them.
            # If the replacement block is different, swap it.
            # Then re-run Pass 1.
            
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                
                block = current_blocks[i]
                critical_tsets = []
                for ts in itertools.combinations(block, t):
                    idx = tset_to_idx[ts]
                    if counts[idx] == 1:
                        critical_tsets.append(ts)
                        
                if not critical_tsets:
                    continue # Should have been removed in Pass 1
                
                # Union of critical t-sets
                union_set = set()
                for ts in critical_tsets:
                    union_set.update(ts)
                    
                if len(union_set) > k:
                    continue # Cannot cover in one block
                
                # Try to construct a replacement block
                # It must contain union_set.
                # Add random elements to fill up to k.
                base = list(union_set)
                remaining_slots = k - len(base)
                candidates = [x for x in range(v) if x not in base]
                
                # Try a few random completions
                for _ in range(10):
                    if remaining_slots > 0:
                        extra = rng.sample(candidates, remaining_slots)
                    else:
                        extra = []
                    
                    new_block = tuple(sorted(base + extra))
                    
                    # Check if new_block is already in current_blocks
                    if list(new_block) in current_blocks:
                        continue
                    
                    # Simulate swap
                    # Remove old block
                    for ts in itertools.combinations(block, t):
                        counts[tset_to_idx[ts]] -= 1
                        
                    # Add new block
                    for ts in itertools.combinations(new_block, t):
                        counts[tset_to_idx[ts]] += 1
                        
                    # Check if any t-set became uncovered (should not if union_set is covered)
                    # But we must ensure new_block covers all critical t-sets.
                    # It does by construction.
                    # However, did we break anything else?
                    # No, we only decreased counts for old block's t-sets.
                    # If a t-set was covered ONLY by old block, it is in critical_tsets, 
                    # and thus in union_set, and thus in new_block. So it's still covered.
                    
                    # Accept the swap? 
                    # We accept if it leads to a reduction in block count in the next Pass 1.
                    # Or just accept it to change the landscape.
                    # Let's accept it and set improved = True to trigger Pass 1 again.
                    
                    current_blocks[i] = list(new_block)
                    improved = True
                    break # Try next block
                    
    return current_blocks

def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0 # Safety margin
    
    # Precompute
    all_tsets, tset_to_idx = _precompute_tsets(v, t)
    num_tsets = len(all_tsets)
    counts = np.zeros(num_tsets, dtype=np.int32)
    
    rng = random.Random(42)
    
    # Phase 1: Randomized Rounding / Sampling
    # We don't have a linear programming solver, so we use a heuristic "weighting".
    # Let's just sample a set of blocks.
    # How many? 
    # Start with a greedy covering to get a baseline size.
    
    # Quick Greedy Initialization
    deadline_init = start_time + 5.0
    uncovered = np.ones(num_tsets, dtype=np.int32)
    blocks = []
    
    while time.time() < deadline_init:
        # Find block that covers most uncovered t-sets
        # To speed this up, we can sample candidates.
        best_block = None
        best_gain = -1
        
        # Sample 100 random blocks
        for _ in range(100):
            block = tuple(sorted(rng.sample(range(v), k)))
            gain = 0
            for ts in itertools.combinations(block, t):
                idx = tset_to_idx[ts]
                if uncovered[idx] > 0:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is None or best_gain == 0:
            break
            
        blocks.append(list(best_block))
        for ts in itertools.combinations(best_block, t):
            idx = tset_to_idx[ts]
            uncovered[idx] -= 1
            
        if np.all(uncovered == 0):
            break
            
    # Ensure validity
    # If not valid, repair
    counts[:] = 0
    for b in blocks:
        for ts in itertools.combinations(b, t):
            counts[tset_to_idx[ts]] += 1
            
    if np.any(counts == 0):
        blocks = _greedy_repair(blocks, v, k, t, tset_to_idx, all_tsets, counts, rng, time_limit=10.0)
        # Rebuild counts
        counts[:] = 0
        for b in blocks:
            for ts in itertools.combinations(b, t):
                counts[tset_to_idx[ts]] += 1
                
    # Phase 2: Local Search
    deadline_ls = start_time + total_budget - 5.0
    blocks = _local_search(blocks, v, k, t, tset_to_idx, all_tsets, counts, rng, time_limit=deadline_ls - time.time())
    
    # Final Verification
    counts[:] = 0
    for b in blocks:
        for ts in itertools.combinations(b, t):
            counts[tset_to_idx[ts]] += 1
            
    if np.any(counts == 0):
        # Fallback to a known valid covering? 
        # We should have a valid one from Phase 1.
        # If we broke it, we need to repair again.
        blocks = _greedy_repair(blocks, v, k, t, tset_to_idx, all_tsets, counts, rng, time_limit=5.0)
        
    # Remove duplicates
    unique_blocks = []
    seen = set()
    for b in blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            unique_blocks.append(b)
            
    return unique_blocks

# EVOLVE-BLOCK-END
