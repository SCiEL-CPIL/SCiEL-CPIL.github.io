"""
Strategy: Simulated Annealing / Local Search over Valid Coverings
1. Generate a large, diverse pool of candidate blocks.
2. Precompute bitmask representations of triples covered by each block.
3. Initialize with a valid covering using a fast greedy algorithm.
4. Iteratively improve the covering by:
   a. Removing redundant blocks (if any triple is uniquely covered, block is critical; if no critical triples, block is removable).
   b. Swapping blocks: If no removal is possible, try to replace a block with another that covers the same 'critical' triples but potentially allows further reductions. This explores the space of same-size coverings to escape local minima.
5. Use vectorized numpy operations for fast coverage checks.
"""

import itertools
import random
import time
import numpy as np
from collections import defaultdict

# Precomputation for v=24, k=7, t=3
# Total triples: C(24, 3) = 2024
# We map each triple to an integer ID 0..2023

def _precompute_triple_ids(v, t):
    """Map each t-subset of 0..v-1 to a unique integer ID."""
    triple_to_id = {}
    id_to_triple = []
    idx = 0
    for comb in itertools.combinations(range(v), t):
        triple_to_id[comb] = idx
        id_to_triple.append(comb)
        idx += 1
    return triple_to_id, id_to_triple

def _get_block_mask(block, triple_to_id):
    """
    Return a set of triple IDs covered by the block.
    Using a set of ints is fast for intersection/difference in Python.
    Alternatively, a numpy array of bools, but sets of ints are often faster for small universes in pure Python loops if we avoid numpy overhead per operation.
    However, for 2024 items, a bitmask integer (Python int) is extremely fast for bitwise ops.
    Let's use Python int as a bitmask. 2024 bits fits in ~32 64-bit words.
    """
    mask = 0
    for triple in itertools.combinations(block, t):
        tid = triple_to_id[tuple(triple)]
        mask |= (1 << tid)
    return mask

def _generate_candidate_blocks(v, k, t, num_candidates, seed):
    """
    Generate a pool of candidate blocks.
    Mix of greedy and random to ensure diversity and coverage capability.
    """
    rng = random.Random(seed)
    candidates = []
    seen = set()
    
    # 1. Generate a baseline greedy cover to ensure we have a valid starting point
    # We'll use a simple greedy here for speed
    uncovered = set(itertools.combinations(range(v), t))
    baseline_blocks = []
    
    while uncovered:
        # Pick a random uncovered triple to start
        first_triple = tuple(rng.choice(list(uncovered)))
        block = list(first_triple)
        
        # Greedily fill the rest
        while len(block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for subset in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(subset + (x,)))
                    if new_triple in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in seen:
            seen.add(b_tuple)
            candidates.append(b_tuple)
            baseline_blocks.append(b_tuple)
            
        for triple in itertools.combinations(block, t):
            uncovered.discard(tuple(triple))
            
    # 2. Generate random variations to enrich the pool
    # We want blocks that are "good" in the sense that they cover many triples, 
    # but since all 7-subsets cover 35 triples, "goodness" is about *which* triples.
    # Random blocks are fine for diversity.
    while len(candidates) < num_candidates:
        block = tuple(sorted(rng.sample(range(v), k)))
        if block not in seen:
            seen.add(block)
            candidates.append(block)
            
    return candidates, baseline_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # 1. Precompute triple IDs
    triple_to_id, id_to_triple = _precompute_triple_ids(v, t)
    num_triples = len(id_to_triple)
    
    # 2. Generate Candidate Pool
    # We need a pool large enough to allow swaps.
    # 500-1000 blocks is a good trade-off between generation time and search space.
    num_candidates = 800
    candidates, baseline_blocks = _generate_candidate_blocks(v, k, t, num_candidates, seed=42)
    
    # 3. Precompute masks for all candidates
    # mask[i] is an integer bitmask of triples covered by candidates[i]
    masks = []
    for block in candidates:
        masks.append(_get_block_mask(block, triple_to_id))
        
    # Map block tuple to index in candidates list for quick lookup
    block_to_idx = {block: i for i, block in enumerate(candidates)}
    
    # 4. Initialize with a valid covering
    # Use the baseline blocks from generation (they form a valid cover by construction)
    # Convert to list of indices
    current_indices = []
    for b in baseline_blocks:
        idx = block_to_idx[b]
        current_indices.append(idx)
        
    # Ensure no duplicates in current_indices (shouldn't happen, but safe)
    current_indices = list(set(current_indices))
    
    # Verify initial validity
    current_cover_mask = 0
    for idx in current_indices:
        current_cover_mask |= masks[idx]
        
    full_mask = (1 << num_triples) - 1
    if current_cover_mask != full_mask:
        # Fallback: run a simple greedy set cover on the pool if baseline failed (unlikely)
        # For now, assume baseline is valid.
        pass
        
    best_count = len(current_indices)
    best_indices = list(current_indices)
    
    # 5. Local Search Loop
    # We maintain:
    # - current_indices: list of block indices in the current solution
    # - cover_counts: array of length num_triples, count of how many blocks cover each triple
    # - current_cover_mask: bitmask of covered triples (should always be full_mask)
    
    # Initialize cover_counts
    cover_counts = np.zeros(num_triples, dtype=np.int32)
    for idx in current_indices:
        # Update counts using the mask
        # Efficient way: iterate over set bits in mask
        m = masks[idx]
        while m:
            # Get index of lowest set bit
            lsb = m & -m
            tid = lsb.bit_length() - 1
            cover_counts[tid] += 1
            m ^= lsb
            
    # Helper to get set bits from mask
    def get_set_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            bits.append(lsb.bit_length() - 1)
            mask ^= lsb
        return bits

    # Precompute list of blocks not in current solution for swap search?
    # This changes as we swap. We'll just iterate over all candidates and check if in current.
    # To speed up, maintain a set of current indices.
    current_set = set(current_indices)
    
    # Track stagnation
    last_improvement_time = time.time()
    stagnation_counter = 0
    
    while time.time() < deadline - 5.0:
        # 1. Try to remove a block
        # A block is removable if all triples it covers have count > 1
        # i.e., for all tid in mask[idx], cover_counts[tid] > 1
        
        removed = False
        # Try to find a removable block
        # We can iterate through current_indices
        # To be smart, iterate through blocks with "fewer" unique triples first?
        # Or just random order to avoid bias.
        
        # Let's try to find ANY removable block
        removable_idx = -1
        for idx in current_indices:
            m = masks[idx]
            # Check if any triple in m has count == 1
            # If not, it's removable
            is_critical = False
            # Iterate set bits
            mm = m
            while mm:
                lsb = mm & -mm
                tid = lsb.bit_length() - 1
                if cover_counts[tid] == 1:
                    is_critical = True
                    break
                mm ^= lsb
            if not is_critical:
                removable_idx = idx
                break
                
        if removable_idx != -1:
            # Remove it
            m = masks[removable_idx]
            # Update counts
            mm = m
            while mm:
                lsb = mm & -mm
                tid = lsb.bit_length() - 1
                cover_counts[tid] -= 1
                mm ^= lsb
                
            current_indices.remove(removable_idx)
            current_set.remove(removable_idx)
            best_count = len(current_indices)
            best_indices = list(current_indices)
            last_improvement_time = time.time()
            stagnation_counter = 0
            removed = True
            continue # Restart loop as counts changed
            
        if not removed:
            # No block is removable. Try a Swap.
            # Pick a block to swap out.
            # Heuristic: Pick a block that is "most critical" (has many count==1 triples)
            # or "least redundant" (has few count>1 triples)?
            # Actually, we want to swap out a block that is "hard" to replace,
            # or a block that is "easy" to replace?
            # If we swap out a block with many unique triples, we need a replacement that covers all of them.
            # That's hard.
            # If we swap out a block with few unique triples, it's easier to find a replacement.
            # But swapping a "redundant" block might not help escape the minimum.
            
            # Let's pick a random block from current solution.
            if current_indices:
                swap_out_idx = random.choice(current_indices)
                m_out = masks[swap_out_idx]
                
                # Determine the triples that will become uncovered if we remove this block
                # These are the triples where cover_counts[tid] == 1
                critical_tids = []
                mm = m_out
                while mm:
                    lsb = mm & -mm
                    tid = lsb.bit_length() - 1
                    if cover_counts[tid] == 1:
                        critical_tids.append(tid)
                    mm ^= lsb
                    
                # If there are no critical tids, the block is removable (handled above).
                # So critical_tids is non-empty.
                
                # We need to find a block b' not in current_set such that
                # b' covers ALL critical_tids.
                # And ideally, b' covers as many other triples as possible to increase redundancy.
                
                # Optimization: Precompute for each candidate block, the set of critical tids it covers?
                # No, critical tids change.
                
                # Iterate through candidates not in current_set
                # To make this fast, we can precompute a "coverage profile" or just check intersection.
                
                # Condition: (m_in & m_out) must contain all bits in m_out that are critical.
                # Let M_crit be the bitmask of critical_tids.
                M_crit = 0
                for tid in critical_tids:
                    M_crit |= (1 << tid)
                    
                # We need m_in such that (m_in & M_crit) == M_crit
                # i.e., m_in covers all critical tids.
                
                found_swap = False
                # To avoid O(N) scan every time, we could index blocks by the triples they cover.
                # But for now, a simple scan of ~800 blocks is fast enough in Python if we use bit ops.
                
                # We also want to prefer swaps that increase the "margin" of coverage.
                # Score = number of new triples covered (redundancy added)
                
                best_swap_idx = -1
                best_score = -1
                
                for idx_in in range(len(candidates)):
                    if idx_in in current_set:
                        continue
                    
                    m_in = masks[idx_in]
                    
                    # Check if m_in covers all critical tids
                    if (m_in & M_crit) != M_crit:
                        continue
                    
                    # Calculate score: how many *other* triples does m_in cover?
                    # This increases redundancy for those triples, making them less likely to be critical.
                    # We want to maximize the minimum count or total redundancy.
                    # Simple heuristic: count bits in m_in that are NOT in m_out.
                    # These are new triples covered, increasing their count by 1.
                    new_triples = m_in & ~m_out
                    score = bin(new_triples).count('1')
                    
                    if score > best_score:
                        best_score = score
                        best_swap_idx = idx_in
                        
                if best_swap_idx != -1:
                    # Perform the swap
                    # Remove swap_out_idx
                    m_out = masks[swap_out_idx]
                    mm = m_out
                    while mm:
                        lsb = mm & -mm
                        tid = lsb.bit_length() - 1
                        cover_counts[tid] -= 1
                        mm ^= lsb
                        
                    # Add best_swap_idx
                    m_in = masks[best_swap_idx]
                    mm = m_in
                    while mm:
                        lsb = mm & -mm
                        tid = lsb.bit_length() - 1
                        cover_counts[tid] += 1
                        mm ^= lsb
                        
                    current_indices.remove(swap_out_idx)
                    current_set.remove(swap_out_idx)
                    current_indices.append(best_swap_idx)
                    current_set.add(best_swap_idx)
                    
                    # Accept swap even if count doesn't change, to escape local minimum
                    # But we should track if this leads to improvement later.
                    # For now, accept neutral swaps.
                    # To prevent infinite loops of neutral swaps, we can limit the number of neutral swaps
                    # or use a random restart if too many.
                    stagnation_counter += 1
                    
                    # If we've done too many neutral swaps without improvement, restart?
                    # Or just continue. The random choice of swap_out ensures exploration.
                    
                    # Check if this swap accidentally made a block removable?
                    # The loop restarts, so it will check.
                    
                    continue # Restart loop to check for removals
            
        # If we reach here, no move was made (shouldn't happen if current_indices is not empty)
        # If current_indices is empty, we are done (should not happen for covering)
        
    # Final check and return
    # Ensure we return the best valid covering found
    # The current state is always valid (we only do valid swaps/removals)
    
    # Convert indices back to blocks
    final_blocks = [list(candidates[i]) for i in best_indices]
    
    # Final Validation
    covered = set()
    for block in final_blocks:
        for triple in itertools.combinations(block, t):
            covered.add(tuple(triple))
            
    universe = set(itertools.combinations(range(v), t))
    if covered != universe:
        # Fallback to baseline
        final_blocks = [list(b) for b in baseline_blocks]
        
    return final_blocks

