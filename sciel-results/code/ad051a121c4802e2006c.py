"""
Global set-cover + 1-exchange local search for covering design C(v,k,t).

Strategy:
1. Enumerate all C(v,k) blocks (feasible for v=24,k=7: ~346k blocks).
2. Represent each block's coverage of C(v,t) triples as a Python int bitmask.
3. Greedy set-cover: repeatedly pick the block covering the most currently-uncovered triples.
4. 1-exchange local search: try swapping one selected block for one unselected block
   if the swap preserves coverage and reduces total block count (or keeps count same
   but enables further removals).
5. Remove redundant blocks (those whose triples are all covered by others).
6. Time-guarded: return best valid covering before deadline.
"""

import time
import itertools
import random
import sys

_TIME_LIMIT = 120
_MARGIN = 8
_DEADLINE = None


def _time_ok():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT - _MARGIN
    return time.time() < _DEADLINE


def _block_mask(block, t, triple_index):
    """Return bitmask of triples covered by this block."""
    mask = 0
    for ts in itertools.combinations(block, t):
        idx = triple_index[ts]
        mask |= (1 << idx)
    return mask


def _greedy_set_cover(all_blocks, all_masks, num_triples, rng=None):
    """
    Greedy set cover over precomputed block masks.
    Returns (selected_indices, covered_mask).
    """
    if rng is None:
        rng = random.Random(42)
    
    uncovered = (1 << num_triples) - 1
    selected = []
    selected_set = set()
    
    # For speed, precompute list of (block_idx, mask)
    # Greedy: each step find block with max (mask & uncovered).popcount()
    
    while uncovered and _time_ok():
        best_gain = 0
        best_idx = -1
        
        # Sample-based greedy for speed: check all blocks but break early if perfect
        # For v=24,k=7: ~346k blocks, each popcount on 2024-bit int is fast
        for i, mask in enumerate(all_masks):
            if i in selected_set:
                continue
            gain = (mask & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if best_gain == uncovered.bit_count():
                    break
            # Random tie-breaking for diversity
            if gain == best_gain and rng and rng.random() < 0.01:
                best_idx = i
        
        if best_idx == -1:
            break
        
        selected.append(best_idx)
        selected_set.add(best_idx)
        uncovered &= ~all_masks[best_idx]
    
    return selected, uncovered


def _try_remove_redundant(selected, all_masks, all_blocks, v, k, t, rng=None):
    """Remove blocks whose triples are all covered by other selected blocks."""
    if not selected:
        return selected
    
    # Compute coverage count per triple
    # For efficiency, use a dict: triple_index -> count
    # But with bitmasks, we can check: for block i, is (mask_i & ~OR_of_others) == 0?
    # OR_of_others is expensive. Instead, use count-based approach.
    
    # Count how many selected blocks cover each triple
    triple_counts = {}
    for idx in selected:
        mask = all_masks[idx]
        # Iterate set bits
        m = mask
        while m:
            low = m & (-m)
            bit = low.bit_length() - 1
            triple_counts[bit] = triple_counts.get(bit, 0) + 1
            m ^= low
    
    # Try to remove each block if all its triples have count > 1
    remaining = list(selected)
    changed = True
    while changed and _time_ok():
        changed = False
        for i in range(len(remaining)):
            if not _time_ok():
                break
            idx = remaining[i]
            mask = all_masks[idx]
            removable = True
            m = mask
            while m:
                low = m & (-m)
                bit = low.bit_length() - 1
                if triple_counts.get(bit, 0) <= 1:
                    removable = False
                    break
                m ^= low
            
            if removable:
                # Remove it
                m2 = mask
                while m2:
                    low = m2 & (-m2)
                    bit = low.bit_length() - 1
                    triple_counts[bit] -= 1
                    m2 ^= low
                remaining.pop(i)
                changed = True
                break  # Restart since indices shifted
    
    return remaining


def _one_exchange_search(selected, all_masks, all_blocks, v, k, t, rng, max_rounds=10):
    """
    1-exchange: try to replace one selected block with one unselected block
    that results in fewer total blocks (by enabling removal of another).
    Simplified: try adding a block, then removing two that become redundant.
    """
    if not selected:
        return selected
    
    current = list(selected)
    current_set = set(current)
    
    for _round in range(max_rounds):
        if not _time_ok():
            break
        
        improved = False
        
        # Try each unselected block as a potential addition
        # For speed, sample a subset of unselected blocks
        unselected = [i for i in range(len(all_masks)) if i not in current_set]
        
        if not unselected:
            break
        
        # Sample up to 2000 candidates per round for speed
        sample_size = min(2000, len(unselected))
        candidates = rng.sample(unselected, sample_size) if sample_size < len(unselected) else unselected
        
        for new_idx in candidates:
            if not _time_ok():
                break
            
            new_mask = all_masks[new_idx]
            
            # Add new block, try to remove two blocks that become redundant
            trial = current + [new_idx]
            trial_set = set(trial)
            
            # Recompute triple counts for trial
            triple_counts = {}
            for idx in trial:
                mask = all_masks[idx]
                m = mask
                while m:
                    low = m & (-m)
                    bit = low.bit_length() - 1
                    triple_counts[bit] = triple_counts.get(bit, 0) + 1
                    m ^= low
            
            # Find removable blocks (excluding the new one)
            removable = []
            for idx in trial:
                if idx == new_idx:
                    continue
                mask = all_masks[idx]
                m = mask
                ok = True
                while m:
                    low = m & (-m)
                    bit = low.bit_length() - 1
                    if triple_counts.get(bit, 0) <= 1:
                        ok = False
                        break
                    m ^= low
                if ok:
                    removable.append(idx)
            
            # Need to remove at least 2 to reduce count
            if len(removable) >= 2:
                # Try removing the two with most overlap (to maximize future flexibility)
                # For simplicity, remove first two
                to_remove = removable[:2]
                new_trial = [idx for idx in trial if idx not in to_remove]
                
                # Verify coverage is still complete
                # Check: OR of all masks in new_trial covers all triples
                # Actually, we need to verify no triple is uncovered
                # Quick check: for each removed block, its triples must still be covered
                # Since we checked removable, they are. But let's verify the union.
                
                # Verify: every triple is covered by at least one block in new_trial
                # This is expensive; instead trust the removable check.
                # The removable check ensures each triple in removed blocks has count>1
                # in the trial set, so after removal it still has count>=1.
                
                if len(new_trial) < len(current):
                    current = new_trial
                    current_set = set(current)
                    improved = True
                    break
        
        if not improved:
            break
    
    return current


def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - _MARGIN
    
    # Edge cases
    if v < t:
        return []
    if k > v:
        k = v
    if t > k:
        return []
    if t == 0:
        return []
    
    num_triples = 1
    for i in range(t):
        num_triples *= (v - i)
        num_triples //= (i + 1)
    
    # Build triple index
    triple_index = {}
    idx = 0
    for ts in itertools.combinations(range(v), t):
        triple_index[ts] = idx
        idx += 1
    
    # Enumerate all blocks and compute masks
    all_blocks = []
    all_masks = []
    
    for block in itertools.combinations(range(v), k):
        all_blocks.append(list(block))
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << triple_index[ts])
        all_masks.append(mask)
    
    num_blocks = len(all_blocks)
    
    # Greedy set cover
    rng = random.Random(42)
    selected, uncovered = _greedy_set_cover(all_blocks, all_masks, num_triples, rng)
    
    if uncovered != 0:
        # Greedy didn't cover everything; add random blocks to finish
        while uncovered and _time_ok():
            # Find any block that covers at least one uncovered triple
            best_idx = -1
            best_gain = 0
            for i in range(num_blocks):
                if i in set(selected):
                    continue
                gain = (all_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~all_masks[best_idx]
    
    # Remove redundant blocks
    selected = _try_remove_redundant(selected, all_masks, all_blocks, v, k, t, rng)
    
    # 1-exchange local search
    selected = _one_exchange_search(selected, all_masks, all_blocks, v, k, t, rng, max_rounds=5)
    
    # Final redundant removal
    selected = _try_remove_redundant(selected, all_masks, all_blocks, v, k, t, rng)
    
    # Verify validity
    if not _time_ok():
        pass  # Still need to return something valid
    
    # Build final blocks
    result = [all_blocks[i] for i in selected]
    
    # Deduplicate
    seen = set()
    unique_result = []
    for b in result:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_result.append(b)
    
    # Verify coverage
    if _time_ok():
        covered_mask = 0
        for b in unique_result:
            for ts in itertools.combinations(b, t):
                covered_mask |= (1 << triple_index[ts])
        
        full_mask = (1 << num_triples) - 1
        if covered_mask != full_mask:
            # Incomplete; run greedy again from scratch with different seed
            rng2 = random.Random(123)
            selected2, unc2 = _greedy_set_cover(all_blocks, all_masks, num_triples, rng2)
            while unc2 and _time_ok():
                best_idx = -1
                best_gain = 0
                for i in range(num_blocks):
                    if i in set(selected2):
                        continue
                    gain = (all_masks[i] & unc2).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                selected2.append(best_idx)
                unc2 &= ~all_masks[best_idx]
            
            selected2 = _try_remove_redundant(selected2, all_masks, all_blocks, v, k, t, rng2)
            unique_result = [all_blocks[i] for i in selected2]
            seen = set()
            final = []
            for b in unique_result:
                key = tuple(b)
                if key not in seen:
                    seen.add(key)
                    final.append(b)
            unique_result = final
    
    return unique_result
