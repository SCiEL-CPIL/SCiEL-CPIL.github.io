"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np
from collections import defaultdict


def _greedy_cover(v, k, t, rng, deadline):
    """Greedy covering with randomized block selection using numpy for speed."""
    all_tsubs = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            return None  # timeout
        
        # Pick a random uncovered t-subset to start a new block
        start_tsub = rng.choice(list(uncovered))
        block = list(start_tsub)
        block_set = set(block)
        
        while len(block) < k:
            # Find point that covers most uncovered t-subsets
            best_gain = -1
            best_point = -1
            
            # Precompute combinations of block of size t-1
            block_combos = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in block_combos:
                    # Create the t-subset with x added
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # No point improves; pick random
                candidates = [x for x in range(v) if x not in block_set]
                best_point = rng.choice(candidates)
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    
    return blocks


def _remove_redundant_blocks(blocks, v, t):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    # Count how many blocks cover each t-subset
    cover_count = defaultdict(int)
    block_tsubs = []
    
    for block in blocks:
        tsubs = list(itertools.combinations(block, t))
        block_tsubs.append(tsubs)
        for ts in tsubs:
            cover_count[ts] += 1
    
    # Remove blocks where all their t-subsets are covered by other blocks
    kept_blocks = []
    for i, block in enumerate(blocks):
        all_redundant = True
        for ts in block_tsubs[i]:
            if cover_count[ts] == 1:
                all_redundant = False
                break
        if not all_redundant:
            kept_blocks.append(block)
    
    # Update cover counts for remaining blocks
    if len(kept_blocks) < len(blocks):
        cover_count2 = defaultdict(int)
        for block in kept_blocks:
            for ts in itertools.combinations(block, t):
                cover_count2[ts] += 1
        
        # Second pass: remove blocks that are now redundant
        final_blocks = []
        for i, block in enumerate(kept_blocks):
            all_redundant = True
            for ts in itertools.combinations(block, t):
                if cover_count2[ts] == 1:
                    all_redundant = False
                    break
            if not all_redundant:
                final_blocks.append(block)
        
        return final_blocks
    
    return blocks


def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    all_tsubs = set(itertools.combinations(range(v), t))
    return all_tsubs.issubset(covered)


def _count_redundant_blocks(blocks, v, t):
    """Count how many blocks are redundant (all their t-subsets covered by other blocks)."""
    if not blocks:
        return 0, []
    
    cover_count = defaultdict(int)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            cover_count[ts] += 1
    
    redundant = []
    for i, block in enumerate(blocks):
        is_redundant = True
        for ts in itertools.combinations(block, t):
            if cover_count[ts] <= 1:
                is_redundant = False
                break
        if is_redundant:
            redundant.append(i)
    
    return len(redundant), redundant


def _remove_redundant_iterative(blocks, v, t, deadline=None):
    """Aggressively remove redundant blocks in multiple passes."""
    if not blocks:
        return blocks
    
    changed = True
    while changed:
        if deadline and time.time() > deadline:
            break
        changed = False
        _, redundant = _count_redundant_blocks(blocks, v, t)
        if redundant:
            # Remove all redundant blocks at once
            remove_set = set(redundant)
            new_blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
            blocks = new_blocks
            changed = True
    
    return blocks


def _local_search(blocks, v, k, t, rng, deadline):
    """Enhanced local search: swap points to create redundancy, then delete redundant blocks."""
    if not blocks:
        return blocks
    
    current_blocks = [list(b) for b in blocks]
    best_blocks = [list(b) for b in current_blocks]
    best_count = len(current_blocks)
    
    # Compute coverage counts
    cover_count = defaultdict(int)
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            cover_count[ts] += 1
    
    # First, remove any initially redundant blocks
    current_blocks = _remove_redundant_iterative(current_blocks, v, t, deadline)
    best_blocks = [list(b) for b in current_blocks]
    best_count = len(current_blocks)
    
    # Recompute cover counts after removal
    cover_count = defaultdict(int)
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            cover_count[ts] += 1
    
    # Try swaps: for each pair of blocks, try swapping points to increase redundancy
    max_swap_rounds = 50
    for swap_round in range(max_swap_rounds):
        if time.time() > deadline:
            break
        
        made_swap = False
        n_blocks = len(current_blocks)
        
        # Shuffle block order for diversity
        block_order = list(range(n_blocks))
        rng.shuffle(block_order)
        
        for i_idx in block_order:
            if time.time() > deadline:
                break
            i = i_idx
            if i >= len(current_blocks):
                continue
            
            block_i = current_blocks[i]
            
            # For each point in block_i, try swapping with a point from another block
            for pos in range(len(block_i)):
                if time.time() > deadline:
                    break
                orig_point = block_i[pos]
                
                # Try removing orig_point and adding a new point
                # Gain: how many t-subsets become covered (or more covered)
                # We want to increase the minimum coverage count to enable deletion
                
                # Compute t-subsets affected by removing orig_point
                # A t-subset is affected if it contains orig_point and (t-1) other points from block_i
                affected_remove = []
                for ts in itertools.combinations(block_i, t):
                    if orig_point in ts:
                        affected_remove.append(ts)
                
                # Try candidate new points
                best_gain = 0
                best_new_point = -1
                best_new_block_j = -1
                best_new_pos = -1
                
                for j in range(n_blocks):
                    if j == i:
                        continue
                    block_j = current_blocks[j]
                    for pos_j in range(len(block_j)):
                        new_point = block_j[pos_j]
                        if new_point in block_i:
                            continue
                        
                        # Simulate: remove orig_point from block_i, add new_point
                        # Remove orig_point's t-subsets from coverage
                        # Add new_point's t-subsets to coverage
                        
                        # Count how many affected_remove t-subsets would lose coverage
                        loss = 0
                        for ts in affected_remove:
                            if cover_count[ts] <= 1:
                                loss += 1
                        
                        if loss > 0:
                            continue  # This swap would break coverage
                        
                        # Count gain: t-subsets with new_point that gain coverage
                        # New t-subsets: (t-1) from block_i (without orig_point) + new_point
                        block_i_without_orig = [p for p in block_i if p != orig_point]
                        gain = 0
                        for ts_rest in itertools.combinations(block_i_without_orig, t - 1):
                            ts_new = tuple(sorted(ts_rest + (new_point,)))
                            if cover_count.get(ts_new, 0) == 1:
                                gain += 1  # Becomes double-covered, potentially enabling deletion
                        
                        # Also consider: does this make block_j more redundant?
                        # If we remove new_point from block_j, check if it's still covered
                        # This is complex; approximate by checking if block_j's t-subsets 
                        # with new_point are all covered by others
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_new_point = new_point
                            best_new_block_j = j
                            best_new_pos = pos_j
                
                if best_gain > 0 and best_new_point != -1:
                    # Perform the swap
                    j = best_new_block_j
                    pos_j = best_new_pos
                    new_point = best_new_point
                    orig_point = block_i[pos]
                    
                    # Update block_i: replace orig_point with new_point
                    block_i[pos] = new_point
                    block_i.sort()
                    
                    # Update block_j: remove new_point (or replace with something)
                    # Actually, we're "stealing" new_point from block_j
                    # We need to either leave a hole or find a replacement
                    # For simplicity, just remove it from block_j (making it size k-1)
                    # This isn't valid... let me reconsider.
                    
                    # Better approach: swap orig_point and new_point between blocks
                    current_blocks[j][pos_j] = orig_point
                    current_blocks[j].sort()
                    
                    # Update cover counts
                    # Remove old t-subsets from block_i (with orig_point)
                    old_block_i = [p for p in block_i if p != new_point] + [orig_point]
                    old_block_i.sort()
                    for ts in itertools.combinations(old_block_i, t):
                        cover_count[ts] -= 1
                    
                    # Remove old t-subsets from block_j (with new_point)
                    old_block_j = [p for p in current_blocks[j] if p != orig_point] + [new_point]
                    old_block_j.sort()
                    for ts in itertools.combinations(old_block_j, t):
                        cover_count[ts] -= 1
                    
                    # Add new t-subsets
                    for ts in itertools.combinations(block_i, t):
                        cover_count[ts] += 1
                    for ts in itertools.combinations(current_blocks[j], t):
                        cover_count[ts] += 1
                    
                    # Check for and remove redundant blocks
                    current_blocks = _remove_redundant_iterative(current_blocks, v, t, deadline)
                    if len(current_blocks) < best_count:
                        best_blocks = [list(b) for b in current_blocks]
                        best_count = len(current_blocks)
                    
                    # Recompute cover counts after potential removals
                    cover_count = defaultdict(int)
                    for block in current_blocks:
                        for ts in itertools.combinations(block, t):
                            cover_count[ts] += 1
                    
                    made_swap = True
                    break  # Restart inner loop with new block structure
            
            if made_swap:
                break
        
        if not made_swap:
            break
    
    return best_blocks


def construct(v, k, t):
    deadline = time.time() + 140  # Safety margin before 150s limit
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts with varied seeds
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline - 20:  # Leave 20s for final polish
            break
        
        # Use different seed each restart
        rng_restart = random.Random(42 + restart * 7)
        
        blocks = _greedy_cover(v, k, t, rng_restart, deadline - 20)
        if blocks is None:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, t)
        
        # Local search refinement
        blocks = _local_search(blocks, v, k, t, rng_restart, deadline - 20)
        
        # Verify validity
        if _verify_covering(blocks, v, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
        
        # Early exit if we hit a good solution
        if best_count <= 76:
            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy (original approach)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
        best_count = len(best_blocks)
    
    # Final polish pass: use remaining time to refine the best solution
    if time.time() < deadline - 5:
        polish_deadline = deadline - 5
        polish_rng = random.Random(999)
        polished = _local_search(best_blocks, v, k, t, polish_rng, polish_deadline)
        
        # Verify and use if valid
        if _verify_covering(polished, v, t):
            if len(polished) < best_count:
                best_blocks = polished
                best_count = len(polished)
        
        # One more aggressive deletion pass
        final = _remove_redundant_iterative(best_blocks, v, t, deadline - 2)
        if _verify_covering(final, v, t):
            best_blocks = final
    
    return best_blocks
# EVOLVE-BLOCK-END
