"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np
from collections import defaultdict


def _greedy_cover(v, k, t, rng, deadline):
    """
    Fast greedy covering.
    Uses a bitset (int) to represent the set of uncovered t-subsets.
    Each point x has a precomputed 'signature' bitmask: which t-subsets contain x.
    A block's coverage is the OR of its points' signatures.
    """
    # 1. Index all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # 2. Build point signatures
    # sig[x] is a bitset of t-subsets containing x
    sig = [0] * v
    for i, ts in enumerate(all_tsubs):
        bit = 1 << i
        for x in ts:
            sig[x] |= bit
    
    # 3. Greedy Loop
    uncovered_mask = (1 << n_tsubs) - 1
    blocks = []
    
    while uncovered_mask:
        if time.time() > deadline:
            return None
        
        # Pick a random uncovered t-subset to seed the block
        # Get an index of a set bit
        # For speed, just pick a random index and check if it's set?
        # No, we need a *set* bit.
        # Since n_tsubs can be large, finding a set bit in a Python int is fast (C level).
        # We can pick a random position and test, or just find the lowest set bit.
        # Let's pick a random t-subset from the list and check if it's uncovered.
        # If n_tsubs is ~2000, this is fast.
        
        # Optimization: maintain a list of uncovered tsub indices? 
        # Updating the list is O(N). Updating the mask is O(1).
        # Let's just find any set bit.
        
        # To randomize, we can rotate the mask or pick a random offset.
        # Simple approach: pick a random index 0..n_tsubs-1, if not set, retry.
        # Or: find lowest set bit, then randomize by shifting?
        # Let's just pick the lowest set bit for determinism in the "core" greedy, 
        # but the task asks for randomized restarts. 
        # Actually, the current code picks `rng.choice(list(uncovered))`.
        # Converting mask to list is O(N).
        # Let's do: `idx = (uncovered_mask & -uncovered_mask).bit_length() - 1` (lowest set bit).
        # To randomize, we can pick a random `shift` and find lowest set bit of `mask >> shift`? No.
        
        # Better: Just iterate through a random permutation of indices? No.
        
        # Let's use the lowest set bit for speed, but add randomization to the *block construction* 
        # or use multiple restarts with different "biases".
        # Actually, the current code's randomization is in `start_tsub`.
        # We can simulate this by picking a random `k` and checking bit `k`.
        
        # Since n_tsubs is small (2024 for v=24, t=3), we can afford to do:
        # candidates = [i for i in range(n_tsubs) if (uncovered_mask >> i) & 1]
        # This is O(N). With 100 restarts, 100 * 2000 = 200k ops. Negligible.
        
        # Let's stick to a fast method:
        # Find one set bit.
        # `lsb = uncovered_mask & -uncovered_mask`
        # `start_idx = lsb.bit_length() - 1`
        
        # To allow randomization as in the original:
        # We can pick a random offset `r` in [0, n_tsubs)
        # and check if bit `r` is set. If yes, use it. If no, fallback to lowest set bit?
        # Or just use the lowest set bit and rely on the "random" part being the order of points added?
        # The original code's random start tsub is crucial for diversity.
        
        # Let's generate a random index and check.
        start_idx = -1
        for _ in range(10): # Max 10 tries to find a random uncovered tsub
            r = rng.randint(0, n_tsubs - 1)
            if (uncovered_mask >> r) & 1:
                start_idx = r
                break
        if start_idx == -1:
            # Fallback to lowest set bit
            lsb = uncovered_mask & -uncovered_mask
            start_idx = lsb.bit_length() - 1
            
        start_tsub = all_tsubs[start_idx]
        
        # Construct block
        block = list(start_tsub)
        block_set = set(block)
        
        # Current coverage mask of this block
        block_mask = 0
        for x in block:
            block_mask |= sig[x]
        
        while len(block) < k:
            # Find point that covers most *new* uncovered t-subsets
            best_gain = -1
            best_point = -1
            best_sig = 0
            
            # We want to maximize | (sig[x] & uncovered_mask) |
            # We can iterate over all x not in block.
            
            # Optimization: Precompute candidates?
            # v=24 is small.
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count bits in (sig[x] & uncovered_mask)
                # Python int.bit_count() is fast.
                gain = (sig[x] & uncovered_mask).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_sig = sig[x]
            
            if best_point == -1:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_point = rng.choice(candidates)
                best_sig = sig[best_point]
            
            block.append(best_point)
            block_set.add(best_point)
            block_mask |= best_sig
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered mask
        # uncovered_mask &= ~block_mask
        uncovered_mask &= ~block_mask
        
    return blocks


def _remove_redundant_blocks(blocks, v, t):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    # Count how many blocks cover each t-subset
    # Using a dict is fine for ~2000 tsubs.
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


def _local_search(blocks, v, k, t, rng, deadline):
    """
    Local search: try swapping points in blocks to increase redundancy,
    then remove redundant blocks.
    """
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
    current_blocks = _remove_redundant_blocks(current_blocks, v, t)
    best_blocks = [list(b) for b in current_blocks]
    best_count = len(current_blocks)
    
    # Recompute cover counts
    cover_count = defaultdict(int)
    for block in current_blocks:
        for ts in itertools.combinations(block, t):
            cover_count[ts] += 1
    
    # Precompute t-subsets for each block for faster diffing
    block_tsubs = [set(itertools.combinations(b, t)) for b in current_blocks]
    
    # Local Search Loop
    # Strategy: Randomly pick a block and a point. Try swapping it with a random point from another block.
    # If the swap is valid (doesn't uncover any t-subset), and increases "redundancy score", accept.
    # "Redundancy score" = sum of cover_counts for all t-subsets.
    # Higher score -> more potential for deletion.
    
    # Actually, a simpler heuristic:
    # Try to make one block redundant by "stealing" its coverage from others? No, that increases blocks.
    # We want to make *existing* blocks redundant.
    # So we want to increase cover_count of t-subsets in a specific block.
    
    # Let's try: Pick a block B. Try to modify B (swap a point) such that B's t-subsets become more covered by OTHER blocks.
    # Wait, if we change B, we lose B's coverage.
    # The standard approach for covering design reduction:
    # 1. Try to delete a block. If it fails, find a t-subset that is ONLY covered by this block.
    # 2. Modify another block to cover that t-subset.
    # 3. Now the original block is redundant. Delete it.
    
    # This is "2-opt" or "swap" style.
    
    # Let's implement this specific logic:
    # 1. Identify candidate blocks for deletion (blocks with many "unique" t-subsets? No, blocks with FEW unique t-subsets are easier to delete).
    # Actually, if a block has 0 unique t-subsets, it's already deleted.
    # So we look for blocks with minimal unique t-subsets.
    
    # Let's iterate through blocks. For each block, try to find a way to cover its unique t-subsets using other blocks (by modifying them).
    
    max_iter = 2000
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random block to try to delete
        if not current_blocks:
            break
        idx = rng.randint(0, len(current_blocks) - 1)
        block_to_delete = current_blocks[idx]
        
        # Find unique t-subsets in this block
        unique_tsubs = []
        for ts in block_tsubs[idx]:
            if cover_count[ts] == 1:
                unique_tsubs.append(ts)
        
        if not unique_tsubs:
            # Block is redundant!
            # Remove it
            for ts in block_tsubs[idx]:
                cover_count[ts] -= 1
            current_blocks.pop(idx)
            block_tsubs.pop(idx)
            best_blocks = [list(b) for b in current_blocks]
            best_count = len(current_blocks)
            continue
        
        # We need to cover these unique_tsubs using other blocks.
        # Pick one unique t-subset.
        target_ts = rng.choice(unique_tsubs)
        
        # Find a block that can be modified to cover target_ts.
        # Ideally, a block that already covers (t-1) elements of target_ts.
        
        # Heuristic: Look for a block J != idx that contains (t-1) elements of target_ts.
        # Then we can swap one element in J to complete target_ts.
        
        found_swap = False
        # To speed up, we can just try random blocks
        for _ in range(10):
            j = rng.randint(0, len(current_blocks) - 1)
            if j == idx:
                continue
            
            block_j = current_blocks[j]
            block_j_set = set(block_j)
            
            # Check how many elements of target_ts are in block_j
            common = block_j_set.intersection(target_ts)
            
            if len(common) == t - 1:
                # We need to add the missing element and remove one from block_j
                missing_elem = next(x for x in target_ts if x not in block_j_set)
                
                # Try to find an element in block_j to remove such that:
                # 1. The new block_j covers target_ts.
                # 2. We don't lose coverage of any t-subset that was ONLY covered by block_j.
                
                # If we remove y from block_j, we lose t-subsets containing y and (t-1) from block_j\{y}.
                # We gain t-subsets containing missing_elem and (t-1) from block_j\{y}.
                
                # This is complex to evaluate quickly for all y.
                # Let's just try a random y from block_j that is NOT in target_ts.
                # (If y is in target_ts, removing it breaks the (t-1) match).
                
                candidates_to_remove = [y for y in block_j if y not in target_ts]
                if not candidates_to_remove:
                    continue
                
                y = rng.choice(candidates_to_remove)
                
                # Check validity
                # Lost t-subsets: those containing y and (t-1) from block_j without y.
                # Gained t-subsets: those containing missing_elem and (t-1) from block_j without y.
                
                block_j_without_y = [x for x in block_j if x != y]
                
                valid = True
                # Check lost
                for rest in itertools.combinations(block_j_without_y, t - 1):
                    ts_lost = tuple(sorted(rest + (y,)))
                    if cover_count[ts_lost] == 1:
                        valid = False
                        break
                
                if valid:
                    # Check gained (just to ensure we actually cover target_ts, which we know we do)
                    # But we should check if this creates redundancy elsewhere?
                    # For now, just accept if valid.
                    
                    # Perform swap
                    # Update cover counts
                    for rest in itertools.combinations(block_j_without_y, t - 1):
                        ts_lost = tuple(sorted(rest + (y,)))
                        cover_count[ts_lost] -= 1
                        ts_gained = tuple(sorted(rest + (missing_elem,)))
                        cover_count[ts_gained] += 1
                    
                    # Update block_j
                    block_j.remove(y)
                    block_j.append(missing_elem)
                    block_j.sort()
                    current_blocks[j] = block_j
                    block_tsubs[j] = set(itertools.combinations(block_j, t))
                    
                    # Now try to delete block idx again
                    # Recompute unique tsubs for idx
                    # (Cover counts for idx's tsubs haven't changed, except if j==idx which is false)
                    # So we can just check if target_ts is now covered by j.
                    # Actually, we modified j to cover target_ts.
                    # So cover_count[target_ts] increased by 1.
                    # If it was 1 (only idx), now it is 2.
                    # So idx might have fewer unique tsubs.
                    
                    # Let's just loop back to the top of the while loop?
                    # No, let's try to delete idx immediately.
                    
                    # Check if idx is redundant
                    is_redundant = True
                    for ts in block_tsubs[idx]:
                        if cover_count[ts] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        # Delete idx
                        for ts in block_tsubs[idx]:
                            cover_count[ts] -= 1
                        current_blocks.pop(idx)
                        block_tsubs.pop(idx)
                        best_blocks = [list(b) for b in current_blocks]
                        best_count = len(current_blocks)
                    
                    found_swap = True
                    break
        
        if found_swap:
            continue # Restart search with new state
        # If no swap found, continue to next iteration (randomness will help)
        
    return best_blocks


def construct(v, k, t):
    deadline = time.time() + 140  # Safety margin before 150s limit
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized restarts
    # Since greedy is fast, we can do many restarts.
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 20:  # Leave 20s for final polish
            break
        
        rng_restart = random.Random(42 + restart * 13)
        
        blocks = _greedy_cover(v, k, t, rng_restart, deadline - 20)
        if blocks is None:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, t)
        
        # Local search refinement
        # Give local search a smaller deadline so we can do multiple restarts
        ls_deadline = deadline - 20 - 5 # 5s for local search per restart?
        # Actually, local search can be expensive.
        # Let's allocate 2s per restart for local search if we have many restarts.
        # Or just do local search on the best few.
        
        # Strategy:
        # 1. Run greedy + simple reduction.
        # 2. Track best.
        # 3. After all restarts, run local search on the best one with remaining time.
        
        if _verify_covering(blocks, v, t):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
        
        # Early exit if we hit a good solution
        if best_count <= 76:
            break
    
    if best_blocks is None:
        # Fallback
        return []
    
    # Final polish: Local search on the best solution found
    if time.time() < deadline - 5:
        polish_deadline = deadline - 5
        polish_rng = random.Random(999)
        polished = _local_search(best_blocks, v, k, t, polish_rng, polish_deadline)
        
        if _verify_covering(polished, v, t):
            if len(polished) < best_count:
                best_blocks = polished
                best_count = len(polished)
        
        # One more aggressive deletion pass
        final = _remove_redundant_blocks(best_blocks, v, t)
        if _verify_covering(final, v, t):
            best_blocks = final
    
    return best_blocks
# EVOLVE-BLOCK-END
