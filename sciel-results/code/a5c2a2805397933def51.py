import itertools
import random
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute(v, k, t):
    """Precompute all t-sets, all k-blocks, and mappings."""
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, which t-sets it covers
    block_tsets = []
    for block in all_blocks:
        tset_indices = set()
        for ts in itertools.combinations(block, t):
            tset_indices.add(tset_to_idx[ts])
        block_tsets.append(frozenset(tset_indices))
        
    # For each t-set, which blocks cover it
    tset_to_blocks = defaultdict(list)
    for b_idx, ts_set in enumerate(block_tsets):
        for ts_idx in ts_set:
            tset_to_blocks[ts_idx].append(b_idx)
            
    return all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks

def _greedy_fill(uncovered_tsets, block_tsets, tset_to_blocks, n_tsets, rng):
    """
    Greedily add blocks to cover all uncovered t-sets.
    Returns a list of new block indices.
    """
    if not uncovered_tsets:
        return []
    
    new_blocks = []
    remaining_uncovered = set(uncovered_tsets)
    
    # We want to cover all 'remaining_uncovered'
    # Greedy: pick block covering most uncovered
    # To speed up, we can pick a random uncovered t-set and find best block for it
    # Or iterate. Let's do a simplified greedy:
    # While there are uncovered t-sets:
    #   Pick a random uncovered t-set
    #   Find the block (from all_blocks) that covers this t-set and maximizes coverage of OTHER uncovered t-sets
    #   Add that block
    #   Update uncovered
    
    # Optimization: Pre-compute for each t-set, the list of blocks covering it is in tset_to_blocks.
    # But we need to evaluate gain across ALL uncovered.
    
    while remaining_uncovered:
        # Pick a random uncovered t-set to anchor the search
        rand_ts = random.choice(list(remaining_uncovered))
        
        best_block = -1
        best_gain = -1
        
        # Candidates are blocks covering rand_ts
        for b_idx in tset_to_blocks[rand_ts]:
            gain = 0
            for ts_idx in block_tsets[b_idx]:
                if ts_idx in remaining_uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1 or best_gain == 0:
            # Should not happen if a covering exists
            break
            
        new_blocks.append(best_block)
        
        # Remove covered t-sets from remaining_uncovered
        for ts_idx in block_tsets[best_block]:
            if ts_idx in remaining_uncovered:
                remaining_uncovered.remove(ts_idx)
                
    return new_blocks

def _construct_initial_covering(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, target_size):
    """
    Construct a covering of size close to target_size.
    Strategy: Start with random blocks, then fill gaps greedily.
    """
    # Start with a random set of blocks of size target_size
    # But random blocks might not cover everything.
    # Better: Start with empty, greedy fill to get a covering.
    # Then try to reduce size?
    # The directive says: "start from a random 76-block covering... and aggressively try to delete".
    # So we need a valid covering first.
    
    # 1. Greedy construction to get a valid covering
    uncovered = set(range(n_tsets))
    current_blocks = []
    
    while uncovered:
        rand_ts = random.choice(list(uncovered))
        best_block = -1
        best_gain = -1
        
        for b_idx in tset_to_blocks[rand_ts]:
            gain = 0
            for ts_idx in block_tsets[b_idx]:
                if ts_idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1:
            break
            
        current_blocks.append(best_block)
        for ts_idx in block_tsets[best_block]:
            if ts_idx in uncovered:
                uncovered.discard(ts_idx)
                
    # Now we have a valid covering 'current_blocks'.
    # It might be larger than target_size.
    # If it's smaller, great. If larger, we can try to swap/delete.
    # But the strategy is "deletion-first".
    # Let's assume the greedy covering is around 70-80.
    
    return current_blocks

def _try_delete_and_repair(blocks_idx, block_tsets, tset_to_blocks, n_tsets, rng, time_limit):
    """
    Try to delete a random block from the covering.
    If coverage breaks, repair by adding the best single block to fix the most damage.
    Repeat until no improvement or time limit.
    """
    deadline = time.time() + time_limit
    current_blocks = list(blocks_idx)
    
    # Compute coverage counts
    covered = np.zeros(n_tsets, dtype=int)
    for b_idx in current_blocks:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
            
    best_size = len(current_blocks)
    best_blocks = current_blocks[:]
    
    # We will iterate: try to delete a block. If it causes uncovered t-sets,
    # check if we can repair by adding ONE block that covers all newly uncovered t-sets.
    # If yes, we have a smaller covering (size - 1 + 1 = same size? No, we deleted 1, added 1 -> same size).
    # Wait, if we delete 1 and add 1, size is same.
    # We want to REDUCE size.
    # So we should delete a block and see if the remaining blocks STILL cover everything.
    # If yes, size decreases by 1.
    # If no, we can't just delete.
    
    # So the "repair" strategy for DELETION is:
    # 1. Delete a block.
    # 2. If still covered, keep it (size - 1).
    # 3. If not covered, we have a set of "broken" t-sets.
    #    Can we find a DIFFERENT block that covers ALL broken t-sets AND allows us to delete the original?
    #    No, that's a swap.
    #    
    # The prompt says: "aggressively try to delete blocks while using a fast, deterministic 'fill-in' repair to restore coverage".
    # This implies: Delete -> If broken, Fill-in (add blocks) -> Then try to delete again?
    # If I delete 1 and add 1, size is same.
    # If I delete 1 and add 0 (because it wasn't needed), size -1.
    # If I delete 1 and add 2, size +1 (bad).
    
    # So the core loop for size reduction is:
    # Check if any block is redundant. If yes, remove it.
    # This is what _remove_redundant does.
    
    # The NEW strategy must be different.
    # "Spectral clustering... symmetry-reduced instance... branch-and-bound".
    # This is complex. Let's stick to the "Deletion-First with Repair" idea but make it smarter.
    # Maybe: Randomly delete a block. If broken, try to repair by adding the SINGLE best block.
    # If the repair succeeds (all covered), we are at the same size.
    # Then we try to delete ANOTHER block from this new set.
    # This allows us to "walk" through the space of coverings of size M, M-1, etc.
    # But how do we get to M-1?
    # Only if a block is redundant.
    
    # Perhaps the "Repair" is not just adding 1 block, but allowing the set to change structure.
    
    # Let's try a different structural change: **Random Restart with Local Search on Block Substitution**.
    # But the current program already does SA.
    
    # Let's look at the "Spectral Clustering" hint.
    # Points 0..23. Blocks of size 7.
    # We can compute a similarity matrix between points based on how often they appear together in a "good" covering.
    # But we don't have a good covering yet.
    
    # Alternative Structural Change: **Iterated Local Search (ILS) with a specific perturbation**.
    # Perturbation: Remove a random block, then re-add the best block to cover the most newly uncovered t-sets.
    # This is a 1-1 swap.
    # Current SA does this.
    
    # What if we do **2-1 swaps**? Remove 2 blocks, add 1 block?
    # This can reduce size by 1.
    # Or **1-0 swaps** (deletion) if redundant.
    
    # Let's implement a search that specifically looks for **redundancies** created by **swaps**.
    # 1. Start with a greedy covering.
    # 2. Randomly pick a block B to remove.
    # 3. Identify uncovered t-sets U.
    # 4. Find a block B' that covers U. (If none, skip).
    # 5. Replace B with B'.
    # 6. Now check if any block in the new set is redundant.
    # 7. If yes, remove it. Size decreases.
    
    # This is a "Swap-then-Prune" strategy.
    
    def _get_uncovered(blocks_idx):
        cov = np.zeros(n_tsets, dtype=int)
        for b in blocks_idx:
            for ts in block_tsets[b]:
                cov[ts] += 1
        return np.where(cov == 0)[0]
        
    # Initial state
    current_blocks = list(blocks_idx)
    best_blocks = current_blocks[:]
    best_size = len(current_blocks)
    
    # Pre-compute for speed: for each block, list of t-sets it covers
    # Already in block_tsets
    
    while time.time() < deadline:
        # Pick a random block to remove
        if not current_blocks:
            break
        pos = rng.randint(0, len(current_blocks) - 1)
        old_b = current_blocks[pos]
        
        # Calculate uncovered t-sets if we remove old_b
        # We need coverage counts
        # To make this fast, maintain coverage counts
        # But for simplicity in this step, let's recompute or maintain carefully.
        
        # Let's maintain coverage counts outside the loop
        pass # This logic is getting messy inside the function. Let's refactor.

    # Refactored Approach:
    # 1. Maintain coverage counts array.
    # 2. Loop:
    #    a. Pick random block B to remove.
    #    b. Tentatively remove B.
    #    c. Check if any t-set becomes uncovered (count goes to 0).
    #    d. If no uncovered t-sets:
    #       - Keep B removed. Size - 1. Update best.
    #    e. If uncovered t-sets exist (set U):
    #       - Find a block B' that covers ALL t-sets in U.
    #       - If found:
    #           - Replace B with B'. Size same.
    #           - Now check if any OTHER block is redundant (because B' might cover things covered by others, or B' might be redundant itself? No, B' covers U which was uncovered).
    #           - Actually, after swap, check if ANY block in the set is now redundant.
    #           - If yes, remove it. Size - 1.
    #       - If not found:
    #           - Revert.
            
    return best_blocks

def _swap_prune_search(blocks_idx, block_tsets, tset_to_blocks, n_tsets, rng, time_limit):
    """
    Search strategy: Swap a block with another to enable deletion of a redundant block.
    """
    deadline = time.time() + time_limit
    current_blocks = list(blocks_idx)
    M = len(current_blocks)
    
    if M == 0:
        return current_blocks
    
    # Coverage counts
    covered = np.zeros(n_tsets, dtype=int)
    for b_idx in current_blocks:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
            
    best_size = M
    best_blocks = current_blocks[:]
    
    # Map from t-set to list of blocks covering it
    # Already have tset_to_blocks
    
    while time.time() < deadline:
        # 1. Pick a random block to remove
        pos = rng.randint(0, M - 1)
        old_b = current_blocks[pos]
        
        # 2. Tentatively remove old_b
        for ts_idx in block_tsets[old_b]:
            covered[ts_idx] -= 1
            
        # 3. Check for uncovered t-sets
        uncovered_indices = np.where(covered == 0)[0]
        
        if len(uncovered_indices) == 0:
            # Success! Block was redundant.
            # Remove it permanently
            del current_blocks[pos]
            M -= 1
            if M < best_size:
                best_size = M
                best_blocks = current_blocks[:]
            # Continue loop to find more redundancies
            continue
        else:
            # Failure: Some t-sets are uncovered
            # Find a block B' that covers ALL uncovered_indices
            # To do this fast, find intersection of tset_to_blocks for each uncovered t-set
            
            if len(uncovered_indices) > 0:
                # Start with blocks covering the first uncovered t-set
                candidate_blocks = set(tset_to_blocks[uncovered_indices[0]])
                for ts_idx in uncovered_indices[1:]:
                    candidate_blocks &= set(tset_to_blocks[ts_idx])
                    if not candidate_blocks:
                        break
                
                if candidate_blocks:
                    # Pick one candidate (random or best)
                    new_b = random.choice(list(candidate_blocks))
                    
                    # Check if new_b is already in current_blocks (excluding old_b which is removed)
                    # If new_b is in current_blocks, we can't add it (duplicates not allowed in set? 
                    # The problem says "blocks must be distinct").
                    # So we need new_b not in current_blocks.
                    if new_b not in current_blocks:
                        # Add new_b
                        current_blocks[pos] = new_b
                        for ts_idx in block_tsets[new_b]:
                            covered[ts_idx] += 1
                        
                        # Now check if any block is redundant
                        # We can do a quick scan for redundancy
                        # Since M is small (~70), scanning is fast
                        changed = False
                        for i in range(M):
                            b_idx = current_blocks[i]
                            redundant = True
                            for ts_idx in block_tsets[b_idx]:
                                if covered[ts_idx] == 1:
                                    redundant = False
                                    break
                            if redundant:
                                # Remove it
                                del current_blocks[i]
                                for ts_idx in block_tsets[b_idx]:
                                    covered[ts_idx] -= 1
                                M -= 1
                                changed = True
                                if M < best_size:
                                    best_size = M
                                    best_blocks = current_blocks[:]
                                break # Restart scan after modification
                        # If no redundancy found, we are at same size.
                        # Continue to next iteration
                    else:
                        # Revert
                        for ts_idx in block_tsets[new_b]:
                            covered[ts_idx] -= 1
                        # old_b is still removed, need to add it back?
                        # No, we are reverting the swap.
                        # So we put old_b back.
                        current_blocks[pos] = old_b
                        for ts_idx in block_tsets[old_b]:
                            covered[ts_idx] += 1
                else:
                    # No candidate block found. Revert.
                    current_blocks[pos] = old_b
                    for ts_idx in block_tsets[old_b]:
                        covered[ts_idx] += 1
            else:
                # Should not happen
                pass
                
    return best_blocks

def _greedy_init(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, time_limit):
    """Fast greedy set cover initialization."""
    deadline = time.time() + time_limit
    uncovered = np.ones(n_tsets, dtype=bool)
    blocks_chosen = []
    
    while np.any(uncovered) and time.time() < deadline:
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a random uncovered t-set
        rand_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
        
        # Find best block covering this t-set
        best_block = -1
        best_gain = -1
        for b_idx in tset_to_blocks[rand_idx]:
            gain = sum(1 for ts_idx in block_tsets[b_idx] if uncovered[ts_idx])
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block == -1 or best_gain <= 0:
            break
            
        blocks_chosen.append(best_block)
        for ts_idx in block_tsets[best_block]:
            uncovered[ts_idx] = False
            
    return blocks_chosen

def _remove_redundant(blocks_idx, block_tsets, n_tsets, time_limit):
    """Remove redundant blocks from a covering."""
    deadline = time.time() + time_limit
    current = list(blocks_idx)
    
    if not current:
        return current
    
    # Count coverage
    covered = np.zeros(n_tsets, dtype=int)
    for b_idx in current:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
            
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            b_idx = current[i]
            # Check if any t-set is only covered by this block
            redundant = True
            for ts_idx in block_tsets[b_idx]:
                if covered[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                del current[i]
                for ts_idx in block_tsets[b_idx]:
                    covered[ts_idx] -= 1
                changed = True
                break
                
    return current

def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0  # Safety margin before 150s limit
    
    # Precompute
    all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks = _precompute(v, k, t)
    
    rng = random.Random(42)
    
    # 1. Get a good initial covering
    greedy_start = _greedy_init(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, 5.0)
    greedy_start = _remove_redundant(greedy_start, block_tsets, n_tsets, 2.0)
    
    best_blocks = greedy_start[:]
    best_size = len(greedy_start)
    
    # 2. Run Swap-Prune Search
    # This is the structural change: actively swapping to enable deletion.
    elapsed = time.time()
    search_budget = deadline - time.time() - 3.0
    if search_budget > 5.0:
        sa_result = _swap_prune_search(greedy_start, block_tsets, tset_to_blocks, n_tsets, rng, search_budget)
        sa_result = _remove_redundant(sa_result, block_tsets, n_tsets, 2.0)
        if len(sa_result) < best_size:
            best_blocks = sa_result[:]
            best_size = len(sa_result)
            
    # 3. Try a few more greedy starts
    for seed in range(5):
        if time.time() > deadline - 5.0:
            break
        rng2 = random.Random(seed * 1000 + 42)
        greedy2 = _greedy_init(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng2, 3.0)
        greedy2 = _remove_redundant(greedy2, block_tsets, n_tsets, 1.0)
        if len(greedy2) < best_size:
            best_blocks = greedy2[:]
            best_size = len(greedy2)
            
        # Quick Swap-Prune from this start
        remaining = deadline - time.time() - 3.0
        if remaining > 3.0:
            sp2 = _swap_prune_search(greedy2, block_tsets, tset_to_blocks, n_tsets, rng, min(remaining - 3.0, 10.0))
            sp2 = _remove_redundant(sp2, block_tsets, n_tsets, 1.0)
            if len(sp2) < best_size:
                best_blocks = sp2[:]
                best_size = len(sp2)
                
    # Final verification
    if not best_blocks:
        # Fallback
        best_blocks = _greedy_init(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng, 10.0)
        best_blocks = _remove_redundant(best_blocks, block_tsets, n_tsets, 5.0)
        
    # Verify
    covered = np.zeros(n_tsets, dtype=int)
    for b_idx in best_blocks:
        for ts_idx in block_tsets[b_idx]:
            covered[ts_idx] += 1
    if not np.all(covered > 0):
        # Rebuild
        rng_fallback = random.Random(789)
        best_blocks = _greedy_init(v, k, t, all_tsets, n_tsets, tset_to_idx, all_blocks, n_blocks, block_tsets, tset_to_blocks, rng_fallback, 10.0)
        best_blocks = _remove_redundant(best_blocks, block_tsets, n_tsets, 5.0)
        
    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result

# EVOLVE-BLOCK-END
