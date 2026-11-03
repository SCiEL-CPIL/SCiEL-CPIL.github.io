import time
import random
import itertools
import math
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit

    # 1. Precomputation
    # All t-subsets (items to cover)
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx_t = {s: i for i, s in enumerate(all_t)}
    
    # All k-subsets (potential blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    M = len(all_blocks)
    
    # block_cover[b] = list of indices of t-subsets covered by block b
    # We store these as sets for faster intersection operations if needed, 
    # but lists are fine for counting.
    block_cover = []
    for b in all_blocks:
        covered = [idx_t[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # t_to_blocks[i] = set of block indices that cover t-subset i
    # Using sets for fast intersection in the repair phase
    t_to_blocks = [set() for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].add(bi)

    # Helper: Check if a list of block indices forms a valid cover
    def is_valid(blocks):
        if not blocks:
            return N == 0
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return np.all(cnt > 0)

    # Helper: Compute coverage counts
    def get_counts(blocks):
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        return cnt

    def prune_to_minimal(blocks):
        """Remove redundant blocks from a valid cover."""
        cnt = get_counts(blocks)
        current_blocks = list(blocks)
        i = 0
        while i < len(current_blocks):
            bi = current_blocks[i]
            # Check if bi is redundant
            redundant = True
            for ti in block_cover[bi]:
                if cnt[ti] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove it
                current_blocks.pop(i)
                for ti in block_cover[bi]:
                    cnt[ti] -= 1
                # Do not increment i, check next block at same index
            else:
                i += 1
        return current_blocks

    def probabilistic_construct(p, seed):
        """
        Construct a cover using a probabilistic approach.
        1. Randomly select blocks with probability p.
        2. Identify uncovered t-subsets.
        3. For each uncovered t-subset, find the 'best' block to add 
           (heuristic: block that covers the most *other* uncovered t-subsets).
        4. Prune the result.
        """
        rng = random.Random(seed)
        
        # Phase 1: Random Selection
        selected_blocks = set()
        for bi in range(M):
            if rng.random() < p:
                selected_blocks.add(bi)
        
        # Phase 2: Repair
        # Calculate current coverage
        cnt = np.zeros(N, dtype=np.int32)
        for bi in selected_blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        
        # Identify uncovered items
        uncovered_indices = np.where(cnt == 0)[0]
        
        # While there are uncovered items, add blocks
        # To speed up, we process uncovered items in batches or one by one.
        # Since N is small (11628), we can afford a loop.
        # However, adding one block might cover multiple uncovered items.
        
        # Strategy: 
        # While uncovered set is not empty:
        #   Pick an uncovered item 'target'
        #   Find all blocks covering 'target'
        #   Among them, pick the one that covers the most *currently uncovered* items
        #   Add that block, update counts, update uncovered set.
        
        # To make this efficient, we maintain a list/set of uncovered items.
        uncovered_set = set(uncovered_indices.tolist())
        
        # Optimization: Pre-compute "potential" for each block? 
        # No, potential changes as we cover more items.
        
        # We can sort uncovered items by "scarcity" (number of blocks covering them)
        # to handle hard-to-cover items first.
        # But computing scarcity for all items every step is expensive.
        
        # Simpler repair:
        # Just iterate through uncovered items. If an item is still uncovered, 
        # find the best block for it.
        
        # To avoid O(N * M) in the worst case, we can be smarter.
        # But with p ~ 0.01-0.05, the initial random selection covers a lot.
        # The repair phase should be short.
        
        # Let's use a queue of uncovered items.
        # When we add a block, we remove all newly covered items from the queue.
        
        # Re-initialize queue with current uncovered
        queue = list(uncovered_indices.tolist())
        
        # To make the "best block" selection faster, we can pre-compute 
        # for each block, the list of t-subsets it covers (we have block_cover).
        
        while queue:
            if time.time() > deadline - 5.0:
                break # Timeout, return what we have (might be invalid)
                
            # Pick an item to cover. 
            # Heuristic: Pick the item with the fewest candidate blocks?
            # Or just pick the first one?
            # Picking the first one is fast. Let's try that first.
            target = queue.pop(0)
            
            # Check if it's still uncovered (might have been covered by a previous addition in this batch? 
            # No, we update cnt immediately after adding a block, but we don't remove from queue immediately 
            # unless we do a full scan. 
            # To be safe and simple, let's check cnt[target] > 0.
            if cnt[target] > 0:
                continue
            
            # Find candidates
            candidates = t_to_blocks[target]
            if not candidates:
                return None # Should not happen
            
            # Find the candidate that covers the most *currently uncovered* items
            # "Currently uncovered" means cnt[ti] == 0.
            best_b = -1
            max_gain = -1
            
            # Optimization: If the number of candidates is small, check all.
            # If large, sample? No, accuracy is better for repair.
            for bi in candidates:
                gain = 0
                for ti in block_cover[bi]:
                    if cnt[ti] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_b = bi
                    if max_gain == len(block_cover[bi]):
                        break # Can't do better than covering all its items
            
            if best_b == -1:
                return None
            
            # Add best_b
            selected_blocks.add(best_b)
            
            # Update counts
            for ti in block_cover[best_b]:
                cnt[ti] += 1
                # If it was 0, it's now covered.
                # We don't explicitly remove from queue here to save time, 
                # we rely on the `if cnt[target] > 0: continue` check.
                # This means the queue might have stale entries, but that's fine.
            
            # Optional: If we want to be cleaner, we could rebuild the queue, 
            # but that's O(N). The stale check is O(1).

        # Phase 3: Prune
        if not selected_blocks:
            return []
            
        blocks_list = list(selected_blocks)
        # Ensure validity before pruning (pruning assumes valid input)
        # If the repair loop broke early due to timeout, it might be invalid.
        if not is_valid(blocks_list):
             # Fallback: Try to finish repair? Or just return invalid? 
             # The task requires a valid cover. 
             # If we timed out in repair, we should probably try a different p or seed.
             # For now, let's assume repair finishes or we return None to try next seed.
             # But if we return None, we might not have a valid cover at the end.
             # Let's force validity check in the main loop.
             pass
             
        pruned = prune_to_minimal(blocks_list)
        
        # Final validity check
        if is_valid(pruned):
            return pruned
        else:
            return None # Invalid, discard

    # Main Loop: Try different probabilities
    best_blocks = None
    best_score = float('inf')
    
    # Probabilities to try. 
    # Expected number of blocks is M * p.
    # M = 75582.
    # If p=0.005, ~377 blocks.
    # If p=0.002, ~151 blocks (likely too few, repair will add many).
    # If p=0.01, ~755 blocks.
    
    # The optimal p is likely such that the random selection covers ~90% of items,
    # and repair fixes the rest.
    
    # Let's try a range of p values and seeds.
    p_values = [0.005, 0.008, 0.01, 0.015, 0.02]
    seeds = range(20)
    
    # To maximize diversity, we can interleave p and seeds.
    trials = []
    for p in p_values:
        for s in seeds:
            trials.append((p, s))
            
    random.shuffle(trials)
    
    for p, seed in trials:
        if time.time() > deadline - 2.0:
            break
            
        # Run probabilistic construct
        cand = probabilistic_construct(p, seed)
        
        if cand is not None:
            if len(cand) < best_score:
                best_score = len(cand)
                best_blocks = cand
                
        # If we found a very good solution, we can break early?
        # Lower bound is 264. If we get close, maybe stop.
        if best_score <= 270:
            # Try a few more to see if we can beat it, but don't spend too much time
            # Let's just continue the loop, the time check will handle it.
            pass

    # If we didn't find a valid cover (shouldn't happen with reasonable p), 
    # fallback to a simple greedy
    if best_blocks is None:
        # Simple greedy fallback
        uncovered = set(range(N))
        blocks = []
        while uncovered:
            target = next(iter(uncovered))
            candidates = t_to_blocks[target]
            best_b = -1
            max_gain = -1
            for bi in candidates:
                gain = sum(1 for ti in block_cover[bi] if ti in uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_b = bi
            blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
        best_blocks = prune_to_minimal(blocks)
        best_score = len(best_blocks)

    # Final validation
    if not is_valid(best_blocks):
        # This is a critical failure. Return a known valid cover if possible?
        # Or just return the blocks (evaluator will mark invalid).
        # Let's try to fix it quickly?
        # For safety, if invalid, return empty list? No, must be valid.
        # The greedy fallback is usually valid.
        # If even that fails, something is wrong with precomputation.
        pass

    return [list(all_blocks[bi]) for bi in best_blocks]
