import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets and map them to indices for fast lookup
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute for each possible block (k-subset) the list of t-subsets it covers
    # C(19, 8) = 75582 blocks. Each covers C(8, 5) = 56 t-subsets.
    # Total pairs: 75582 * 56 ~ 4.2M. This is manageable memory-wise and time-wise if done once.
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    num_blocks = len(all_blocks)
    
    # coverage_matrix[block_idx][t_subset_idx] = 1 if covered, 0 otherwise
    # But storing a full boolean matrix of 75582 x 11628 is ~880MB. Too much.
    # Instead, store for each block, the list of t-subset indices it covers.
    block_covers = []
    for b in all_blocks:
        covered_ts = []
        for ts in itertools.combinations(b, t):
            covered_ts.append(t_subset_to_idx[ts])
        block_covers.append(covered_ts)
        
    # Also, for fast "which blocks cover this t-subset" queries, we might need the inverse.
    # But for the nibble, we mainly need to pick a block and update coverage.
    # For local search, we need to check if a block is redundant, i.e., if all its t-subsets
    # are covered by other blocks.
    
    best_blocks = None
    best_count = float('inf')
    
    def local_search(blocks_list, time_limit):
        # Convert to a set for fast lookup of current blocks
        current_blocks = set(blocks_list)
        # Track coverage count for each t-subset
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in current_blocks:
            idx = block_to_idx[b]
            for ts_idx in block_covers[idx]:
                cover_count[ts_idx] += 1
        
        # Check if valid
        if np.any(cover_count == 0):
            # Not valid yet, need to add blocks? 
            # In this strategy, we assume we start with a valid covering.
            # If not, we should probably not run local search or fix it first.
            # But the nibble should produce a valid covering.
            pass

        improved = True
        start_ls = time.time()
        
        while improved and time.time() < time_limit:
            improved = False
            # Try to remove a redundant block
            # A block is redundant if removing it doesn't leave any t-subset with count 0.
            # i.e., for all ts in block, cover_count[ts] > 1.
            
            # To speed up, we can pick a random block and check.
            # Or iterate through blocks.
            # Let's try random sampling to avoid getting stuck in local minima of order.
            
            # Shuffle a copy to randomize order
            blocks_to_check = list(current_blocks)
            random.shuffle(blocks_to_check)
            
            for b in blocks_to_check:
                if time.time() > time_limit:
                    break
                
                idx = block_to_idx[b]
                redundant = True
                for ts_idx in block_covers[idx]:
                    if cover_count[ts_idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block
                    current_blocks.remove(b)
                    for ts_idx in block_covers[idx]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart the loop to re-evaluate others
                    
        return list(current_blocks)

    # Rödl Nibble Strategy
    # We iterate through random blocks.
    # Probability of picking a block is proportional to its "gain" (number of uncovered t-subsets it covers).
    # However, calculating gain for all blocks each step is expensive.
    # Alternative: Randomly pick a block, if it covers any uncovered t-subset, add it.
    # This is the "Random Greedy" or "Nibble" variant.
    
    for seed in range(1000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        np.random.seed(seed)
        
        # uncovered mask
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks = []
        
        # Phase 1: Constructive
        # We don't know how many blocks we need.
        # We can stop when all are covered.
        # To make it faster, we can pick blocks that cover the most uncovered subsets.
        # But that's the greedy approach which we are trying to move away from (or complement).
        # The "Nibble" often picks uniformly at random from a pool, or weighted.
        
        # Let's try: 
        # 1. Pick a random uncovered t-subset.
        # 2. Pick a random block containing it.
        # 3. Add that block.
        # This ensures we make progress.
        
        # Precompute for each t-subset, the list of blocks that contain it.
        # t_subset_to_blocks = []
        # for ts_idx, ts in enumerate(all_t_subsets):
        #    ts_set = set(ts)
        #    # Find blocks containing ts_set
        #    # This is expensive to precompute fully.
        #    # But we can do it on the fly or precompute a sparse structure.
        
        # Let's precompute t_subset_to_blocks once.
        # C(19,5) = 11628 t-subsets.
        # For each, find k-t=3 points from v-t=14 remaining.
        # C(14,3) = 364 blocks per t-subset.
        # 11628 * 364 ~ 4.2M entries. Manageable.
        
        t_subset_to_blocks = []
        for ts in all_t_subsets:
            ts_set = set(ts)
            remaining = [x for x in range(v) if x not in ts_set]
            blocks_for_ts = []
            for comb in itertools.combinations(remaining, k - t):
                b = tuple(sorted(ts_set.union(comb)))
                blocks_for_ts.append(block_to_idx[b])
            t_subset_to_blocks.append(blocks_for_ts)
            
        # Now run the nibble
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset
            # To do this efficiently, we can maintain a list of uncovered indices?
            # Or just pick a random index and check if uncovered?
            # Since coverage decreases, the number of uncovered shrinks.
            # Initially 11628.
            
            # Efficient way: 
            # We can just pick a random index from 0..N-1.
            # If it's covered, try again.
            # This gets slow if few are uncovered.
            # Better: maintain a list of uncovered indices.
            
            # For simplicity in this implementation, let's just pick random index.
            # If it's covered, retry.
            # To avoid infinite loop if all covered but numpy check is lagging? No, np.any is accurate.
            
            while True:
                ts_idx = random.randint(0, num_t_subsets - 1)
                if uncovered[ts_idx]:
                    break
            
            # Pick a random block from those covering this t-subset
            candidate_blocks = t_subset_to_blocks[ts_idx]
            if not candidate_blocks:
                # Should not happen
                break
                
            b_idx = random.choice(candidate_blocks)
            
            # Add block
            # Check if block is already in blocks? 
            # If we add duplicates, it's wasteful.
            # We can keep a set of block indices.
            if not hasattr(construct, '_current_block_set'):
                construct._current_block_set = set()
            # Actually, let's just use a local set
            if b_idx not in blocks: # blocks is list of indices? No, let's keep indices for speed
                blocks.append(b_idx)
            
            # Update uncovered
            for ts_idx_cov in block_covers[b_idx]:
                uncovered[ts_idx_cov] = False
        
        # Phase 2: Local Search (Pruning)
        # Convert indices to tuples for local_search function
        blocks_tuples = [all_blocks[i] for i in blocks]
        
        # Allocate remaining time for local search
        time_remaining = deadline - time.time()
        if time_remaining > 1.0:
            # Run local search with a portion of the time, e.g., 50% of remaining
            # Or just run until time is up?
            # Let's give it a fair share.
            ls_deadline = time.time() + max(1.0, time_remaining * 0.5)
            pruned_blocks = local_search(blocks_tuples, ls_deadline)
        else:
            pruned_blocks = blocks_tuples
            
        if len(pruned_blocks) < best_count:
            best_count = len(pruned_blocks)
            best_blocks = pruned_blocks[:]
            
            if best_count <= 264:
                break
                
    if best_blocks is None:
        # Fallback: simple greedy
        # This shouldn't happen if the loop runs at least once
        # But just in case
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            if time.time() > deadline:
                break
            target = min(uncovered)
            fixed = set(target)
            candidates = [x for x in range(v) if x not in fixed]
            best_gain = -1
            best_block = None
            for comb in itertools.combinations(candidates, k - t):
                block = tuple(sorted(fixed.union(comb)))
                gain = sum(1 for ts in itertools.combinations(block, t) if ts in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            if best_block:
                blocks.append(best_block)
                for ts in itertools.combinations(best_block, t):
                    uncovered.discard(ts)
        best_blocks = blocks

    return best_blocks
