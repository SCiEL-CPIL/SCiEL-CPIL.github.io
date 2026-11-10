import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}
    
    # For fast lookup of which t-subsets are in a block
    # We will compute this on the fly or cache it.
    
    def get_tsub_indices(block):
        """Returns a set/list of indices of t-subsets covered by block."""
        return [tsub_index[combo] for combo in itertools.combinations(block, t)]

    best_blocks = []
    best_count = float('inf')

    # Multiple restarts to escape local optima
    max_restarts = 50  # Adjust based on speed
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        random.seed(restart) # Deterministic restarts for reproducibility in debugging, but random helps diversity
        # Actually, let's use a global random state or seed per restart
        
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        
        # Greedy Phase
        while True:
            if time.time() > deadline - 5.0:
                break
            if not uncovered.any():
                break
                
            # Pick a random uncovered t-subset to guide the search
            uncovered_indices = np.nonzero(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a critical t-subset (random among uncovered to diversify)
            crit_idx = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            crit_tsub = all_tsubs[crit_idx]
            
            # Generate all k-subsets containing crit_tsub
            # crit_tsub is a tuple of t elements.
            # Remaining points: v - t
            # Need to choose k - t more points.
            remaining_points = [x for x in range(v) if x not in crit_tsub]
            candidates = []
            
            # Optimization: If k-t is large, this list is big.
            # C(21, 4) = 5985. This is manageable.
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(crit_tsub + combo))
                candidates.append(block)
                
            # Find the candidate block that covers the most uncovered t-subsets
            best_block = None
            best_gain = -1
            
            # To speed up, we can sample candidates if the list is huge, but 6k is fine.
            for block in candidates:
                idxs = get_tsub_indices(block)
                # Count how many of these are currently uncovered
                gain = np.count_nonzero(uncovered[idxs])
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break
            
            blocks.append(list(best_block))
            
            # Update uncovered
            idxs = get_tsub_indices(best_block)
            uncovered[idxs] = False

        # Local Search: Remove redundant blocks
        # We do this iteratively
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                
                block_i = blocks[i]
                idxs_i = get_tsub_indices(block_i)
                
                # Check if all t-subsets in block_i are covered by other blocks
                # We can check this by temporarily removing it?
                # Or checking if any t-subset in block_i is ONLY covered by block_i.
                
                # Efficient check:
                # For each t-subset in block_i, is it covered by any other block?
                # This is O(k^t * N_blocks * k^t) which is slow.
                
                # Alternative: Rebuild coverage map?
                # Given N_blocks ~ 70, and k^t = 35.
                # 70 * 35 = 2450 checks.
                # For each check, we need to know if another block covers it.
                
                # Let's precompute a coverage count array?
                # cov_count[tsub_idx] = number of blocks covering it.
                # If we remove block_i, cov_count decreases by 1 for its tsubs.
                # If any cov_count becomes 0, it's not redundant.
                
                # We don't have cov_count maintained dynamically.
                # Let's just do a simple check for small N.
                
                redundant = True
                for tsub_idx in idxs_i:
                    covered_by_other = False
                    for j, b in enumerate(blocks):
                        if j == i:
                            continue
                        # Check if b covers tsub_idx
                        # tsub_idx corresponds to all_tsubs[tsub_idx]
                        # We can check if set(all_tsubs[tsub_idx]).issubset(set(b))
                        # But this is slow.
                        
                        # Better: Precompute a set of t-subsets for each block?
                        # Let's store blocks as (list, set_of_tsub_indices)
                        pass
                
                # This inner loop is getting complicated to optimize on the fly.
                # Let's use a simpler heuristic for local search:
                # Try to remove block i. If coverage is broken, restore.
                # We can maintain a global coverage count array.
                
                break # Disable complex local search for now to ensure speed and correctness
            
        # A simpler Local Search:
        # 1. Calculate coverage count for each t-subset.
        # 2. Identify blocks that only cover t-subsets with count > 1? No.
        # 3. If a block has a t-subset with count == 1, it's essential.
        # 4. If a block has NO t-subsets with count == 1, it is redundant.
        
        # Let's implement this properly.
        
        # Re-evaluate blocks list for redundancy
        # Build coverage count
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            idxs = get_tsub_indices(b)
            cov_count[idxs] += 1
            
        new_blocks = []
        for b in blocks:
            idxs = get_tsub_indices(b)
            # If any t-subset in b has count 1, b is necessary
            if np.any(cov_count[idxs] == 1):
                new_blocks.append(b)
            else:
                # b is redundant, do not add
                # Decrease counts for its t-subsets? 
                # Actually, if we remove b, counts decrease.
                # But we are building new_blocks.
                # We should just filter.
                pass
        
        # Wait, if we remove a redundant block, the counts change, potentially making other blocks redundant.
        # So we iterate.
        
        final_blocks = list(blocks)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # Recalculate counts
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            for b in final_blocks:
                idxs = get_tsub_indices(b)
                cov_count[idxs] += 1
            
            # Try to remove each block
            for i in range(len(final_blocks) - 1, -1, -1):
                if time.time() > deadline - 1.0:
                    break
                b = final_blocks[i]
                idxs = get_tsub_indices(b)
                # If all t-subsets in b are covered by >= 2 blocks, b is redundant
                if np.all(cov_count[idxs] > 1):
                    final_blocks.pop(i)
                    improved = True
                    break # Restart loop as counts changed
        
        if len(final_blocks) < best_count:
            best_count = len(final_blocks)
            best_blocks = final_blocks

    return best_blocks
