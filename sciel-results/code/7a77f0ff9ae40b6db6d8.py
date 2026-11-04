import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)  # 11628

    # Map each t-subset to an index
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)  # 75582

    # Precompute t-subsets for each block
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)

    # Inverse: for each t-subset, which blocks cover it?
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)

    # NumPy arrays for fast operations
    block_tsubs_np = np.array(block_tsubs, dtype=np.int32)
    
    # Deadline
    start_time = time.time()
    deadline = start_time + 145.0

    def get_elapsed():
        return time.time() - start_time

    def check_time(budget=10.0):
        return get_elapsed() > deadline - budget

    def luby_greedy(random_seed=None):
        """
        Luby's weighted greedy algorithm.
        We maintain a weight for each uncovered t-subset.
        At each step, we pick a random uncovered t-subset, 
        then pick the block that covers the most high-weight uncovered t-subsets.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random

        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        uncovered_count = num_tsubs
        weights = np.ones(num_tsubs, dtype=np.float32)
        selected_blocks = []

        # Luby's sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
        # We can approximate by increasing weights of uncovered sets.
        
        while uncovered_count > 0:
            if check_time(15.0):
                return None

            # Pick a random uncovered t-subset
            # For efficiency, if uncovered_count is small, build a list
            if uncovered_count < 500:
                uncov_indices = np.where(uncovered == 1)[0]
                if len(uncov_indices) == 0:
                    break
                s_idx = int(rng.choice(uncov_indices))
            else:
                while True:
                    s_idx = rng.randint(0, num_tsubs - 1)
                    if uncovered[s_idx]:
                        break

            # Get candidate blocks
            candidates = np.array(tsub_blocks[s_idx], dtype=np.int32)
            
            if len(candidates) == 0:
                break

            # Compute weighted gain for each candidate block
            # Gain = sum of weights of uncovered t-subsets in the block
            cand_tsubs = block_tsubs_np[candidates]
            # Mask with uncovered
            gains = np.sum(weights[cand_tsubs] * uncovered[cand_tsubs], axis=1)

            # Find best
            best_local = int(np.argmax(gains))
            best_block_idx = int(candidates[best_local])

            selected_blocks.append(best_block_idx)

            # Update uncovered and weights
            # Increase weights of uncovered t-subsets not covered by this block? 
            # Or just increase all uncovered? 
            # Standard Luby: increase weight of the picked element.
            # Here, we can increase weights of all uncovered t-subsets by 1.
            # Or, more effectively, increase weights of uncovered t-subsets 
            # that are NOT covered by the selected block to push them higher.
            
            # Let's use a simpler heuristic: 
            # Increase weight of all uncovered t-subsets by 1.
            # This makes older uncovered items more "expensive" to leave alone.
            weights[uncovered == 1] += 1.0
            
            # Update uncovered
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1

        if uncovered_count > 0:
            return None

        return selected_blocks

    def deep_local_search(block_indices):
        """
        Perform intensive local search to reduce block count.
        1. Remove redundant blocks.
        2. Try 2-for-1 swaps: Remove 2 blocks, add 1 block.
        3. Try 1-for-1 swaps that enable further removals.
        """
        block_indices = list(block_indices)
        
        # Helper to check if a set of block indices covers all t-subsets
        def is_valid(indices):
            if not indices:
                return num_tsubs == 0
            # Use a temporary cover count
            # This is expensive if done naively. 
            # Instead, we maintain a global cover count in the main loop.
            pass

        # Build initial cover count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs_np[b_idx]] += 1
        
        # Map from t-subset to list of blocks covering it (for quick lookup of alternatives)
        # We already have tsub_blocks, but it's a list of lists.
        # For speed in local search, we might need to update this dynamically, 
        # which is hard. So we rely on cover_count.

        changed = True
        max_passes = 50
        pass_count = 0
        
        while changed and pass_count < max_passes:
            if check_time(10.0):
                break
            pass_count += 1
            changed = False
            
            # Step 1: Remove redundant blocks
            new_block_indices = []
            for b_idx in block_indices:
                b_tsubs = block_tsubs_np[b_idx]
                # If any t-subset in this block has cover_count == 1, it's essential
                if np.any(cover_count[b_tsubs] == 1):
                    new_block_indices.append(b_idx)
                else:
                    # Remove it
                    changed = True
                    cover_count[b_tsubs] -= 1
            
            block_indices = new_block_indices
            
            if not block_indices:
                break

            # Step 2: Try 2-for-1 swaps
            # For each pair of blocks (i, j), check if there exists a block z 
            # such that removing i and j and adding z maintains coverage.
            # This is O(N^2 * C) which is too slow.
            # Heuristic: Only try pairs that are "similar" or cover overlapping rare sets.
            # Simpler heuristic: 
            # For each block i, find its "unique" t-subsets (cover_count == 1).
            # If block i has no unique t-subsets, it was already removed.
            # So every block in the list has at least one unique t-subset.
            # Let U_i be the set of unique t-subsets for block i.
            # We want to find j and z such that U_i U U_j is covered by z.
            # And z must also cover any other t-subsets that i or j covered uniquely? 
            # No, if a t-subset is unique to i, and we remove i, it must be covered by z.
            # So z must contain U_i. Similarly for j.
            # So z must contain U_i U U_j.
            
            # Optimization: 
            # Precompute for each block the bitmask of its unique t-subsets? 
            # Too big. 
            # Instead, iterate through blocks.
            
            # To keep it fast, we'll do a limited search.
            # Pick a random block i.
            # Get its unique t-subsets U_i.
            # If |U_i| is large, it's hard to replace.
            # Look for another block j that shares some unique t-subsets? No, they are unique to i and j respectively.
            
            # Alternative: Try to replace block i with a block z that covers U_i 
            # and allows the removal of another block j.
            # This is complex.
            
            # Let's stick to a simpler but effective move: 
            # "Greedy Swap": 
            # For each block i, try replacing it with a block z that covers all 
            # unique t-subsets of i, and see if that allows removing another block.
            
            # Given time constraints, we will do a few random swap attempts.
            # Pick 10 random blocks.
            for _ in range(10):
                if check_time(5.0):
                    break
                if not block_indices:
                    break
                i_idx = rng.choice(block_indices)
                i_tsubs = block_tsubs_np[i_idx]
                # Find unique t-subsets of i
                # unique_mask = (cover_count[i_tsubs] == 1)
                # We need a block z that covers all i_tsubs where cover_count == 1.
                # Let required = [ts for ts in i_tsubs if cover_count[ts] == 1]
                # If required is empty, i is redundant (should have been removed).
                
                # Find candidate z's that cover all required.
                # Intersect tsub_blocks for each ts in required.
                if not required:
                    continue
                
                # Start with blocks covering the first required ts
                candidates = set(tsub_blocks[required[0]])
                for ts in required[1:]:
                    candidates &= set(tsub_blocks[ts])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Pick a random candidate z
                z_idx = rng.choice(list(candidates))
                
                # Simulate: Remove i, Add z
                # Check if this allows removing another block.
                # Temporarily update cover_count
                old_cover = cover_count.copy()
                
                # Remove i
                cover_count[i_tsubs] -= 1
                # Add z
                z_tsubs = block_tsubs_np[z_idx]
                cover_count[z_tsubs] += 1
                
                # Check if any other block j can be removed
                # A block j can be removed if all its t-subsets have cover_count > 1 
                # (after the swap).
                # Note: cover_count reflects the state with i removed and z added.
                # But we haven't removed i from the list yet.
                
                removable_j = None
                for j_idx in block_indices:
                    if j_idx == i_idx:
                        continue
                    j_tsubs = block_tsubs_np[j_idx]
                    # If all t-subsets in j are covered by others (count > 1), 
                    # then j is redundant in the new configuration.
                    if np.all(cover_count[j_tsubs] > 1):
                        removable_j = j_idx
                        break
                
                if removable_j is not None:
                    # We can remove i and removable_j, and add z.
                    # Net change: -2 blocks, +1 block = -1 block.
                    # Apply the swap
                    block_indices.remove(i_idx)
                    block_indices.remove(removable_j)
                    block_indices.append(z_idx)
                    
                    # Update cover_count for the removed blocks
                    # We already subtracted i and added z.
                    # Now subtract removable_j
                    cover_count[block_tsubs_np[removable_j]] -= 1
                    
                    changed = True
                else:
                    # Rollback
                    cover_count = old_cover

        return block_indices

    # Main loop: multiple restarts
    best_len = float('inf')
    best_block_list = []

    # First, run a deterministic greedy to get a baseline
    initial = luby_greedy(random_seed=42)
    if initial is not None:
        reduced = deep_local_search(initial)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_block_list = [all_blocks[i] for i in reduced]

    # Now, run randomized restarts
    num_restarts = 100
    for restart in range(num_restarts):
        if check_time(15.0):
            break

        seed = restart * 1000 + 7
        selected = luby_greedy(random_seed=seed)
        if selected is None:
            continue

        reduced = deep_local_search(selected)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_block_list = [all_blocks[i] for i in reduced]

            # Early exit if we hit a very good number
            if best_len <= 320:
                break

    # If we still have time, do more aggressive reduction on the best solution
    if best_block_list and not check_time(20.0):
        # Convert best_block_list back to indices
        # This is slow if we use .index(). 
        # We should have kept the indices. 
        # Let's assume best_block_list contains indices from the last run? 
        # No, it contains the blocks (tuples).
        # Let's create a map from block tuple to index.
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        best_indices = [block_to_idx[b] for b in best_block_list]
        
        # Run deep_local_search again on the best
        reduced = deep_local_search(best_indices)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_block_list = [all_blocks[i] for i in reduced]

    # Return the best solution
    return [list(b) for b in best_block_list]
