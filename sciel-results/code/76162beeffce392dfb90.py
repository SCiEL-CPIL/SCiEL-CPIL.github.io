import itertools
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill

    # 1. Precompute all t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Precompute all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    n_blocks = len(k_subsets)

    # For each t-subset, find all blocks that contain it
    # blocks_containing[t_idx] = list of block indices
    blocks_containing = defaultdict(list)
    # For each block, compute the list of t-subsets it contains
    block_to_tsubsets = []
    
    for b_idx, block in enumerate(k_subsets):
        contained_tsubs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            contained_tsubs.append(idx)
            blocks_containing[idx].append(b_idx)
        block_to_tsubsets.append(contained_tsubs)

    # 3. Greedy set cover with vectorized gain computation
    # uncovered: boolean array of length num_t_subsets
    uncovered = np.ones(num_t_subsets, dtype=bool)
    selected_blocks = []
    
    # For efficiency, we'll track gains differently
    # Instead of checking all blocks each iteration (50k blocks * 150 iterations = 7.5M checks),
    # we can be smarter. But let's first try the direct approach with numpy.
    
    # Precompute for each block: which t-subsets does it cover?
    # We already have block_to_tsubsets as lists.
    # Let's create a sparse representation.
    
    # Actually, let's just do the greedy properly.
    # Each iteration: for each block, count how many of its t-subsets are still uncovered.
    
    # To speed this up, we can precompute a bitset or use numpy.
    # Given num_t_subsets = C(19,4) = 3876, a bitset is feasible.
    # But 50388 blocks * 3876 bits = ~24M bits = 3MB per iteration? No, we just need to count.
    
    # Let's use a different approach: for each t-subset, track how many selected blocks cover it.
    # And maintain a list of uncovered t-subsets.
    
    # Actually, the standard greedy for set cover:
    # - Maintain uncovered set
    # - For each candidate set, compute |candidate ∩ uncovered|
    # - Pick the one with maximum intersection
    # - Update uncovered
    
    # With 50388 candidates and ~3876 uncovered items, this is expensive per iteration.
    
    # Alternative: use the dual formulation.
    # For each t-subset, we know which blocks cover it.
    # Initially, all t-subsets are uncovered.
    # When we select a block, we "satisfy" all t-subsets it covers.
    
    # Let's try a simpler optimization:
    # 1. Sort blocks by some heuristic (e.g., random or by size of overlap with others)
    # 2. Use a priority queue or just iterate
    
    # Given time constraints, let's use a randomized greedy with multiple restarts if time permits.
    # First, let's get a good greedy solution.
    
    # We'll use the following approach:
    # - Maintain an array cover_count[i] = number of selected blocks covering t-subset i
    # - uncovered[i] = (cover_count[i] == 0)
    # - When selecting a block, increment cover_count for all t-subsets in that block
    # - Gain of a block = sum of uncovered[i] for i in block
    
    # To compute gain efficiently, we can precompute for each block the list of t-subsets.
    # Then gain = sum(uncovered[i] for i in block_to_tsubsets[b_idx])
    
    # This is still O(|block_to_tsubsets[b_idx]|) per block per iteration.
    # |block_to_tsubsets[b_idx]| = C(7,4) = 35 for each block.
    # So per iteration: 50388 * 35 = ~1.7M operations.
    # For 150 iterations: 255M operations. This might be too slow in Python.
    
    # Let's use numpy to vectorize.
    # Create a matrix M where M[b_idx, t_idx] = 1 if block b_idx covers t-subset t_idx.
    # This is 50388 x 3876 = ~195M entries. Too large for memory? 195M * 1 byte = 195MB. Maybe ok.
    # But creating it is slow.
    
    # Alternative: use the blocks_containing structure.
    # For each t-subset, we know which blocks cover it.
    # We can maintain for each block a count of how many of its t-subsets are uncovered.
    # block_gain[b_idx] = number of uncovered t-subsets in block b_idx
    
    # When we select a block, we update block_gain for all blocks that share t-subsets with it.
    # This is complex.
    
    # Let's just use a straightforward approach with some optimization:
    # 1. Start with all t-subsets uncovered.
    # 2. For each iteration, compute gain for all blocks.
    # 3. Pick the best.
    # 4. Update uncovered.
    
    # To speed up, we can use the fact that only blocks that cover newly covered t-subsets
    # need their gains recomputed. But that's complex.
    
    # Let's try a different strategy: local search / simulated annealing on a subset of blocks.
    # Or use a genetic algorithm.
    
    # Actually, let's first try the simple greedy and see if it finishes in time.
    # If not, we'll optimize.
    
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    uncovered = np.ones(num_t_subsets, dtype=bool)
    uncovered_count = num_t_subsets
    
    selected_blocks = []
    
    for iteration in range(200):  # Max 200 iterations
        if time.time() > deadline - 10:  # Leave 10s margin
            break
        if uncovered_count == 0:
            break
            
        # Compute gain for each block
        # gain[b_idx] = sum(uncovered[i] for i in block_to_tsubsets[b_idx])
        
        # We'll compute this in a loop, but try to be efficient.
        best_gain = 0
        best_idx = -1
        
        # Optimization: if time is tight, we can sample blocks instead of checking all.
        # But let's first try checking all.
        
        for b_idx in range(n_blocks):
            tsubs = block_to_tsubsets[b_idx]
            # Sum of uncovered for these t-subsets
            gain = 0
            for t_idx in tsubs:
                if uncovered[t_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = b_idx
            # Early termination if gain == best possible
            # Best possible is at most len(tsubs) = 35
            # But uncovered_count might be less.
            # If gain == min(uncovered_count, 35), we can't do better.
            # But we don't know if there's a block with exactly that gain.
            
        if best_gain == 0:
            # No block can cover any uncovered t-subset? Should not happen.
            break
        
        # Select the best block
        selected_blocks.append(best_idx)
        
        # Update uncovered
        for t_idx in block_to_tsubsets[best_idx]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                uncovered_count -= 1
        
        if time.time() > deadline - 5:
            break
    
    # Check if we have a valid covering
    if uncovered_count > 0:
        # We didn't finish in time. Try a different approach.
        # Use a randomized local search or return what we have (invalid, so score 0).
        # Let's try a quick repair: for each uncovered t-subset, add a block that covers it.
        # This might add many blocks but ensures validity.
        
        for t_idx in range(num_t_subsets):
            if uncovered[t_idx]:
                # Find any block that covers this t-subset
                # blocks_containing[t_idx] has all blocks that cover it
                if blocks_containing[t_idx]:
                    # Pick the first one (or random)
                    b_idx = blocks_containing[t_idx][0]
                    if b_idx not in selected_blocks:
                        selected_blocks.append(b_idx)
                        for t_i in block_to_tsubsets[b_idx]:
                            uncovered[t_i] = False
        
        # This is a valid covering now, but might have many blocks.
        # Let's try to clean it up with a simple local search.
        # Actually, given time constraints, let's just return this.
        # But we can do better: let's use the greedy solution as a starting point
        # and try to remove blocks.
        
        # For now, return the greedy solution if it's valid, otherwise the repaired one.
    
    # Convert to list of blocks
    result = [list(k_subsets[i]) for i in selected_blocks]
    
    # Try to improve with local search: remove blocks if possible
    # This is a simple post-processing step.
    # For each block, check if removing it still leaves all t-subsets covered.
    # This is expensive, so we'll do it for a subset of blocks.
    
    # Build coverage count
    cover_count_final = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_blocks:
        for t_idx in block_to_tsubsets[b_idx]:
            cover_count_final[t_idx] += 1
    
    # Try to remove blocks that are "redundant"
    # A block is redundant if all its t-subsets are covered by other blocks.
    # i.e., for all t_idx in block_to_tsubsets[b_idx], cover_count_final[t_idx] > 1
    
    improved_selected = list(selected_blocks)
    for b_idx in selected_blocks:
        redundant = True
        for t_idx in block_to_tsubsets[b_idx]:
            if cover_count_final[t_idx] <= 1:
                redundant = False
                break
        if redundant:
            # Remove this block
            improved_selected.remove(b_idx)
            for t_idx in block_to_tsubsets[b_idx]:
                cover_count_final[t_idx] -= 1
    
    # Convert to final result
    result = [list(k_subsets[i]) for i in improved_selected]
    
    return result
