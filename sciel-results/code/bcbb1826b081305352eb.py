import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute the list of t-subset indices for every possible k-subset (block)
    # There are C(19, 8) = 75,582 possible blocks.
    # Each block contains C(8, 5) = 56 t-subsets.
    # Total storage: 75582 * 56 ints. This is small.
    # Precomputing this avoids repeated calculation in greedy and local search.
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_tsubs[i] is a list of indices into all_tsubs
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    # Inverse mapping: for each t-subset, which blocks cover it?
    # tsub_blocks[j] is a list of indices into all_blocks
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
            
    # Deadline: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    def get_elapsed():
        return time.time() - start_time

    # Initial Construction: Weighted Greedy
    # We want to pick blocks that cover the most "hard" (uncovered) t-subsets.
    # A good heuristic is to pick the block that covers the most currently uncovered t-subsets.
    # To make it deterministic and robust, we can use a priority queue or just scan.
    # Scanning 75k blocks every step is too slow if we have ~300 steps (22M ops).
    # Better: Maintain a count of uncovered t-subsets covered by each block.
    # When we add a block, we decrement the "uncovered count" for its t-subsets.
    # Then we need to find the block with max coverage of remaining uncovered t-subsets.
    
    # Let's try a simpler greedy first: 
    # 1. Start with empty set of blocks.
    # 2. While there are uncovered t-subsets:
    #    a. Find an uncovered t-subset S.
    #    b. Among all blocks containing S, pick the one that covers the most OTHER uncovered t-subsets.
    #    c. Add that block.
    
    # This is the standard greedy set cover.
    # Optimization: 
    # - Maintain `uncovered` boolean array of size num_tsubs.
    # - When we need to pick a block for an uncovered t-subset `s_idx`:
    #   - Candidates are `tsub_blocks[s_idx]`.
    #   - For each candidate block `b_idx`, calculate `gain` = number of `j in block_tsubs[b_idx]` such that `uncovered[j]` is True.
    #   - Pick `b_idx` with max gain.
    
    # This might be slow if we do it naively.
    # Let's estimate:
    # num_tsubs = 11628.
    # Each t-subset is in C(19-5, 8-5) = C(14,3) = 364 blocks.
    # So candidates per step is ~364.
    # Gain calculation: iterate 56 t-subsets in block.
    # Cost per step: 364 * 56 ~ 20k ops.
    # Number of steps: ~335 (blocks in solution).
    # Total ops: 335 * 20k ~ 6.7M ops. This is very fast in Python (< 1s).
    
    def greedy_construct():
        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []
        
        while uncovered_count > 0:
            if get_elapsed() > deadline - 10.0:
                return selected_blocks # Return partial if timeout, but hopefully not needed
            
            # Find an uncovered t-subset
            # Randomly picking one helps break symmetry and might lead to different local minima?
            # But for a single run, deterministic is fine. Let's pick the first one found.
            # Or better: pick a random one to avoid bias?
            # Actually, standard greedy picks the element with the fewest covering sets? 
            # Or just any uncovered element.
            # Let's pick a random uncovered element to introduce some variety if we restart? 
            # But we are doing a single high-quality run.
            # Let's just find the first uncovered one.
            
            s_idx = -1
            for i in range(num_tsubs):
                if uncovered[i]:
                    s_idx = i
                    break
            
            if s_idx == -1:
                break
            
            # Find best block covering s_idx
            candidates = tsub_blocks[s_idx]
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                        # Early exit if gain > best_gain? No, we want max.
                        # But if gain == len(block_tsubs[b_idx]) (56), we can't do better.
                        if gain == 56:
                            break
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            # Add best_block_idx
            selected_blocks.append(best_block_idx)
            
            # Update uncovered
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        return selected_blocks

    # Local Search: Block Replacement and Removal
    # We have a list of block indices.
    # Goal: Reduce the number of blocks while maintaining coverage.
    
    def local_search(block_indices):
        # Convert to list of sets for fast membership? 
        # We need to check if removing a block breaks coverage.
        # We need to check if replacing a block with another improves things (fewer blocks? or same count but better position?).
        # Since we want to minimize count, we primarily look for removals.
        # Then we look for swaps that allow subsequent removals?
        
        # Step 1: Iterative Removal
        # Count how many times each t-subset is covered.
        cover_count = [0] * num_tsubs
        for b_idx in block_indices:
            for j in block_tsubs[b_idx]:
                cover_count[j] += 1
        
        changed = True
        while changed:
            changed = False
            new_block_indices = []
            for b_idx in block_indices:
                # Check if any t-subset in this block has cover_count == 1
                is_essential = False
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        is_essential = True
                        break
                
                if is_essential:
                    new_block_indices.append(b_idx)
                else:
                    # Remove it
                    changed = True
                    for j in block_tsubs[b_idx]:
                        cover_count[j] -= 1
            
            block_indices = new_block_indices
            
            if get_elapsed() > deadline - 5.0:
                break
                
        return block_indices

    # Step 2: Block Swap / Exchange
    # Try to replace a block B1 with a block B2 such that the new set is still a covering.
    # This doesn't reduce the count directly, but might enable a removal later?
    # Or we can look for a 2-for-1 replacement?
    # Given time constraints, simple removal is the most effective for reducing count from a greedy start.
    # Greedy usually produces ~340-350 blocks. Removal can shave off 5-10.
    
    # Let's run the greedy construction
    initial_blocks = greedy_construct()
    
    # Run local search (removal)
    optimized_blocks = local_search(initial_blocks)
    
    # We have time left. Can we do more?
    # Maybe try a few random restarts of the greedy phase if time permits?
    # The greedy phase is fast (~0.5s). Local search is also fast.
    # We have ~140s left.
    
    best_len = len(optimized_blocks)
    best_block_list = [all_blocks[i] for i in optimized_blocks]
    
    # Try a few more restarts with slight randomization in greedy
    # Randomize the order of candidate blocks or the selection of the starting t-subset.
    
    for restart in range(50):
        if get_elapsed() > deadline - 10.0:
            break
            
        # Randomized Greedy
        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []
        
        while uncovered_count > 0:
            if get_elapsed() > deadline - 10.0:
                break
            
            # Pick random uncovered t-subset
            # Optimization: maintain list of uncovered?
            # If uncovered_count is high, random probing is fine.
            # If low, build list.
            if uncovered_count < 1000:
                uncov_list = [i for i in range(num_tsubs) if uncovered[i]]
                if not uncov_list:
                    break
                s_idx = random.choice(uncov_list)
            else:
                while True:
                    s_idx = random.randint(0, num_tsubs - 1)
                    if uncovered[s_idx]:
                        break
            
            candidates = tsub_blocks[s_idx]
            best_block_idx = -1
            best_gain = -1
            
            # Randomize tie-breaking
            random.shuffle(candidates)
            
            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                        if gain == 56:
                            break
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            selected_blocks.append(best_block_idx)
            
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        if not selected_blocks:
            continue
            
        optimized = local_search(selected_blocks)
        
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_block_list = [all_blocks[i] for i in optimized]
            
            # Early exit if we hit a very good number
            if best_len <= 325:
                break

    # Final check: ensure validity (should be guaranteed by construction)
    # Return list of lists
    return [list(b) for b in best_block_list]
