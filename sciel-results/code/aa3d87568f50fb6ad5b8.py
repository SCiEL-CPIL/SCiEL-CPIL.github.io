import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Helper function for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures:
    # block_to_ts[j]: list of indices of t-subsets contained in block j
    # ts_to_blocks[i]: list of indices of blocks containing t-subset i
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        
    # Precompute the t-subsets for each block as a list of indices for faster access in local search
    # block_to_ts is already this.
    
    # State for greedy
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks = []
    
    # Greedy Phase (Deterministic)
    while True:
        if time.time() > deadline - 10.0:
            break
            
        max_idx = np.argmax(gain)
        if gain[max_idx] <= 0:
            break
            
        solution_blocks.append(list(all_k_blocks[max_idx]))
        is_selected[max_idx] = True
        gain[max_idx] = -1
        
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1
                    
    best_solution = solution_blocks
    best_len = len(solution_blocks)
    
    # Function to run a randomized greedy
    def randomized_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol

    # Run randomized greedy multiple times to find a better starting point
    # Each run is fast (~50-100ms), so we can do many.
    num_runs = 100
    for i in range(num_runs):
        if time.time() > deadline - 5.0:
            break
            
        sol = randomized_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Local Search: Remove redundant blocks
    def optimize_redundancy(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        for b in blocks:
            for ts_idx in block_to_ts[all_k_blocks.index(tuple(b))]: # This index lookup is slow.
                # Better to store the index of the block in the solution or just recompute.
                # Since blocks are lists, let's just recompute the ts indices.
                pass
                
        # Let's rewrite optimize_redundancy to be efficient.
        # We need to know which t-subsets are covered by which block in the current solution.
        # Let's build a set of (block_idx, ts_idx) or just count.
        
        # Rebuild cover counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        # Map solution blocks to their indices in all_k_blocks? 
        # It's easier to just iterate the solution blocks and update cover_count.
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 1.0:
                    return blocks
                
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index_map[ts]] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    for ts in itertools.combinations(block_to_remove, t):
                        cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
                    
        return blocks

    best_solution = optimize_redundancy(best_solution)
    
    # Local Search: 1-move optimization (Swap)
    # Try to replace a block B in the solution with a block B' not in the solution
    # such that the solution remains valid and potentially allows removing another block?
    # Or simply: if replacing B with B' allows us to keep the same size or reduce it?
    # Actually, a better local search is:
    # 1. Identify "critical" t-subsets (covered by exactly 1 block).
    # 2. For each block B, if it covers any critical t-subset, it cannot be removed.
    # 3. Try to swap a block B (that doesn't cover critical t-subsets? No, that's removal) 
    #    with a block B' that covers the same critical t-subsets?
    
    # A simpler and effective local search for covering designs:
    # "Pairwise Exchange": 
    # Pick a block B in the solution.
    # Pick a block B' not in the solution.
    # If (Solution - {B} + {B'}) is still a valid cover, and len(Solution) doesn't change, 
    # it doesn't help directly unless it enables further reductions.
    # However, if we can find a B' such that B' covers all t-subsets that B uniquely covers,
    # AND B' helps cover other t-subsets, we might be able to remove another block later.
    
    # Let's try a different local search: "Min-Cost Max-Flow" style is too complex.
    # Let's stick to "Randomized Local Search":
    # 1. Start with best_solution.
    # 2. Repeat:
    #    a. Pick a random block B from the solution.
    #    b. Remove it.
    #    c. Check if any t-subset becomes uncovered.
    #    d. If yes, we must add a new block B' that covers all newly uncovered t-subsets.
    #       To minimize size increase, we want B' to cover as many of these as possible.
    #       Actually, if removing B causes m t-subsets to be uncovered, we need to add
    #       at least one block that covers all of them? No, we can add multiple blocks.
    #       But if we add 1 block, size stays same. If we add 0, size decreases (handled by redundancy).
    #       So, we look for a block B' that covers ALL t-subsets that were uniquely covered by B.
    #       If such a B' exists, we can swap B -> B'. This keeps size same.
    #       Does it help? It might change the "uniqueness" pattern, allowing future removals.
    
    def local_search_swap(blocks):
        # Precompute unique coverage for current blocks
        # unique_ts[block_idx] = set of t-subsets covered ONLY by this block
        # This is expensive to compute from scratch every step.
        
        # Let's use the cover_count array.
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        improved = False
        # Try swapping each block
        for i in range(len(blocks)):
            if time.time() > deadline - 1.0:
                return blocks, improved
            
            # Find t-subsets uniquely covered by blocks[i]
            unique_ts = []
            for ts in itertools.combinations(blocks[i], t):
                idx = ts_index_map[ts]
                if cover_count[idx] == 1:
                    unique_ts.append(idx)
                    
            if not unique_ts:
                continue # Redundant, should have been removed by optimize_redundancy
            
            # We need to find a block B' (not in blocks) that covers all ts in unique_ts.
            # And B' should be different from blocks[i].
            
            # To speed this up, we can iterate through all_k_blocks? 75k is okay if we do it a few times.
            # But checking "covers all" is expensive.
            
            # Alternative: Just try random blocks?
            # Or, find the intersection of ts_to_blocks for each ts in unique_ts.
            # The candidate blocks must be in the intersection of ts_to_blocks[ts] for all ts in unique_ts.
            
            # Get candidates from the first unique ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            # Remove the current block itself
            # We need the index of blocks[i] in all_k_blocks.
            # Since blocks are lists, we can't easily map back to index without a dict.
            # Let's build a dict of tuple(block) -> index.
            
            # This is getting complex. Let's simplify.
            # If unique_ts is small, the intersection will be small.
            
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
                
            # Remove the current block's index from candidates?
            # We don't have the index handy. But if we pick a candidate that IS the current block,
            # the swap is a no-op. We should avoid that.
            
            # Let's just pick a random candidate from the intersection.
            # If it's the same block, we skip.
            
            # To check if a candidate is the same as blocks[i], we can compare sets.
            # But candidates are indices.
            # Let's build a map from tuple(block) to index for the current solution blocks.
            # This is O(N) per iteration, which is fine for N~300.
            
            block_indices_in_sol = {}
            for j, b in enumerate(blocks):
                block_indices_in_sol[tuple(b)] = j
                
            # Now filter candidates
            valid_candidates = []
            for c_idx in candidates:
                if tuple(all_k_blocks[c_idx]) not in block_indices_in_sol:
                    valid_candidates.append(c_idx)
            
            if not valid_candidates:
                continue
                
            # Pick one randomly
            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            
            # Perform swap
            old_block = blocks[i]
            blocks[i] = new_block
            
            # Update cover counts
            # Decrement for old_block
            for ts in itertools.combinations(old_block, t):
                cover_count[ts_index_map[ts]] -= 1
            # Increment for new_block
            for ts in itertools.combinations(new_block, t):
                cover_count[ts_index_map[ts]] += 1
            
            # Check if validity is maintained.
            # Since we ensured new_block covers all unique_ts of old_block,
            # and other ts were covered by >=2 blocks before (so >=1 after removing old_block),
            # validity is maintained.
            
            improved = True
            
            # After a swap, run redundancy removal again?
            # Yes, the swap might have made another block redundant.
            # But that's expensive to do inside the loop.
            # We'll do a final pass of optimize_redundancy after the local search loop.
            
        return blocks, improved

    # Run local search for a few iterations
    for _ in range(20):
        if time.time() > deadline - 2.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if not improved:
            break
        best_solution = optimize_redundancy(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    return best_solution
