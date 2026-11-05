import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s kill
    
    # 1. Pre-computation
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Precompute numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    def run_greedy_full(random_seed=None, noise_scale=0.3):
        """
        Run a full greedy construction with random tie-breaking/noise.
        Returns list of block indices.
        """
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        # Initial gain is just the size of the block (all t-subsets within it)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        selected_blocks = []
        # Track which blocks are selected to avoid duplicates if needed, 
        # though greedy usually picks distinct ones.
        
        while True:
            if time.time() > deadline - 10:
                break
            if not np.any(uncovered):
                break
                
            # Add noise to break ties and escape local minima
            noise = rng.uniform(0, noise_scale, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_subset_arrays[best_idx]
            # Find which of the subsets in this block are currently uncovered
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            uncovered[newly_covered] = False
            
            # Update gains for all blocks that cover the newly covered subsets
            # A block's gain decreases by 1 for each subset it covers that is no longer uncovered
            for s in newly_covered:
                blocks_covering = subset_block_arrays[s]
                gains[blocks_covering] -= 1
                    
        return selected_blocks

    def minimize_with_deletions(blocks_indices, time_limit, random_seed=None):
        """
        Remove redundant blocks.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_subset_arrays[b_idx]] += 1
            
        improved = True
        start_t = time.time()
        
        while improved and (time.time() - start_t) < time_limit:
            improved = False
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(current_blocks):
                    continue
                b_idx = current_blocks[i]
                is_redundant = True
                b_subs = block_subset_arrays[b_idx]
                for s_idx in b_subs:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in b_subs:
                        cover_count[s_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break  # Restart scan after removal
                    
        return current_blocks

    def minimize_with_swaps(blocks_indices, time_limit, random_seed=None, max_swaps_per_pass=20):
        """
        Local search with swaps to reduce block count or improve coverage distribution.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_subset_arrays[b_idx]] += 1
            
        current_set = set(current_blocks)
        improved = True
        start_t = time.time()
        
        while improved and (time.time() - start_t) < time_limit:
            improved = False
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            swaps_done = 0
            for i in indices:
                if i >= len(current_blocks):
                    continue
                if swaps_done >= max_swaps_per_pass:
                    break
                if (time.time() - start_t) > time_limit:
                    break
                    
                old_block = current_blocks[i]
                old_subs = block_subset_arrays[old_block]
                
                # Find unique subsets covered by old_block (covered only by this block)
                unique_subs = []
                for s_idx in old_subs:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                        
                if not unique_subs:
                    # Block is not critical. Try swapping with a random block to see if it helps
                    # or just keep it. Actually, if it's not critical, we might have already removed it
                    # in the deletion pass. If it's here, it's not redundant, so every subset it covers
                    # is covered by at least one other block.
                    # We can try swapping it for a different non-redundant block?
                    # Or just skip.
                    continue
                else:
                    # Block is critical. Find a new block that covers all unique_subs.
                    # Candidate blocks must cover the intersection of unique_subs.
                    # To be efficient, pick the rarest unique subset (fewest covering blocks)
                    # and only consider blocks that cover it.
                    
                    # Sort unique subsets by frequency (ascending)
                    unique_subs_sorted = sorted(unique_subs, key=lambda s: len(subset_to_blocks[s]))
                    rarest_sub = unique_subs_sorted[0]
                    candidates = subset_to_blocks[rarest_sub]
                    
                    best_new_block = -1
                    best_new_cost = len(current_blocks) # We want to minimize total blocks, so cost is 0 for swap, but we want to maximize coverage of *other* uncovered things? 
                    # Actually, the goal of a swap is usually to enable further deletions.
                    # A good heuristic: swap to a block that covers the most "hard" subsets.
                    # Or simply: find a block that covers all unique_subs and has the highest "gain" in terms of subsets it covers that are currently covered exactly twice (making them potentially deletable for other blocks).
                    
                    # Let's just find a block that covers all unique_subs.
                    valid_candidates = []
                    for c_idx in candidates:
                        if c_idx == old_block or c_idx in current_set:
                            continue
                        c_subs = block_subset_arrays[c_idx]
                        # Check if c_idx covers all unique_subs
                        # This is an O(|unique_subs|) check
                        c_set = set(c_subs.tolist())
                        if all(s in c_set for s in unique_subs):
                            valid_candidates.append(c_idx)
                            
                    if not valid_candidates:
                        continue
                        
                    # Choose the best candidate.
                    # Heuristic: Maximize the number of subsets covered by the new block that are currently covered exactly once by *other* blocks? No, that makes them harder to cover.
                    # Heuristic: Maximize the number of subsets covered by the new block that are currently covered exactly twice. This might allow removing another block later.
                    
                    best_c = -1
                    best_score = -1
                    for c_idx in valid_candidates:
                        c_subs = block_subset_arrays[c_idx]
                        score = 0
                        for s_idx in c_subs:
                            # If this subset is covered exactly twice, and one of those is the old_block, then after swap it's still covered twice.
                            # If it's covered once by old_block and once by c_idx, it stays covered.
                            # We want to find a swap that creates redundancy elsewhere.
                            # Let's just pick the one with the highest total coverage (size of block) or random.
                            # Simpler: Random pick from valid candidates.
                            pass
                        
                        # For now, just pick the first one or random
                        if best_c == -1:
                            best_c = c_idx
                            best_score = 0
                        # Add a simple score: number of subsets in c_idx that are covered by >1 block (excluding old_block)
                        # This is expensive. Let's just randomize.
                        break # Just take the first valid candidate for speed, or randomize list first
                    
                    if best_c == -1:
                        continue
                            
                    # Perform swap
                    for s_idx in old_subs:
                        cover_count[s_idx] -= 1
                    new_subs = block_subset_arrays[best_c]
                    for s_idx in new_subs:
                        cover_count[s_idx] += 1
                        
                    current_set.discard(old_block)
                    current_set.add(best_c)
                    current_blocks[i] = best_c
                    swaps_done += 1
                    improved = True
                    break # Restart scan after swap
                    
        return current_blocks

    def spectral_partition_and_select():
        """
        Attempt to use spectral clustering on block overlap graph to guide selection.
        This is a heuristic to potentially find a better starting point or local structure.
        Given time constraints, we will run a few greedy instances with different noise
        and then refine.
        """
        pass

    # Main logic
    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy instances with different random seeds and noise levels
    # to diversify the search.
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        seed = i * 12345 + 678
        noise = 0.1 + 0.2 * (i % 3) # Vary noise
        
        # Run greedy
        blocks = run_greedy_full(random_seed=seed, noise_scale=noise)
        
        if not blocks:
            continue
            
        # Minimize with deletions
        blocks = minimize_with_deletions(blocks, time_limit=2.0, random_seed=seed)
        
        # Minimize with swaps
        blocks = minimize_with_swaps(blocks, time_limit=3.0, random_seed=seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = blocks
            
        # Early stop if we hit a very good number (e.g., close to theoretical lower bound if known, or just a good heuristic)
        # Lower bound for C(19,8,5) is 264. If we get below 300, that's great.
        if best_len <= 280:
            break

    # If no solution found (should not happen), fall back to simple greedy
    if best_solution is None:
        best_solution = run_greedy_full(random_seed=0, noise_scale=0.0)
        best_solution = minimize_with_deletions(best_solution, time_limit=5.0)

    # Final polish: try a few more swap passes on the best solution
    if time.time() < deadline - 5:
        best_solution = minimize_with_swaps(best_solution, time_limit=deadline - time.time() - 2, random_seed=99999)
        best_solution = minimize_with_deletions(best_solution, time_limit=1.0)

    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return final_blocks
