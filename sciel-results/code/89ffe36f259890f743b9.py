import itertools
import numpy as np
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s kill

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Build data structures for fast access
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    rows = []
    cols = []
    data = []

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    # Convert to numpy arrays for vectorized operations where possible
    # Note: subsets_of_block is a list of lists of different lengths, so we keep it as list of lists
    # but we can use numpy for the 'gains' and 'uncovered' arrays.
    
    # We don't need the full sparse matrix for the new approach, 
    # but we can use it for initial LP duals if we want.
    # However, to save time and memory, let's skip the LP and use a simpler heuristic for weights
    # or a random initialization.
    
    # For now, let's use a simple weight: number of subsets covered
    # This is a common greedy heuristic.
    base_block_weights = np.array([len(subsets_of_block[j]) for j in range(n_k)], dtype=np.float64)
    
    # Normalize weights to [0, 1]
    max_w = np.max(base_block_weights)
    if max_w > 0:
        base_block_weights /= max_w

    best_solution = None
    best_size = float('inf')

    # 2. Multi-start Greedy Search with Local Refinement
    # We use a greedy approach to build an initial solution, then refine it with local search.
    
    num_restarts = 5000 # Reduced to allow time for local search
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        # Initialize gains with base weights + noise
        # The noise helps escape symmetry and find different local optima
        if restart == 0:
            gains = base_block_weights.copy()
        else:
            # Add noise proportional to the scale of weights
            noise_scale = 0.1
            gains = base_block_weights + rng.uniform(0, noise_scale, n_k)

        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        
        # Greedy construction
        while True:
            if time.time() > deadline - 5:
                break

            # Find the block with the highest gain
            best_idx = np.argmax(gains)
            
            if gains[best_idx] <= 1e-9:
                # If max gain is ~0, we are stuck or done.
                if np.any(uncovered_mask):
                    # Fallback: pick any block that covers at least one uncovered subset
                    found = False
                    for j in range(n_k):
                        # Check if block j covers any uncovered subset
                        if any(uncovered_mask[s] for s in subsets_of_block[j]):
                            best_idx = j
                            found = True
                            break
                    if not found:
                        break
                else:
                    break

            # Pick best_idx
            solution_blocks.append(best_idx)
            
            # Update uncovered_mask
            subs_b = subsets_of_block[best_idx]
            newly_covered_indices = [s for s in subs_b if uncovered_mask[s]]
            
            if not newly_covered_indices:
                # Should not happen if gain > 0
                continue

            uncovered_mask[newly_covered_indices] = False
            
            # Update gains: subtract the weight of newly covered subsets
            # This is a simple approximation of the LP dual update
            gains[subs_b] -= 1.0  # Simplified update
            
            # Optimization: If all covered, break
            if not np.any(uncovered_mask):
                break

        # Check if solution is valid and better
        if len(solution_blocks) < best_size:
            # Verify coverage just in case
            if not np.any(uncovered_mask):
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()
                
                # Early exit if we hit a very good score?
                if best_size <= 151:
                    # Keep searching for a while
                    pass

    # 3. Local Search Refinement on Best Solution
    # Try to improve the best solution found by local search
    if best_solution is not None and time.time() < deadline - 2:
        # Convert to set for faster lookups
        current_blocks = set(best_solution)
        covered_mask = np.zeros(n_t, dtype=bool)
        
        # Compute coverage
        for b_idx in best_solution:
            covered_mask[subsets_of_block[b_idx]] = True
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Pass 1: Remove redundant blocks
            for b_idx in list(current_blocks):
                subs_b = subsets_of_block[b_idx]
                # Check if all subsets in this block are covered by other blocks
                can_remove = True
                for s_idx in subs_b:
                    # Count how many blocks cover this subset
                    cover_count = sum(1 for other_b in current_blocks if other_b != b_idx and s_idx in subsets_of_block[other_b])
                    if cover_count == 0:
                        can_remove = False
                        break
                
                if can_remove:
                    current_blocks.remove(b_idx)
                    improved = True
                    # Recompute coverage
                    covered_mask = np.zeros(n_t, dtype=bool)
                    for b_idx in current_blocks:
                        covered_mask[subsets_of_block[b_idx]] = True
            
            # Pass 2: Try swapping blocks
            if not improved:
                # For each block, try replacing with a better one
                blocks_list = list(current_blocks)
                for i, b_idx in enumerate(blocks_list):
                    # Find blocks that cover more uncovered subsets
                    # This is expensive, so we'll skip for now
                    pass

        # Update best solution
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_solution = list(current_blocks)

    # 4. Fallback if no solution found (should not happen)
    if best_solution is None:
        # Simple greedy without weights
        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        while np.any(uncovered_mask):
            best_idx = -1
            max_cov = -1
            for j in range(n_k):
                cnt = np.sum(uncovered_mask[subsets_of_block[j]])
                if cnt > max_cov:
                    max_cov = cnt
                    best_idx = j
            if best_idx == -1:
                break
            solution_blocks.append(best_idx)
            uncovered_mask[subsets_of_block[best_idx]] = False
        best_solution = solution_blocks
        best_size = len(solution_blocks)

    # 5. Post-processing: Deletion Pass
    # Try to remove blocks that are redundant.
    if best_solution is not None and time.time() < deadline - 1:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1

        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                # If any subset in this block is covered ONLY by this block, we can't remove it
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break

                if can_remove:
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)

            best_solution = new_solution
            best_size = len(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]
