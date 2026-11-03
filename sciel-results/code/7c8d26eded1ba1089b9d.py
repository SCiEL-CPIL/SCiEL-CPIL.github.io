import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)

    # Filling the array
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(t_in_k):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]
    
    # Convert to a list of arrays for faster indexing if needed, or keep as 2D
    # block_masks[i] is the array of t-indices covered by block i
    block_masks = [block_t_indices[i] for i in range(n_k)]

    # 4. Weighted Greedy Initialization
    # Calculate initial weights based on frequency
    t_counts = np.bincount(block_t_indices.ravel(), minlength=n_t)
    t_counts[t_counts == 0] = 1
    t_weights = 1.0 / t_counts.astype(np.float64)

    def greedy_init():
        uncovered = np.ones(n_t, dtype=bool)
        selected = []
        while True:
            if time.time() > deadline - 10.0:
                break
            if not uncovered.any():
                break
            # Calculate weighted gains
            # We want to sum weights of uncovered t-subsets covered by each block
            # gains[i] = sum(t_weights[block_masks[i][uncovered_indices_in_block]] )
            # This is tricky to vectorize directly without loops or heavy memory.
            # Approximation: Use a loop over blocks but vectorize the sum?
            # Or just use standard greedy: pick block covering most uncovered.
            
            # For speed, let's do a simple loop over a subset of candidates or all if fast enough.
            # n_k is 75k. n_t is 11k.
            # Calculating gains for all 75k blocks might be slow in Python loops.
            # Let's use NumPy:
            # For each block, we need to sum t_weights where uncovered is True.
            # We can precompute a boolean matrix? No, too big.
            
            # Alternative: Iterate through blocks, compute gain, track max.
            # 75k iterations * (56 ops) = 4M ops. This is fast in Python if simple.
            
            max_gain = -1
            best_idx = -1
            
            # Optimization: Only consider blocks that cover at least one uncovered t-subset?
            # Hard to check quickly.
            
            for i in range(n_k):
                mask = block_masks[i]
                # Sum weights of uncovered elements in this block
                # uncovered is a bool array of size n_t
                # We need sum(t_weights[mask] * uncovered[mask])
                # This is a dot product of t_weights[mask] and uncovered[mask]
                # Since mask is small (56), this is fast.
                gain = np.dot(t_weights[mask], uncovered[mask])
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            uncovered[block_masks[best_idx]] = False
            
        return selected

    # Run greedy init
    initial_selected = greedy_init()
    
    # Calculate coverage for initial solution
    coverage_counts = np.zeros(n_t, dtype=np.int32)
    for bi in initial_selected:
        coverage_counts[block_masks[bi]] += 1
        
    # Remove redundant blocks
    def remove_redundant(selected, coverage_counts):
        if not selected:
            return selected, coverage_counts
        
        # Create a set for O(1) lookup? Or just list.
        # We iterate and check if removing it keeps all counts > 0.
        # To do this efficiently, we can check if any count in block_masks[bi] is 1.
        # If min(coverage_counts[block_masks[bi]]) > 1, it's redundant.
        
        changed = True
        while changed:
            if time.time() > deadline - 5.0:
                break
            changed = False
            new_selected = []
            for bi in selected:
                if np.min(coverage_counts[block_masks[bi]]) > 1:
                    # Redundant, remove it
                    coverage_counts[block_masks[bi]] -= 1
                    changed = True
                else:
                    new_selected.append(bi)
            selected = new_selected
        return selected, coverage_counts

    initial_selected, coverage_counts = remove_redundant(initial_selected, coverage_counts)
    
    best_solution = initial_selected
    best_size = len(best_solution)
    
    # 5. Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Cost function: size + penalty * uncovered
    penalty = 100.0
    
    def get_cost(selected, coverage_counts):
        uncovered_count = np.sum(coverage_counts == 0)
        return len(selected) + penalty * uncovered_count

    current_selected = initial_selected
    current_coverage = coverage_counts.copy()
    current_cost = get_cost(current_selected, current_coverage)
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute random block indices for moves
    # We will pick random indices from 0 to n_k-1
    
    steps = 0
    while time.time() < deadline - 5.0 and T > min_T:
        steps += 1
        if steps % 1000 == 0 and time.time() > deadline - 10.0:
            break
            
        # Choose move type
        move_type = rng.random()
        
        new_selected = current_selected
        new_coverage = current_coverage
        
        if move_type < 0.3:
            # Remove a random block
            if not current_selected:
                continue
            idx_to_remove = rng.integers(0, len(current_selected))
            bi = current_selected[idx_to_remove]
            
            # Tentative removal
            new_coverage = current_coverage - 1  # Wait, we need to update only the block's parts
            # This creates a new array. Better to update in place and revert if needed?
            # Or just calculate the delta.
            
            # Let's do in-place update on a copy for clarity, or track changes.
            # Since n_t is small, copying is fast.
            new_coverage = current_coverage.copy()
            new_coverage[block_masks[bi]] -= 1
            
            new_selected = [b for i, b in enumerate(current_selected) if i != idx_to_remove]
            
        elif move_type < 0.6:
            # Add a random block
            bi = rng.integers(0, n_k)
            if bi in current_selected:
                continue
            new_coverage = current_coverage.copy()
            new_coverage[block_masks[bi]] += 1
            new_selected = current_selected + [bi]
            
        else:
            # Swap a random block with another random block
            if not current_selected:
                continue
            idx_to_swap = rng.integers(0, len(current_selected))
            bi_old = current_selected[idx_to_swap]
            bi_new = rng.integers(0, n_k)
            
            if bi_new in current_selected:
                continue
                
            new_coverage = current_coverage.copy()
            new_coverage[block_masks[bi_old]] -= 1
            new_coverage[block_masks[bi_new]] += 1
            
            new_selected = current_selected.copy()
            new_selected[idx_to_swap] = bi_new
            
        new_cost = get_cost(new_selected, new_coverage)
        
        # Acceptance criterion
        if new_cost < current_cost:
            current_selected = new_selected
            current_coverage = new_coverage
            current_cost = new_cost
            
            # Update best if valid
            if np.all(current_coverage > 0):
                # Try to remove redundants to update best
                # This is expensive, so only do it occasionally or if size decreased
                if len(current_selected) < best_size:
                    # Quick redundancy check
                    temp_sel, temp_cov = remove_redundant(current_selected, current_coverage.copy())
                    if len(temp_sel) < best_size:
                        best_size = len(temp_sel)
                        best_solution = temp_sel
                        # Update current state to the polished one?
                        # Yes, it's a better state.
                        current_selected = temp_sel
                        current_coverage = temp_cov
                        current_cost = get_cost(current_selected, current_coverage)
                        
        else:
            # Metropolis criterion
            delta = new_cost - current_cost
            if rng.random() < np.exp(-delta / T):
                current_selected = new_selected
                current_coverage = new_coverage
                current_cost = new_cost
        
        T *= cooling_rate
        
    # Final polish on the best solution found
    if best_solution:
        final_selected, final_coverage = remove_redundant(best_solution, np.zeros(n_t, dtype=np.int32))
        # Re-calculate coverage for final selected
        final_coverage = np.zeros(n_t, dtype=np.int32)
        for bi in final_selected:
            final_coverage[block_masks[bi]] += 1
            
        if np.all(final_coverage > 0):
            return [list(k_subsets[i]) for i in final_selected]
            
    # Fallback to initial if SA failed or produced invalid
    if initial_selected:
        return [list(k_subsets[i]) for i in initial_selected]
        
    return []
