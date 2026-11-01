import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using a Local Search approach.
    1. Generate a valid covering using a fast greedy heuristic.
    2. Iteratively minimize the number of blocks by removing redundant blocks.
    3. If stuck, perturb the solution by swapping a block and re-minimize.
    """
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets and their covered t-subset indices
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # cover_matrix[i] contains the indices of t-subsets covered by block i
    # Shape: (n_k_subsets, C(k,t))
    cover_matrix_list = []
    for block in all_k_subsets:
        indices = [t_sub_index[ts] for ts in itertools.combinations(block, t)]
        cover_matrix_list.append(indices)
    
    cover_matrix = np.array(cover_matrix_list, dtype=np.int32)
    c_k_t = cover_matrix.shape[1]
    
    # Helper: Calculate coverage counts for a given set of block indices
    def get_coverage_counts(block_indices):
        # block_indices: list of indices into all_k_subsets
        if not block_indices:
            return np.zeros(n_t_subsets, dtype=np.int32)
        # Sum the coverage vectors
        # This creates an array of shape (len(block_indices), c_k_t)
        # Then we need to count occurrences of each triple index.
        # Using np.bincount is efficient.
        flat_indices = cover_matrix[block_indices].flatten()
        counts = np.bincount(flat_indices, minlength=n_t_subsets)
        return counts.astype(np.int32)
    
    # Helper: Check if a set of block indices is a valid cover
    def is_valid_cover(block_indices):
        counts = get_coverage_counts(block_indices)
        return np.all(counts > 0)
        
    # Helper: Reduce the set of blocks by removing redundant ones
    # Returns the reduced list of block indices
    def reduce_blocks(block_indices):
        # Sort indices for deterministic behavior? Or keep order.
        # We iterate and remove if redundant.
        # To be efficient, we re-calculate counts after each removal? 
        # That's O(N^2 * C(k,t)). N~80, C(k,t)=35. 80*80*35 ~ 224k ops. Very fast.
        
        current_blocks = list(block_indices)
        # We can speed this up by maintaining counts dynamically, but for N=80, recalculation is fine.
        
        changed = True
        while changed:
            changed = False
            counts = get_coverage_counts(current_blocks)
            # Iterate through a copy of the list to allow removal
            # We want to remove a block if ALL its triples have count > 1
            for i in range(len(current_blocks) - 1, -1, -1):
                b_idx = current_blocks[i]
                b_triples = cover_matrix[b_idx]
                # Check if all counts for this block's triples are > 1
                if np.all(counts[b_triples] > 1):
                    current_blocks.pop(i)
                    changed = True
                    # Break to recalculate counts immediately? 
                    # Or continue? If we continue, counts are stale for subsequent checks in this pass.
                    # It's safer to recalculate if we remove. 
                    # However, removing one block might make another redundant.
                    # Recalculating after each removal is O(N) per removal.
                    # Total O(N^2). Acceptable.
                    break
        
        return current_blocks

    # Phase 1: Fast Greedy to get a valid starting point
    def greedy_cover():
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks_found = []
        
        # Precompute gains for all blocks? 
        # We just need to pick the best one each time.
        # For speed, we can just iterate.
        
        while np.any(uncovered):
            best_gain = 0
            best_idx = -1
            # Iterate through all blocks to find max gain
            # This is O(N_blocks * C(k,t)) per step.
            # 346k * 35 ~ 12M ops per step. 80 steps -> 1B ops. Too slow in Python loop.
            # We need to vectorize the gain calculation.
            
            # gains = np.sum(uncovered[cover_matrix], axis=1)
            # This is the same vectorization as the original program.
            # Since we are in a "new approach" directive, using vectorized greedy for *initialization* is fine, 
            # as the core improvement is the Local Search minimization.
            
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[random.randint(0, len(candidates)-1)]
            blocks_found.append(chosen_idx)
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks_found

    # Main Logic
    start_time = time.time()
    timeout = 115
    
    best_blocks = []
    best_size = float('inf')
    
    # Initial Solution
    initial_blocks = greedy_cover()
    initial_blocks = reduce_blocks(initial_blocks)
    best_blocks = initial_blocks
    best_size = len(initial_blocks)
    
    # Local Search Loop
    # We will try to improve the best found solution.
    # Strategy: Start from the best solution, perturb, reduce, if better, update best.
    # Or start from current, perturb, reduce.
    
    # To diversify, we can restart from the best solution every few iterations or use a temperature-like cooling.
    # Simple approach: Always start perturbation from the best known solution.
    
    while time.time() - start_time < timeout:
        # Create a copy of the best blocks to perturb
        current_blocks = list(best_blocks)
        
        if not current_blocks:
            break
            
        # Perturbation: Swap one block with a random other block
        # Pick a random index in current_blocks
        swap_idx = random.randint(0, len(current_blocks) - 1)
        old_block_idx = current_blocks[swap_idx]
        
        # Pick a new block from all_k_subsets
        # Avoid picking the same one
        new_block_idx = random.randint(0, n_k_subsets - 1)
        if new_block_idx == old_block_idx:
            continue
            
        current_blocks[swap_idx] = new_block_idx
        
        # Reduce
        reduced_blocks = reduce_blocks(current_blocks)
        
        # Check validity (reduce_blocks doesn't guarantee validity if the swap broke coverage and no removals fixed it? 
        # Wait, reduce_blocks only REMOVES. It doesn't ADD.
        # If the swap broke coverage, reduced_blocks will still be invalid (some triples count 0).
        # We must check validity before accepting.
        
        if len(reduced_blocks) < best_size:
            # Check if it's actually valid
            if is_valid_cover(reduced_blocks):
                best_blocks = reduced_blocks
                best_size = len(best_blocks)
                
        # Optional: If we are stuck for a long time, try a stronger perturbation (multiple swaps)
        # Or just keep swapping single blocks.
        
    # Convert indices back to lists of points
    final_blocks = [list(all_k_subsets[i]) for i in best_blocks]
    
    return final_blocks
