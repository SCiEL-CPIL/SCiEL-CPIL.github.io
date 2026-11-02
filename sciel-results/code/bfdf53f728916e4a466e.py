import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_triples = len(all_t_subsets)
    triple_to_idx = {triple: i for i, triple in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets and their coverage of t-subsets
    all_k_subsets = list(itertools.combinations(all_points, k))
    n_k_subsets = len(all_k_subsets)
    
    # cover_matrix[i] contains the indices of t-subsets covered by all_k_subsets[i]
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    cover_matrix = np.zeros((n_k_subsets, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_k_subsets):
        for j, triple in enumerate(itertools.combinations(block, t)):
            cover_matrix[i, j] = triple_to_idx[triple]
            
    # Precompute triple_to_blocks: for each triple, which blocks cover it?
    # This is useful for swap local search.
    # triple_to_blocks[i] is a numpy array of block indices.
    triple_to_blocks = [np.array([], dtype=np.int32) for _ in range(n_triples)]
    for i in range(n_k_subsets):
        blocks_covering = cover_matrix[i]
        for triple_idx in blocks_covering:
            # Append i to triple_to_blocks[triple_idx]
            # Since we are in a loop, we can just collect indices first or use a list of lists then convert
            pass
    
    # Efficient way to build triple_to_blocks
    # Using a list of lists first is faster in Python for appending, then convert to numpy
    t2b_lists = [[] for _ in range(n_triples)]
    for i in range(n_k_subsets):
        for triple_idx in cover_matrix[i]:
            t2b_lists[triple_idx].append(i)
    triple_to_blocks = [np.array(lst, dtype=np.int32) for lst in t2b_lists]

    best_blocks = None
    best_len = float('inf')
    
    def fast_greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            blocks.append(chosen_idx)
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks

    def local_search(block_indices):
        # block_indices is a list of indices into all_k_subsets
        if not block_indices:
            return block_indices
            
        # Use a set for O(1) lookup of current blocks
        current_set = set(block_indices)
        
        # Initialize coverage counts
        coverage_counts = np.zeros(n_triples, dtype=np.int32)
        for idx in block_indices:
            coverage_counts[cover_matrix[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            
            # 1. Try to remove redundant blocks
            to_remove = []
            for idx in block_indices:
                # If no triple in this block is covered only by this block, it's redundant
                if not np.any(coverage_counts[cover_matrix[idx]] == 1):
                    to_remove.append(idx)
            
            if to_remove:
                for idx in to_remove:
                    current_set.remove(idx)
                    coverage_counts[cover_matrix[idx]] -= 1
                block_indices = [idx for idx in block_indices if idx not in to_remove]
                changed = True
                continue # Restart loop as coverage changed significantly
            
            # 2. Try to swap blocks
            # For each block, check if it can be replaced by another block that covers its unique triples
            # We only consider swaps if the block has unique triples.
            # If a block has NO unique triples, it was already removed in step 1.
            
            swap_found = False
            # We need to be careful not to modify the list while iterating or swapping multiple times incorrectly
            # We will iterate through a copy of the indices
            for i in range(len(block_indices)):
                if time.time() > deadline - 5:
                    break
                idx = block_indices[i]
                
                # Find unique triples in this block
                unique_triples = cover_matrix[idx][coverage_counts[cover_matrix[idx]] == 1]
                
                if len(unique_triples) == 0:
                    continue # Should have been removed, but just in case
                
                # Find candidate replacement blocks
                # A candidate must cover ALL unique triples of idx
                # We intersect the lists of blocks covering each unique triple
                candidates = None
                for tri in unique_triples:
                    blocks_covering_tri = triple_to_blocks[tri]
                    if candidates is None:
                        candidates = set(blocks_covering_tri)
                    else:
                        candidates.intersection_update(blocks_covering_tri)
                    
                    if not candidates:
                        break
                
                if candidates is None or not candidates:
                    continue
                
                # Remove current block from candidates
                candidates.discard(idx)
                
                # If there are candidates, pick one randomly or the one that maximizes "goodness"
                # For now, just pick the first one or random to escape local minima
                if candidates:
                    # Heuristic: Pick the candidate that covers the most *other* uncovered/shared triples?
                    # Or just random. Random is good for exploration.
                    replacement = random.choice(list(candidates))
                    
                    # Perform swap
                    # Remove idx
                    current_set.remove(idx)
                    coverage_counts[cover_matrix[idx]] -= 1
                    
                    # Add replacement
                    current_set.add(replacement)
                    coverage_counts[cover_matrix[replacement]] += 1
                    
                    # Update block_indices list
                    block_indices[i] = replacement
                    changed = True
                    swap_found = True
                    break # Restart loop after swap to re-evaluate removals
            
            if not changed:
                break
                
        return block_indices

    # Main loop
    restart_count = 0
    while time.time() < deadline - 5:
        block_indices = fast_greedy_cover(restart_count)
        block_indices = local_search(block_indices)
        
        if len(block_indices) < best_len:
            best_len = len(block_indices)
            best_blocks = block_indices[:]
            
        restart_count += 1
        
    return [list(all_k_subsets[i]) for i in best_blocks] if best_blocks is not None else []
