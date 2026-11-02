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
    # Shape: (n_k_subsets, C(k,t))
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    cover_matrix = np.zeros((n_k_subsets, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_k_subsets):
        for j, triple in enumerate(itertools.combinations(block, t)):
            cover_matrix[i, j] = triple_to_idx[triple]
            
    best_blocks = None
    best_len = float('inf')
    
    def fast_greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Vectorized gain calculation
            # uncovered[cover_matrix] creates a boolean array of shape (n_k_subsets, c_k_t)
            # sum along axis 1 gives the count of uncovered triples covered by each block
            gains = np.sum(uncovered[cover_matrix], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks

    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Convert blocks to tuples for hashing if needed, but we use indices in cover_matrix
        # We need to map block -> index in all_k_subsets for fast lookup?
        # Or just rebuild coverage map from scratch each time.
        
        def build_coverage_map(bks):
            coverage = np.zeros(n_triples, dtype=np.int32)
            block_indices = []
            for b in bks:
                # Find index of b in all_k_subsets? 
                # Since all_k_subsets is a list of tuples, we can use a dict.
                pass
            
            # It is faster to just iterate and use the precomputed cover_matrix if we have the indices.
            # Let's create a map from block tuple to its index in all_k_subsets
            # This is a one-time cost per local search call if we don't cache it, 
            # but all_k_subsets is fixed.
            
            # Actually, we can just use the indices if we store them.
            # Let's assume blocks are stored as tuples from all_k_subsets.
            
            # To avoid O(N * C(k,t)) lookups in a dict, we can just use the indices.
            # But fast_greedy_cover returns tuples.
            # Let's create a global dict: block_to_k_idx
            return coverage
        
        # Precompute block_to_k_idx
        block_to_k_idx = {block: i for i, block in enumerate(all_k_subsets)}
        
        def build_coverage_map(bks):
            coverage = np.zeros(n_triples, dtype=np.int32)
            for b in bks:
                idx = block_to_k_idx[b]
                coverage[cover_matrix[idx]] += 1
            return coverage
        
        coverage = build_coverage_map(blocks)
        changed = True
        
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                idx = block_to_k_idx[block]
                # Check if this block has any unique triples
                # unique if coverage[triple] == 1
                # We can check if np.any(coverage[cover_matrix[idx]] == 1)
                if not np.any(coverage[cover_matrix[idx]] == 1):
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                coverage = build_coverage_map(new_blocks)
                blocks = new_blocks
                changed = True
        
        # Try swap-based local search
        # For each block, if it has unique triples, try to replace it with another block 
        # that covers those unique triples and minimizes total blocks (or just covers them).
        # This is complex. A simpler heuristic: 
        # If we have a block with unique triples, can we find a different block that covers 
        # all those unique triples AND is not in the current set?
        # If yes, swap. This might not reduce count, but might enable further removals.
        
        # Given time constraints, the removal pass is the most effective.
        # The current program had a swap pass. Let's include a simplified version.
        
        for _ in range(2):
            if time.time() > deadline - 10:
                break
                
            coverage = build_coverage_map(blocks)
            improved = False
            
            for i, block in enumerate(blocks):
                idx = block_to_k_idx[block]
                unique_triples_in_block = cover_matrix[idx][coverage[cover_matrix[idx]] == 1]
                
                if len(unique_triples_in_block) == 0:
                    continue
                
                # Try to find a replacement block j not in blocks 
                # such that it covers all unique_triples_in_block
                # And ideally covers more "shared" triples to allow future removals?
                # For now, just check if a replacement exists.
                
                # This is O(N_blocks * N_k_subsets) which is too slow if done naively.
                # Skip complex swaps for now, rely on greedy quality and removal.
                pass
                
            if not improved:
                break
                
        return blocks

    # Main loop
    restart_count = 0
    while time.time() < deadline - 5:
        blocks = fast_greedy_cover(restart_count)
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
        restart_count += 1
        
        # If we found a very good solution, we might want to spend more time polishing it
        # but for now, just keep restarting to find a better greedy seed.
        
    return [list(b) for b in best_blocks] if best_blocks is not None else []
