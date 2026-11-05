import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin for 150s limit
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    # block_triple_indices[i, j] is the index of the j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
    
    # Precompute block sets for fast membership testing
    block_sets = [set(block) for block in all_blocks]
    
    def greedy_cover(random_seed):
        # Use a local RandomState to avoid global side effects
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Calculate gains: number of uncovered triples in each block
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one candidate to break ties randomly
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_idx = block_triple_indices[best_idx]
            uncovered_mask[block_idx] = False
            remaining -= max_gain
            selected_blocks.append(all_blocks[best_idx])
            
        return selected_blocks
    
    def remove_redundant(blocks):
        """Remove blocks that are not needed (their triples are all covered by other blocks)."""
        if not blocks:
            return blocks
        
        # Count how many times each triple is covered
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                triple_count[triple_to_idx[sub]] += 1
        
        # Find blocks that have at least one triple covered only by this block
        essential_blocks = []
        for i, block in enumerate(blocks):
            is_essential = False
            for sub in itertools.combinations(block, t):
                if triple_count[triple_to_idx[sub]] == 1:
                    is_essential = True
                    break
            if is_essential:
                essential_blocks.append(block)
        
        return essential_blocks
    
    def local_search(blocks, random_seed, max_iterations=100):
        """Try to improve the covering by replacing blocks."""
        rng = np.random.RandomState(random_seed)
        current_blocks = list(blocks)
        current_len = len(current_blocks)
        
        if current_len == 0:
            return current_blocks
        
        # Build coverage count
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for block in current_blocks:
            for sub in itertools.combinations(block, t):
                triple_count[triple_to_idx[sub]] += 1
        
        for _ in range(max_iterations):
            # Pick a random block to try removing
            if not current_blocks:
                break
            idx = rng.randint(len(current_blocks))
            candidate_block = current_blocks[idx]
            
            # Temporarily remove and check if coverage is still valid
            for sub in itertools.combinations(candidate_block, t):
                triple_count[triple_to_idx[sub]] -= 1
            
            # Check if all triples are still covered
            if np.min(triple_count) >= 1:
                # Removal is valid
                current_blocks.pop(idx)
                current_len -= 1
            else:
                # Restore
                for sub in itertools.combinations(candidate_block, t):
                    triple_count[triple_to_idx[sub]] += 1
            
            # Try to add a block that covers the most uncovered triples
            if np.any(triple_count == 0):
                # Find blocks that cover at least one uncovered triple
                gains = np.sum((triple_count == 0)[block_triple_indices], axis=1)
                max_gain = np.max(gains)
                if max_gain > 0:
                    candidates = np.where(gains == max_gain)[0]
                    best_idx = candidates[rng.randint(len(candidates))]
                    new_block = all_blocks[best_idx]
                    
                    # Check if this block is already in current_blocks
                    if new_block not in current_blocks:
                        current_blocks.append(new_block)
                        current_len += 1
                        for sub in itertools.combinations(new_block, t):
                            triple_count[triple_to_idx[sub]] += 1
        
        return current_blocks
    
    # Run multiple times with different seeds to find a better covering
    best_blocks = None
    best_len = float('inf')
    
    # Adaptive number of restarts based on time
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        blocks = greedy_cover(i)
        
        # Remove redundant blocks
        blocks = remove_redundant(blocks)
        
        # Local search to improve
        blocks = local_search(blocks, i + 1000, max_iterations=50)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we have time, do more local search on the best solution
        if time.time() < deadline - 10 and best_blocks is not None:
            best_blocks = local_search(best_blocks, i + 2000, max_iterations=100)
            best_len = len(best_blocks)
    
    # Final polish if time permits
    if time.time() < deadline - 5 and best_blocks is not None:
        best_blocks = remove_redundant(best_blocks)
        best_blocks = local_search(best_blocks, 99999, max_iterations=200)
    
    return [list(b) for b in best_blocks]
