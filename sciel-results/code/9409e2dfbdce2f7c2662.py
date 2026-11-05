import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
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
    
    # Calculate frequency of each triple to determine weights
    # Triples that appear in fewer blocks are "rarer" and should be prioritized
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            idx = triple_to_idx[sub]
            block_triple_indices[i, j] = idx
            triple_freq[idx] += 1
            
    # Weights are inversely proportional to frequency
    # This implements a weighted greedy heuristic
    triple_freq = np.maximum(triple_freq, 1)
    weights = 1.0 / triple_freq.astype(np.float64)
    
    def greedy_cover(random_seed, use_soft_max=False):
        # Use a local RandomState to avoid global side effects
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            if time.time() > deadline:
                break
            
            # Calculate weighted gains
            # Only uncovered triples contribute to the gain
            active_weights = np.where(uncovered_mask, weights, 0.0)
            
            # Sum weights for each block
            gains = np.sum(active_weights[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            if use_soft_max:
                # Use soft-max to break ties smoothly
                # Temperature parameter for soft-max
                temp = 0.1
                exp_gains = np.exp(gains / temp - max_gain / temp)
                probs = exp_gains / np.sum(exp_gains)
                best_idx = rng.choice(n_blocks, p=probs)
            else:
                # Find all blocks with max gain
                candidates = np.where(gains == max_gain)[0]
                
                # Randomly pick one candidate to break ties randomly
                best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_triples = block_triple_indices[best_idx]
            
            # Count newly covered triples to update remaining count
            newly_covered = np.sum(uncovered_mask[block_triples])
            remaining -= newly_covered
            
            # Mark as covered
            uncovered_mask[block_triples] = False
            
            selected_blocks.append(list(all_blocks[best_idx]))
            
        return selected_blocks
    
    def local_search_polish(blocks, max_iterations=50):
        """Try to improve the solution by swapping blocks"""
        if not blocks:
            return blocks
        
        current_blocks = list(blocks)
        current_len = len(current_blocks)
        
        # Build coverage count for each triple
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for block in current_blocks:
            for sub in itertools.combinations(block, t):
                coverage_count[triple_to_idx[sub]] += 1
        
        # Find uncovered triples
        uncovered = np.where(coverage_count == 0)[0]
        if len(uncovered) == 0:
            return current_blocks
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1
            
            # Try removing each block and see if we can add a better one
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    return current_blocks
                
                # Temporarily remove block i
                test_blocks = current_blocks[:i] + current_blocks[i+1:]
                
                # Recalculate coverage
                test_coverage = np.zeros(n_triples, dtype=np.int32)
                for block in test_blocks:
                    for sub in itertools.combinations(block, t):
                        test_coverage[triple_to_idx[sub]] += 1
                
                # Check if all triples are still covered
                if np.any(test_coverage == 0):
                    continue
                
                # Try to find a block that covers more uncovered triples
                # and allows us to remove another block
                best_new_block = None
                best_gain = 0
                
                # Sample some candidate blocks
                sample_size = min(100, n_blocks)
                candidates = rng.choice(n_blocks, size=sample_size, replace=False)
                
                for cand_idx in candidates:
                    cand_block = all_blocks[cand_idx]
                    
                    # Calculate how many new triples this block would cover
                    new_triples = 0
                    for sub in itertools.combinations(cand_block, t):
                        if test_coverage[triple_to_idx[sub]] == 0:
                            new_triples += 1
                    
                    if new_triples > best_gain:
                        best_gain = new_triples
                        best_new_block = cand_idx
                
                if best_new_block is not None and best_gain > 0:
                    # Replace block i with the new block
                    current_blocks[i] = list(all_blocks[best_new_block])
                    improved = True
                    break
        
        return current_blocks
    
    # Run multiple times with different seeds to find a better covering
    best_blocks = None
    best_len = float('inf')
    
    # Use both standard and soft-max variants
    n_restarts_standard = 15
    n_restarts_softmax = 10
    
    for i in range(n_restarts_standard):
        if time.time() > deadline - 30:  # Reserve time for polish
            break
        blocks = greedy_cover(i, use_soft_max=False)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    for i in range(n_restarts_softmax):
        if time.time() > deadline - 30:  # Reserve time for polish
            break
        blocks = greedy_cover(100 + i, use_soft_max=True)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    # Apply local search polish to the best solution
    if best_blocks is not None and time.time() < deadline - 10:
        polished = local_search_polish(best_blocks)
        if len(polished) < best_len:
            best_blocks = polished
            best_len = len(polished)
    
    return best_blocks
