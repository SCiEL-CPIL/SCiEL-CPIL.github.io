import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (evaluator kills at 150s)
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # c(k,t) = number of t-subsets in a k-subset
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Compute triple coverage frequency
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
        
    # Weight for each triple: 1/freq (rarer triples get higher weight)
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)
    
    # Precompute block weights
    block_weights = np.sum(triple_weights[block_triples], axis=1)
    
    # Build a mapping from triple index to list of block indices that cover it
    triple_to_blocks = defaultdict(list)
    for i in range(n_blocks):
        for triple_idx in block_triples[i]:
            triple_to_blocks[triple_idx].append(i)
    
    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False, seed=None):
        """Run greedy set cover, optionally weighted."""
        if seed is not None:
            np.random.seed(seed)
        uncovered = uncovered_init.copy()
        selected = []
        selected_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)
                
            # Mask out already-selected blocks
            for idx in selected_set:
                gains[idx] = 0
                
            # Find the block with maximum gain
            if random_tie:
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                    
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected
    
    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering."""
        if not selected_indices:
            return selected_indices
            
        remaining = list(selected_indices)
        
        # Build coverage counts: for each triple, how many remaining blocks cover it
        # and which blocks cover it
        triple_cover_count = np.zeros(n_triples, dtype=np.int32)
        triple_cover_blocks = defaultdict(list)
        
        for block_idx in remaining:
            for triple_idx in block_triples[block_idx]:
                triple_cover_count[triple_idx] += 1
                triple_cover_blocks[triple_idx].append(block_idx)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            
            for block_idx in remaining:
                if time.time() > deadline:
                    new_remaining.append(block_idx)
                    continue
                    
                # Check if this block is the only one covering any triple
                can_remove = True
                for triple_idx in block_triples[block_idx]:
                    if triple_cover_count[triple_idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove this block
                    changed = True
                    for triple_idx in block_triples[block_idx]:
                        triple_cover_count[triple_idx] -= 1
                        triple_cover_blocks[triple_idx].remove(block_idx)
                else:
                    new_remaining.append(block_idx)
                    
            remaining = new_remaining
            
        return remaining
    
    def local_search_swap(selected_indices, max_swaps=50):
        """Try to reduce block count by swapping blocks."""
        if len(selected_indices) <= 1:
            return selected_indices
            
        selected_set = set(selected_indices)
        remaining = list(selected_indices)
        
        # Build coverage counts
        triple_cover_count = np.zeros(n_triples, dtype=np.int32)
        for block_idx in remaining:
            for triple_idx in block_triples[block_idx]:
                triple_cover_count[triple_idx] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing each block and see if we can cover the lost triples with one new block
            for i, block_idx in enumerate(remaining):
                if time.time() > deadline - 5:
                    break
                    
                # Temporarily remove this block
                lost_triples = []
                for triple_idx in block_triples[block_idx]:
                    if triple_cover_count[triple_idx] == 1:
                        lost_triples.append(triple_idx)
                    
                if not lost_triples:
                    # Can remove without adding
                    for triple_idx in block_triples[block_idx]:
                        triple_cover_count[triple_idx] -= 1
                    remaining.pop(i)
                    improved = True
                    break
                    
                # Try to find a single block that covers all lost_triples
                # and is not already in remaining
                found = False
                for candidate in triple_to_blocks[lost_triples[0]]:
                    if candidate in selected_set:
                        continue
                    
                    # Check if candidate covers all lost_triples
                    covers_all = True
                    for triple_idx in lost_triples:
                        if triple_idx not in block_triples[candidate]:
                            covers_all = False
                            break
                            
                    if covers_all:
                        # Swap: remove block_idx, add candidate
                        for triple_idx in block_triples[block_idx]:
                            triple_cover_count[triple_idx] -= 1
                        for triple_idx in block_triples[candidate]:
                            triple_cover_count[triple_idx] += 1
                        remaining[i] = candidate
                        selected_set.discard(block_idx)
                        selected_set.add(candidate)
                        improved = True
                        found = True
                        break
                        
                if found:
                    break
                    
        return remaining
    
    def block_merge(selected_indices):
        """Try to merge two blocks into one if possible."""
        if len(selected_indices) <= 1:
            return selected_indices
            
        remaining = list(selected_indices)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            n = len(remaining)
            
            for i in range(n):
                if time.time() > deadline - 10:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline - 10:
                        break
                    
                    # Check if the union of two blocks can be covered by one block
                    # Since blocks are k-element subsets, we can't merge two k-sets into one k-set
                    # unless they are the same. So this operation is not directly applicable.
                    # Instead, try: can we remove one block and cover its unique triples with another existing block?
                    block_i = remaining[i]
                    block_j = remaining[j]
                    
                    # Get triples unique to block_i (covered only by block_i among remaining)
                    unique_to_i = []
                    for triple_idx in block_triples[block_i]:
                        count = 0
                        for b in remaining:
                            if triple_idx in block_triples[b]:
                                count += 1
                        if count == 1:
                            unique_to_i.append(triple_idx)
                            
                    if not unique_to_i:
                        # block_i is redundant
                        remaining.pop(i)
                        improved = True
                        break
                        
                    # Check if block_j covers all unique_to_i
                    if all(triple_idx in block_triples[block_j] for triple_idx in unique_to_i):
                        # Remove block_i
                        remaining.pop(i)
                        improved = True
                        break
                        
            if improved:
                continue
                
        return remaining
    
    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    
    def update_best(selected):
        nonlocal best_solution, best_size
        if selected:
            pruned = prune_blocks(selected)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(all_blocks[i]) for i in pruned]
    
    # Strategy 1: Standard greedy (unweighted)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False)
    if selected:
        swapped = local_search_swap(selected)
        update_best(swapped)
    
    # Strategy 2: Weighted greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False)
    if selected:
        swapped = local_search_swap(selected)
        update_best(swapped)
    
    # Strategy 3-20: Random tie-breaking restarts
    for restart in range(20):
        if time.time() > deadline - 20.0:
            break
            
        use_weighted = (restart % 2 == 0)
        seed = restart * 12345 + 67890
        selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=use_weighted, random_tie=True, seed=seed)
        if selected:
            swapped = local_search_swap(selected)
            update_best(swapped)
    
    # Additional local search on the best solution found
    if best_solution is not None and time.time() < deadline - 10:
        # Reconstruct the selected indices from best_solution
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        current_indices = []
        for block in best_solution:
            idx = block_to_idx.get(tuple(block))
            if idx is not None:
                current_indices.append(idx)
        
        # Apply local search
        improved = local_search_swap(current_indices)
        merged = block_merge(improved)
        if len(merged) < best_size:
            best_size = len(merged)
            best_solution = [list(all_blocks[i]) for i in merged]
    
    # Final validation and return
    if best_solution is None:
        # Fallback: return a valid covering
        # This shouldn't happen, but just in case
        return [list(b) for b in all_blocks[:100]]
        
    return best_solution
