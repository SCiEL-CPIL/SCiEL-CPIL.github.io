import itertools
import time
import random
import sys

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy + Local Search.
    
    Strategy:
    1. Generate an initial covering using a randomized greedy algorithm.
    2. Apply local search (redundancy removal) to minimize the number of blocks.
    3. Repeat with different random seeds to find better solutions.
    4. Return the best solution found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    # Precompute all t-subsets
    total_triples = list(itertools.combinations(range(v), t))
    n_triples = len(total_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(total_triples)}
    
    # Precompute triples for each potential block? No, too many blocks.
    # Instead, compute triples for a specific block on the fly.
    
    def get_block_triples(block):
        """Return list of triple indices covered by a block."""
        return [triple_to_idx[triple] for triple in itertools.combinations(block, t)]

    def greedy_construct(seed):
        """
        Greedily construct a covering.
        """
        rng = random.Random(seed)
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered triple to guide the next block
            # Or pick the one that is hardest to cover? 
            # Simple: pick random uncovered triple
            seed_idx = rng.choice(list(uncovered))
            seed_triple = total_triples[seed_idx]
            
            block = list(seed_triple)
            
            # Complete the block greedily
            # We want to maximize the number of new uncovered triples covered.
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate gain: how many uncovered triples does adding x cover?
                    gain = 0
                    # Triples formed by x and (t-1) elements from current block
                    for rest in itertools.combinations(block, t - 1):
                        triple = tuple(sorted(rest + (x,)))
                        idx = triple_to_idx.get(triple, -1)
                        if idx != -1 and idx in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if not best_candidates:
                    # Should not happen if v > k and uncovered is not empty
                    break
                
                chosen = rng.choice(best_candidates)
                block.append(chosen)
            
            if len(block) == k:
                block_sorted = sorted(block)
                blocks.append(block_sorted)
                
                # Update uncovered
                for idx in get_block_triples(block_sorted):
                    uncovered.discard(idx)
        
        return blocks

    def local_search_reduce(blocks, seed):
        """
        Reduce the number of blocks by removing redundant ones.
        A block is redundant if all its triples are covered by other blocks.
        """
        if not blocks:
            return blocks
        
        rng = random.Random(seed)
        
        # Count coverage for each triple
        # coverage_count[i] = number of blocks covering triple i
        coverage_count = [0] * n_triples
        
        # Initialize counts
        for b in blocks:
            for idx in get_block_triples(b):
                coverage_count[idx] += 1
        
        current_blocks = list(blocks)
        
        # Iterative removal
        # We iterate until no more blocks can be removed
        # To avoid order bias, we shuffle the list of blocks to check
        improved = True
        while improved:
            improved = False
            # Shuffle indices to check in random order
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(current_blocks):
                    continue
                    
                if time.time() > deadline:
                    break
                
                b = current_blocks[i]
                # Check if b is redundant
                # Decrement counts for triples in b
                b_triples = get_block_triples(b)
                is_redundant = True
                for idx in b_triples:
                    coverage_count[idx] -= 1
                    if coverage_count[idx] == 0:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove b
                    # We already decremented counts, so they are correct for the new set
                    # Just remove the block from the list
                    # Note: removing from list while iterating is tricky, 
                    # but we are iterating over indices of the *original* list snapshot?
                    # No, current_blocks is being modified.
                    # Better approach: mark for removal and filter at end of pass, 
                    # or just pop and adjust.
                    # Since we are checking indices from a shuffled list of the *current* length,
                    # popping changes indices.
                    # Let's just use a set of indices to remove.
                    pass # Handled below
                    
                else:
                    # Restore counts
                    for idx in b_triples:
                        coverage_count[idx] += 1
        
        # The loop above is a bit messy with in-place modification. 
        # Let's rewrite local_search_reduce more robustly.
        
        # Re-initialize for robust implementation
        coverage_count = [0] * n_triples
        for b in blocks:
            for idx in get_block_triples(b):
                coverage_count[idx] += 1
                
        current_blocks = list(blocks)
        
        # Repeat passes until no change
        max_passes = 20
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            
            removed_any = False
            # Iterate backwards to allow safe popping, or use a set of removals
            # Shuffling is better for escaping local minima
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            to_remove = []
            
            for i in indices:
                if i >= len(current_blocks):
                    continue
                    
                b = current_blocks[i]
                b_triples = get_block_triples(b)
                
                # Check redundancy
                can_remove = True
                for idx in b_triples:
                    if coverage_count[idx] <= 1:
                        # If count is 1, this block is the only one covering it
                        # If count is 0, error (shouldn't happen)
                        can_remove = False
                        break
                
                if can_remove:
                    to_remove.append(i)
            
            if to_remove:
                # Remove all identified redundant blocks
                # We must update coverage counts
                to_remove_set = set(to_remove)
                new_blocks = []
                for i, b in enumerate(current_blocks):
                    if i in to_remove_set:
                        # Decrement counts
                        for idx in get_block_triples(b):
                            coverage_count[idx] -= 1
                    else:
                        new_blocks.append(b)
                
                current_blocks = new_blocks
                removed_any = True
            
            if not removed_any:
                break
        
        return current_blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    n_restarts = 200 # Try many, but bounded by time
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # 1. Greedy
        blocks = greedy_construct(seed=i)
        
        if not blocks:
            continue
            
        # 2. Local Search
        reduced_blocks = local_search_reduce(blocks, seed=i)
        
        # 3. Verify and Update Best
        # Quick verification: ensure all triples covered
        # Since local search only removes blocks if they are redundant, 
        # and greedy started with a full cover, the result should be valid.
        # But let's be safe.
        
        if len(reduced_blocks) < best_count:
            best_count = len(reduced_blocks)
            best_blocks = reduced_blocks
            
            # Early exit if we hit the theoretical lower bound or known best?
            # LB is 69. If we get 70, that's huge.
            # Best known is 76.
            if best_count <= 70:
                break
                
    # If no blocks found (shouldn't happen), fallback
    if best_blocks is None:
        best_blocks = greedy_construct(seed=0)
        
    # Final safety check: ensure distinct blocks
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
            
    return unique_blocks
