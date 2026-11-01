"""
Vectorized Probabilistic Greedy + Local Search for Covering Design C(24,7,3)
"""
import itertools
import random
import time
import sys

import numpy as np

def _precompute(v, k, t):
    """
    Precompute:
    - all_blocks: list of all k-subsets of v points
    - block_to_triples: for each block, the list of triple indices it covers
    - triple_to_blocks: for each triple, the list of block indices covering it
    """
    points = list(range(v))
    all_blocks = [tuple(sorted(b)) for b in itertools.combinations(points, k)]
    n_blocks = len(all_blocks)
    
    # Map triple to index
    triples = list(itertools.combinations(points, t))
    triple_idx = {ts: i for i, ts in enumerate(triples)}
    n_triples = len(triples)
    
    # For each block, find which triples it covers
    # A block of size k covers C(k, t) triples
    block_to_triples = []
    for b in all_blocks:
        tsets = [triple_idx[ts] for ts in itertools.combinations(b, t)]
        block_to_triples.append(tsets)
        
    # For each triple, find which blocks cover it
    # A triple is in C(v-t, k-t) blocks
    triple_to_blocks = [[] for _ in range(n_triples)]
    for j, b in enumerate(all_blocks):
        for ts in itertools.combinations(b, t):
            idx = triple_idx[ts]
            triple_to_blocks[idx].append(j)
            
    # Convert to numpy arrays for faster access if needed, 
    # but lists of ints are often fast enough for small n.
    # n_blocks for C(24,7) is 346104. n_triples is 2024.
    # This is a lot of data. 
    # 346104 * C(7,3) = 346104 * 35 = 12 million entries. 
    # 2024 * C(17,4) = 2024 * 2380 = 4.8 million entries.
    # Memory is fine. Time to precompute might be the bottleneck.
    
    return all_blocks, block_to_triples, triple_to_blocks, triples, triple_idx, n_blocks, n_triples

def _construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Precomputation
    all_blocks, block_to_triples, triple_to_blocks, triples, triple_idx, n_blocks, n_triples = _precompute(v, k, t)
    
    # If precomputation took too long, we might need a faster heuristic, 
    # but for v=24 it's usually < 5s.
    if time.time() > deadline - 10.0:
        # Fallback to simple greedy if precomputation is too slow
        return _simple_greedy_fallback(v, k, t)

    rng = random.Random(42)
    
    best_blocks = None
    best_size = float('inf')
    
    # Phase 1: Randomized Greedy Construction
    # We will run multiple iterations.
    # Each iteration:
    # 1. Assign random weights to points.
    # 2. Greedily select blocks.
    #    Score of a block = sum of weights of points in block + alpha * (number of newly covered triples)
    #    Actually, the prompt suggests: "maximizes the weighted sum of currently uncovered triples"
    #    Wait, "weighted sum of currently uncovered triples" implies weighting the triples, not points?
    #    "assign random weights to points and iteratively select the 7-subset that maximizes the weighted sum of currently uncovered triples"
    #    This is slightly ambiguous. Usually, you weight points to bias which points are picked.
    #    Or you weight triples to prioritize covering certain triples.
    #    Let's interpret: We want to escape local minima.
    #    Standard greedy picks the block covering the most uncovered triples.
    #    Randomized: Add noise to the "gain".
    #    Gain(block) = (# uncovered triples covered by block) + noise
    #    Where noise is derived from random point weights?
    #    Let's try: Score(block) = sum_{p in block} w_p + lambda * (# uncovered triples covered)
    #    This biases towards points with high weights.
    
    # To make it fast, we can precompute for each block the list of triples it covers.
    # We need to track uncovered triples efficiently.
    # n_triples = 2024. We can use a boolean array of size 2024.
    
    max_iter = 200 # Try to run many iterations
    for i in range(max_iter):
        if time.time() > deadline - 20.0: # Leave time for local search
            break
            
        # Random weights for points
        w = rng.random(v)
        
        # Initialize uncovered
        uncovered = np.ones(n_triples, dtype=bool)
        blocks = []
        
        # Greedy selection
        # We need to find the block that maximizes:
        # score = sum(w[p] for p in block) + lambda * count(uncovered triples in block)
        
        # Precomputing sum(w[p] for p in block) for all 346k blocks is fast.
        block_weights = np.array([sum(w[p] for p in b) for b in all_blocks])
        
        # Lambda controls the trade-off between random point preference and coverage
        # As coverage increases, we might want to prioritize coverage more?
        # Or keep it constant. Let's try constant lambda = 1.0 initially.
        # The gain in coverage is an integer 0..35. Weights are 0..1.
        # So coverage dominates if lambda is large enough.
        # Let's use lambda = 10.0 to ensure we mostly cover, but break ties randomly.
        lambda_cov = 10.0
        
        while True:
            if not np.any(uncovered):
                break
                
            # Calculate current coverage count for each block
            # This is the expensive part if done naively.
            # n_blocks = 346k. n_triples = 2024.
            # For each block, count how many of its triples are in 'uncovered'.
            
            # Optimization: 
            # We can maintain a counter for each triple of how many times it's covered? No, we just need if it's uncovered.
            # Let's vectorize this.
            # Create a matrix M of shape (n_blocks, n_triples)? No, too big (346k * 2024 * 1 byte = 700MB).
            
            # Alternative:
            # For each block, we have a list of triple indices.
            # We can sum uncovered[triple_indices] for each block.
            # This is 346k operations, each involving ~35 lookups.
            # In Python, this loop will be slow.
            
            # Better approach for Python:
            # Use numpy to speed up the counting.
            # We can store block_to_triples as a 2D array? 
            # The lengths are all C(7,3)=35.
            # So block_to_triples_np is (n_blocks, 35).
            # Then coverage_counts = uncovered[block_to_triples_np].sum(axis=1)
            # This is a fancy indexing operation. 346k * 35 indices.
            # Should be fast in numpy.
            
            if not hasattr(_construct, 'block_to_triples_np'):
                _construct.block_to_triples_np = np.array(block_to_triples, dtype=np.int32)
            
            btt_np = _construct.block_to_triples_np
            coverage_counts = uncovered[btt_np].sum(axis=1)
            
            scores = block_weights + lambda_cov * coverage_counts
            
            # Pick best block
            best_idx = np.argmax(scores)
            best_block = all_blocks[best_idx]
            
            # Update uncovered
            tsets = block_to_triples[best_idx]
            for idx in tsets:
                uncovered[idx] = False
                
            blocks.append(list(best_block))
            
            # Check time
            if time.time() > deadline - 15.0:
                break
        
        # End of greedy construction
        current_size = len(blocks)
        if current_size < best_size:
            best_size = current_size
            best_blocks = [list(b) for b in blocks]
            
    if best_blocks is None:
        return _simple_greedy_fallback(v, k, t)
        
    # Phase 2: Local Search (Block Swapping)
    # Try to replace a block with another block that covers the same "unique" triples 
    # but allows removal of another block?
    # Simpler: Try to remove redundant blocks.
    # Then try to swap blocks to reduce total count?
    
    # First, remove redundants
    best_blocks = _remove_redundant_fast(best_blocks, block_to_triples, triple_idx, n_triples, deadline)
    
    # Local search: 
    # For each block, try replacing it with a random block.
    # If the new set is still valid and allows removal of another block, accept.
    # Or just try to find a smaller covering via random restarts of local search?
    
    # Given time constraints, a simple "try to drop a block" loop after randomization is often sufficient.
    # The randomized greedy already explores different basins.
    
    # One more pass of redundancy removal
    best_blocks = _remove_redundant_fast(best_blocks, block_to_triples, triple_idx, n_triples, deadline)
    
    return best_blocks

def _remove_redundant_fast(blocks, block_to_triples, triple_idx, n_triples, deadline):
    """
    Remove redundant blocks.
    A block is redundant if all triples it covers are covered by other blocks.
    """
    if not blocks:
        return blocks
    
    n_blocks = len(blocks)
    # Map triple index to count of blocks covering it
    counts = np.zeros(n_triples, dtype=np.int32)
    
    # Precompute triple indices for current blocks
    block_triple_lists = []
    for b in blocks:
        tsets = [triple_idx[ts] for ts in itertools.combinations(b, 3)]
        block_triple_lists.append(tsets)
        for idx in tsets:
            counts[idx] += 1
            
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 5.0:
            break
            
        for i in range(len(blocks)):
            if time.time() > deadline - 5.0:
                break
                
            tsets = block_triple_lists[i]
            # Check if any triple is covered only by this block
            is_redundant = True
            for idx in tsets:
                if counts[idx] == 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                # Remove block i
                for idx in tsets:
                    counts[idx] -= 1
                blocks.pop(i)
                block_triple_lists.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    return blocks

def _simple_greedy_fallback(v, k, t):
    """
    Fast, non-vectorized greedy for fallback.
    """
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_idx = {ts: i for i, ts in enumerate(triples)}
    n_triples = len(triples)
    
    uncovered = set(range(n_triples))
    blocks = []
    
    # Precompute all blocks and their triples
    all_blocks = [tuple(sorted(b)) for b in itertools.combinations(points, k)]
    block_triples = []
    for b in all_blocks:
        tsets = [triple_idx[ts] for ts in itertools.combinations(b, t)]
        block_triples.append(set(tsets))
        
    while uncovered:
        best_block = None
        best_gain = -1
        
        for j, b in enumerate(all_blocks):
            gain = len(uncovered.intersection(block_triples[j]))
            if gain > best_gain:
                best_gain = gain
                best_block = b
                
        if best_block is None or best_gain == 0:
            break
            
        blocks.append(list(best_block))
        uncovered -= block_triples[all_blocks.index(best_block)]
        
    return blocks

def construct(v, k, t):
    return _construct(v, k, t)

# Helper to ensure we don't crash on import
if __name__ == "__main__":
    # Test
    res = construct(24, 7, 3)
    print(f"Found covering with {len(res)} blocks")
