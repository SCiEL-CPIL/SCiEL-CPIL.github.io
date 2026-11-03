import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a Randomized Greedy approach with Local Search.
    """
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tup: i for i, tup in enumerate(all_triples)}
    
    # Pre-compute blocks and their covered triples
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    # triple_to_blocks[j] = list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for b_idx, b in enumerate(all_blocks):
        trs = [triple_to_idx[tup] for tup in itertools.combinations(b, t)]
        block_triples.append(trs)
        for tri_idx in trs:
            triple_to_blocks[tri_idx].append(b_idx)
            
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple restarts to find a better solution
    num_restarts = 40
    
    for r in range(num_restarts):
        # Random seed for diversity
        random.seed(r * 12345 + 67890)
        
        blocks = _greedy_construct(v, k, t, all_blocks, triple_to_blocks, block_triples, num_triples)
        blocks = _local_search_delete(blocks, block_triples, num_triples)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
        # Optional: Break early if we hit the lower bound?
        # Lower bound is 69. If we hit 69, we can stop.
        if best_size <= 69:
            break

    # Convert indices back to lists of points
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]

def _greedy_construct(v, k, t, all_blocks, triple_to_blocks, block_triples, num_triples):
    """
    Greedy construction:
    1. Start with all triples uncovered.
    2. Pick a random uncovered triple.
    3. Find the block covering it that maximizes the number of *other* uncovered triples covered.
    4. Add block, update uncovered.
    """
    uncovered = set(range(num_triples))
    selected = []
    
    while uncovered:
        # Pick a random uncovered triple to anchor the search
        # Using list(uncovered) is O(N). N <= 2024. This is fine.
        tri_idx = random.choice(list(uncovered))
        candidates = triple_to_blocks[tri_idx]
        
        best_block = -1
        best_gain = -1
        
        for b_idx in candidates:
            # Calculate gain: number of uncovered triples covered by this block
            gain = 0
            for tri in block_triples[b_idx]:
                if tri in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1:
            # Should not happen if uncovered is not empty
            break
            
        selected.append(best_block)
        
        # Update uncovered
        for tri in block_triples[best_block]:
            uncovered.discard(tri)
            
    return selected

def _local_search_delete(blocks, block_triples, num_triples):
    """
    Remove any block that does not uniquely cover any triple.
    """
    if not blocks:
        return []
        
    coverage = [0] * num_triples
    for b_idx in blocks:
        for tri in block_triples[b_idx]:
            coverage[tri] += 1
            
    blocks_set = set(blocks)
    changed = True
    while changed:
        changed = False
        for b_idx in list(blocks_set):
            essential = False
            for tri in block_triples[b_idx]:
                if coverage[tri] == 1:
                    essential = True
                    break
            if not essential:
                blocks_set.remove(b_idx)
                for tri in block_triples[b_idx]:
                    coverage[tri] -= 1
                changed = True
                break # Restart to ensure consistency
                
    return list(blocks_set)
