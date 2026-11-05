"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
from collections import defaultdict
import heapq

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using optimized Greedy Set Cover
    with incidence matrix and local search refinement.
    """
    # 1. Generate all triples and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # 2. Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Build block_triples: for each block, list of triple indices it covers
    block_triples = []
    for block in all_blocks:
        triples_in_block = [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
        block_triples.append(triples_in_block)
    
    # 4. Build reverse incidence: triple_to_blocks
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi, triples in enumerate(block_triples):
        for ti in triples:
            triple_to_blocks[ti].append(bi)
    
    # 5. Greedy Set Cover with heap-based selection
    uncovered = np.ones(num_triples, dtype=bool)
    used = np.zeros(num_blocks, dtype=bool)
    gains = np.array([len(bt) for bt in block_triples], dtype=np.int32)
    selected_indices = []
    
    # Initialize heap
    heap = [(-gains[i], i) for i in range(num_blocks)]
    heapq.heapify(heap)
    
    for _ in range(num_triples + 10):
        if not uncovered.any():
            break
        
        best_block_idx = -1
        best_gain = 0
        
        while heap:
            neg_gain, bi = heapq.heappop(heap)
            gain = -neg_gain
            if used[bi]:
                continue
            if gain != gains[bi]:
                heapq.heappush(heap, (-gains[bi], bi))
                continue
            best_block_idx = bi
            best_gain = gain
            break
        
        if best_block_idx == -1 or best_gain == 0:
            break
        
        selected_indices.append(best_block_idx)
        used[best_block_idx] = True
        gains[best_block_idx] = 0
        
        # Update uncovered and gains
        for ti in block_triples[best_block_idx]:
            if not uncovered[ti]:
                continue
            uncovered[ti] = False
            for bi in triple_to_blocks[ti]:
                if not used[bi]:
                    gains[bi] -= 1
                    heapq.heappush(heap, (-gains[bi], bi))
    
    selected_blocks = [list(all_blocks[i]) for i in selected_indices]
    
    # 6. Redundancy removal
    cover_count = np.zeros(num_triples, dtype=np.int32)
    for block in selected_blocks:
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            cover_count[idx] += 1
    
    changed = True
    while changed:
        changed = False
        random.shuffle(selected_blocks)
        for i in range(len(selected_blocks)):
            block = selected_blocks[i]
            is_redundant = True
            for triple in itertools.combinations(block, t):
                t_idx = triple_to_idx[triple]
                if cover_count[t_idx] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for triple in itertools.combinations(block, t):
                    t_idx = triple_to_idx[triple]
                    cover_count[t_idx] -= 1
                selected_blocks.pop(i)
                changed = True
                break
    
    return selected_blocks

# EVOLVE-BLOCK-END
