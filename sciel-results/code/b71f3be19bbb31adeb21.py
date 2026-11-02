import itertools
import time
import random
from collections import defaultdict
import numpy as np

# Global cache for precomputed data
_TRIPLE_TO_IDX = None
_ALL_BLOCKS = None
_BLOCK_TO_TRIPLES_IDX = None
_NEIGHBORS = None

def _precompute(v, k, t):
    global _TRIPLE_TO_IDX, _ALL_BLOCKS, _BLOCK_TO_TRIPLES_IDX, _NEIGHBORS
    
    if _TRIPLE_TO_IDX is not None:
        return
        
    # 1. Map triples to indices
    all_triples = list(itertools.combinations(range(v), t))
    _TRIPLE_TO_IDX = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # 2. Generate all blocks and their triple indices
    _ALL_BLOCKS = list(itertools.combinations(range(v), k))
    _BLOCK_TO_TRIPLES_IDX = {}
    
    for block in _ALL_BLOCKS:
        tris = [ _TRIPLE_TO_IDX[tri] for tri in itertools.combinations(block, t) ]
        _BLOCK_TO_TRIPLES_IDX[block] = tris
        
    # 3. Precompute neighbors (blocks sharing at least k-2 elements? or k-1?)
    # Sharing k-1 elements (6 for k=7) is a good neighborhood.
    # Sharing k-2 elements (5 for k=7) is larger but might be too big.
    # Let's stick to sharing at least k-1 elements for speed, or k-2 if we have time.
    # C(17,1) = 17 neighbors for k-1.
    # C(17,2) = 136 neighbors for k-2.
    # Total for k-2: 21 * 136 = 2856.
    # This is acceptable.
    
    _NEIGHBORS = {}
    for block in _ALL_BLOCKS:
        neighbors = []
        block_set = set(block)
        # Blocks sharing k-2 elements
        # Choose k-2 from block, and 2 from outside
        for subset in itertools.combinations(block, k-2):
            subset_set = set(subset)
            outside = [p for p in range(v) if p not in subset_set]
            for new_pair in itertools.combinations(outside, 2):
                new_block = tuple(sorted(subset_set | set(new_pair)))
                if new_block != block:
                    neighbors.append(new_block)
        _NEIGHBORS[block] = neighbors

def _greedy_cover_exhaustive(v, k, t, rng, time_limit):
    _precompute(v, k, t)
    
    all_triples = list(itertools.combinations(range(v), t))
    uncovered = set(all_triples)
    blocks = []
    start_time = time.time()
    
    while uncovered:
        if time.time() > time_limit:
            break
            
        pivot_triple = rng.choice(list(uncovered))
        pivot_set = set(pivot_triple)
        
        other_points = [p for p in range(v) if p not in pivot_set]
        
        best_gain = -1
        best_block = None
        
        for combo in itertools.combinations(other_points, k - t):
            block = tuple(sorted(pivot_set | set(combo)))
            
            # Calculate gain
            gain = 0
            for tri in itertools.combinations(block, t):
                if tri in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is not None:
            blocks.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                uncovered.discard(tri)
            
    return [list(b) for b in blocks]

def _reduce_covering(blocks, v, k, t):
    if not blocks:
        return blocks
    
    triple_cover_count = defaultdict(int)
    block_triples = []
    
    for block in blocks:
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_cover_count[tri] += 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_triples = []
        
        for i, block in enumerate(blocks):
            is_essential = False
            for tri in block_triples[i]:
                if triple_cover_count[tri] == 1:
                    is_essential = True
                    break
            
            if is_essential:
                new_blocks.append(block)
                new_block_triples.append(block_triples[i])
            else:
                changed = True
                for tri in block_triples[i]:
                    triple_cover_count[tri] -= 1
        
        blocks = new_blocks
        block_triples = new_block_triples
    
    return blocks

def _local_search_safe(blocks, v, k, t, rng, time_limit):
    _precompute(v, k, t)
    
    blocks = [tuple(b) for b in blocks]
    
    # Precompute coverage counts and block triples (as indices for speed)
    triple_cover_count = defaultdict(int)
    block_triples_idx = []
    
    for block in blocks:
        tris_idx = _BLOCK_TO_TRIPLES_IDX[block]
        block_triples_idx.append(tris_idx)
        for idx in tris_idx:
            triple_cover_count[idx] += 1
            
    # Helper to check if a block can be removed
    def can_remove(block_idx_list):
        for idx in block_idx_list:
            if triple_cover_count[idx] == 1:
                return False
        return True

    improved = True
    while improved and time.time() < time_limit:
        improved = False
        n_blocks = len(blocks)
        
        # 1. Try to remove redundant blocks
        for i in range(n_blocks):
            if can_remove(block_triples_idx[i]):
                # Remove block i
                for idx in block_triples_idx[i]:
                    triple_cover_count[idx] -= 1
                blocks.pop(i)
                block_triples_idx.pop(i)
                improved = True
                break # Restart loop after removal
        
        if improved:
            continue
            
        # 2. Try to swap blocks
        # Shuffle indices to try different orders
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > time_limit:
                break
                
            current_block = blocks[i]
            current_triples_idx = block_triples_idx[i]
            current_triples_set = set(current_triples_idx)
            
            # Try neighbors
            neighbors = _NEIGHBORS.get(current_block, [])
            # Shuffle neighbors for randomness
            rng.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor in blocks: # Avoid duplicates
                    continue
                    
                neighbor_triples_idx = _BLOCK_TO_TRIPLES_IDX[neighbor]
                neighbor_triples_set = set(neighbor_triples_idx)
                
                # Lost triples: in current, not in neighbor
                lost = current_triples_set - neighbor_triples_set
                
                # Check if lost triples are covered by others
                valid = True
                for idx in lost:
                    if triple_cover_count[idx] == 1:
                        valid = False
                        break
                        
                if valid:
                    # Perform swap
                    # Decrement current
                    for idx in current_triples_idx:
                        triple_cover_count[idx] -= 1
                    # Increment neighbor
                    for idx in neighbor_triples_idx:
                        triple_cover_count[idx] += 1
                        
                    blocks[i] = neighbor
                    block_triples_idx[i] = neighbor_triples_idx
                    
                    improved = True
                    break # Restart loop after swap
                    
            if improved:
                break
                
    return [list(b) for b in blocks]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    best_blocks = None
    best_size = float('inf')
    
    # 1. Initial Greedy
    rng = random.Random(42)
    blocks = _greedy_cover_exhaustive(v, k, t, rng, deadline)
    blocks = _reduce_covering(blocks, v, k, t)
    best_blocks = blocks
    best_size = len(blocks)
    
    # 2. Local Search
    if time.time() < deadline:
        rng_ls = random.Random(123)
        improved_blocks = _local_search_safe(best_blocks, v, k, t, rng_ls, deadline)
        if len(improved_blocks) < best_size:
            best_blocks = improved_blocks
            best_size = len(best_blocks)
            
    # 3. Multiple Randomized Greedy Runs
    seed_counter = 1
    while time.time() < deadline - 10:
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        blocks = _greedy_cover_exhaustive(v, k, t, rng, deadline)
        
        if time.time() > deadline - 5:
            break
            
        blocks = _reduce_covering(blocks, v, k, t)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
            if time.time() < deadline - 15:
                rng_ls = random.Random(seed_counter + 1000)
                improved_blocks = _local_search_safe(best_blocks, v, k, t, rng_ls, deadline - 5)
                if len(improved_blocks) < best_size:
                    best_size = len(improved_blocks)
                    best_blocks = improved_blocks
                    
        if best_size <= 69:
            break
            
    return best_blocks
