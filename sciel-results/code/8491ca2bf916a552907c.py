import itertools
import random
import time
import numpy as np

_PRECOMPUTED = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _PRECOMPUTED:
        return _PRECOMPUTED[key]
    
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each triple, list of blocks that contain it
    blocks_by_triple = [[] for _ in range(num_triples)]
    for b_idx, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            blocks_by_triple[t_idx].append(b_idx)
    
    _PRECOMPUTED[key] = {
        'all_blocks': all_blocks,
        'blocks_by_triple': blocks_by_triple,
        'num_triples': num_triples,
        'triple_to_idx': triple_to_idx,
        'num_blocks': num_blocks
    }
    return _PRECOMPUTED[key]

def construct(v, k, t):
    if v != 24 or k != 7 or t != 3:
        return _simple_greedy(v, k, t)
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    data = _precompute(v, k, t)
    all_blocks = data['all_blocks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy instances with different strategies
    run_id = 0
    while time.time() < deadline:
        # Alternate between different greedy strategies
        if run_id % 3 == 0:
            blocks = _greedy_criticality(v, k, t, data, random_seed=run_id, mode='random_triple')
        elif run_id % 3 == 1:
            blocks = _greedy_criticality(v, k, t, data, random_seed=run_id, mode='rarest_triple')
        else:
            blocks = _greedy_criticality(v, k, t, data, random_seed=run_id, mode='weighted')
        
        run_id += 1
        
        # Apply reduction
        blocks = _reduce_blocks(blocks, v, t)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # If we hit the known best, we can try harder
            if best_count <= 76:
                pass
    
    if best_blocks is None:
        best_blocks = _simple_greedy(v, k, t)
    
    # Final reduction
    best_blocks = _reduce_blocks(best_blocks, v, t)
    
    return [list(b) for b in best_blocks]

def _greedy_criticality(v, k, t, data, random_seed, mode):
    rng = random.Random(random_seed)
    all_blocks = data['all_blocks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    
    # Track uncovered triples
    uncovered = [True] * num_triples
    num_uncovered = num_triples
    
    # Count how many blocks cover each triple (for rarity)
    triple_freq = [len(blocks_by_triple[i]) for i in range(num_triples)]
    
    blocks = []
    
    while num_uncovered > 0:
        # Select target triple based on mode
        if mode == 'random_triple':
            # Pick a random uncovered triple
            while True:
                t_idx = rng.randrange(num_triples)
                if uncovered[t_idx]:
                    break
        elif mode == 'rarest_triple':
            # Pick the uncovered triple with fewest covering blocks
            best_t_idx = -1
            best_freq = float('inf')
            # Sample a subset of uncovered triples for efficiency
            sample_size = min(50, num_uncovered)
            for _ in range(sample_size):
                while True:
                    t_idx = rng.randrange(num_triples)
                    if uncovered[t_idx]:
                        break
                if triple_freq[t_idx] < best_freq:
                    best_freq = triple_freq[t_idx]
                    best_t_idx = t_idx
            t_idx = best_t_idx
        else:  # weighted
            # Pick uncovered triple weighted by inverse frequency
            while True:
                t_idx = rng.randrange(num_triples)
                if uncovered[t_idx]:
                    break
        
        candidates = blocks_by_triple[t_idx]
        
        # Score each candidate block by how many new triples it covers
        best_block_idx = None
        best_gain = -1
        
        for b_idx in candidates:
            # Count how many uncovered triples this block covers
            gain = 0
            for sub in all_blocks[b_idx]:
                pass  # This is too slow, need precomputed masks
            
            # Actually, let's precompute the triples in each block
            # For now, use a simpler approach
            pass
        
        # Better approach: precompute for each block which triples it covers
        # But that's memory intensive. Let's compute on the fly but efficiently.
        
        # For each candidate block, count uncovered triples it covers
        best_block_idx = None
        best_gain = -1
        
        for b_idx in candidates:
            block = all_blocks[b_idx]
            gain = 0
            for sub in itertools.combinations(block, t):
                t_i = data['triple_to_idx'][sub]
                if uncovered[t_i]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
            elif gain == best_gain and rng.random() < 0.3:
                best_block_idx = b_idx
        
        # Add the selected block
        selected_block = all_blocks[best_block_idx]
        blocks.append(selected_block)
        
        # Update uncovered status
        for sub in itertools.combinations(selected_block, t):
            t_i = data['triple_to_idx'][sub]
            if uncovered[t_i]:
                uncovered[t_i] = False
                num_uncovered -= 1
    
    return blocks

def _reduce_blocks(blocks, v, t):
    if not blocks:
        return []
        
    # Map each triple to list of block indices covering it
    triple_to_blocks = {}
    for i, b in enumerate(blocks):
        for sub in itertools.combinations(b, t):
            if sub not in triple_to_blocks:
                triple_to_blocks[sub] = []
            triple_to_blocks[sub].append(i)
            
    active = [True] * len(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if not active[i]:
                continue
            redundant = True
            for sub in itertools.combinations(blocks[i], t):
                covered = False
                for j in triple_to_blocks[sub]:
                    if j != i and active[j]:
                        covered = True
                        break
                if not covered:
                    redundant = False
                    break
            if redundant:
                active[i] = False
                changed = True
                
    return [blocks[i] for i in range(len(blocks)) if active[i]]

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        target = min(uncovered)
        best_block = None
        best_gain = -1
        remaining_points = [p for p in range(v) if p not in target]
        need = k - t
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        blocks.append(best_block)
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks
