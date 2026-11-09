import itertools
import random
import time
import sys

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # block_masks[i] is a bitmask of triples covered by block i
    block_masks = [0] * n_blocks
    # blocks_containing[j] is a list of block indices containing triple j
    blocks_containing = [[] for _ in range(n_triples)]
    # block_triples[i] is a list of triple indices in block i
    block_triples = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
            blocks_containing[idx].append(i)
            block_triples[i].append(idx)
        block_masks[i] = mask
        
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples):
    rng = random.Random(seed)
    uncovered = full_mask
    blocks_used = []
    used_mask = 0
    active_blocks = set(range(n_blocks))
    
    while uncovered:
        best_block_idx = -1
        max_gain = 0
        uncovered_count = uncovered.bit_count()
        
        for idx in active_blocks:
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block_idx = idx
                if gain == uncovered_count:
                    break
            
        if best_block_idx == -1:
            break
            
        blocks_used.append(best_block_idx)
        used_mask |= block_masks[best_block_idx]
        uncovered &= ~used_mask
        active_blocks.remove(best_block_idx)
        
    return blocks_used

def _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, deadline):
    rng = random.Random(seed)
    
    selected = set(initial_blocks)
    coverage_count = [0] * n_triples
    
    for b in selected:
        for tri_idx in block_triples[b]:
            coverage_count[tri_idx] += 1
    
    best_size = len(selected)
    best_selected = set(selected)
    
    no_improve_time = time.time()
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to remove a redundant block
        if selected:
            items = list(selected)
            rng.shuffle(items)
            for b in items:
                redundant = True
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for tri_idx in block_triples[b]:
                        coverage_count[tri_idx] -= 1
                    selected.remove(b)
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    no_improve_time = time.time()
                    break
        
        if not improved:
            # 2. Try swaps: find block with unique triples, replace with better one
            swap_done = False
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            for b in sel_list:
                # Find unique triples for b
                unique_tris = []
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        unique_tris.append(tri_idx)
                
                if not unique_tris:
                    continue
                
                # Pick a random unique triple to find candidates
                tri_idx = rng.choice(unique_tris)
                candidates = blocks_containing[tri_idx]
                
                if len(candidates) <= 1:
                    continue
                
                # Shuffle candidates and try a few
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for b_prime in cand_list[:50]:  # Limit candidates checked
                    if b_prime in selected:
                        continue
                    
                    # Check if b_prime covers all unique triples of b
                    covers_all = True
                    for ut in unique_tris:
                        if not (block_masks[b_prime] & (1 << ut)):
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Perform swap
                    for tri_idx2 in block_triples[b]:
                        coverage_count[tri_idx2] -= 1
                    selected.remove(b)
                    
                    for tri_idx2 in block_triples[b_prime]:
                        coverage_count[tri_idx2] += 1
                    selected.add(b_prime)
                    
                    swap_done = True
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    no_improve_time = time.time()
                    break
                
                if swap_done:
                    break
        
        if not improved:
            # No improvement in this iteration
            if time.time() - no_improve_time > 5.0:
                break
    
    return list(best_selected)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Precompute
    all_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    
    # Run multiple restarts with time-sliced local searches
    while time.time() < deadline - 5:
        # 1. Greedy (fast)
        initial_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples)
        
        if not initial_blocks:
            seed += 1
            continue
        
        # 2. Local Search with sub-deadline
        remaining = deadline - time.time()
        ls_deadline = min(time.time() + min(15.0, remaining - 2.0), deadline - 3.0)
        
        if ls_deadline <= time.time():
            break
            
        blocks = _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, ls_deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            print(f"New best: {best_len} blocks at t={time.time()-start_time:.1f}s")
        
        seed += 1
    
    if best_blocks is None:
        blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, 999, block_triples)
        best_blocks = blocks
    
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
