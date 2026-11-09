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
        best_candidates = []
        uncovered_count = uncovered.bit_count()
        
        for idx in active_blocks:
            gain = (block_masks[idx] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block_idx = idx
                best_candidates = [idx]
                if gain == uncovered_count:
                    break
            elif gain == max_gain:
                best_candidates.append(idx)
            
        if best_block_idx == -1:
            break
            
        # Randomly select from best candidates to ensure diversity
        selected_idx = rng.choice(best_candidates)
        
        blocks_used.append(selected_idx)
        used_mask |= block_masks[selected_idx]
        uncovered &= ~used_mask
        active_blocks.remove(selected_idx)
        
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
    penalty_weight = 1.0  # Penalty per block in the objective
    
    def compute_objective(sel, cov):
        """Objective = uncovered_triples + penalty * |selected|"""
        uncov = sum(1 for i in range(n_triples) if cov[i] == 0)
        return uncov + penalty_weight * len(sel)
    
    current_obj = compute_objective(selected, coverage_count)
    
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
                    current_obj = compute_objective(selected, coverage_count)
                    no_improve_time = time.time()
                    break
        
        if not improved:
            # 2. WalkSAT-style: try moves that improve the weighted objective
            #    Move types: (a) add a block, (b) remove a non-redundant block, (c) swap
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            move_done = False
            
            # Try removals of blocks that aren't strictly redundant but might reduce objective
            for b in sel_list:
                # Compute delta if we remove b
                # Loss: triples that become uncovered (cov was 1)
                # Gain: penalty_weight reduction
                loss = 0
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        loss += 1
                delta = -loss + penalty_weight  # objective change
                
                if delta < 0:  # improves objective
                    for tri_idx in block_triples[b]:
                        coverage_count[tri_idx] -= 1
                    selected.remove(b)
                    move_done = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    current_obj = compute_objective(selected, coverage_count)
                    no_improve_time = time.time()
                    break
            
            if move_done:
                improved = True
                continue
            
            # Try swaps: replace b with b' where b' covers more unique triples of b
            for b in sel_list:
                unique_tris = []
                for tri_idx in block_triples[b]:
                    if coverage_count[tri_idx] == 1:
                        unique_tris.append(tri_idx)
                
                if not unique_tris:
                    continue
                
                tri_idx = rng.choice(unique_tris)
                candidates = blocks_containing[tri_idx]
                
                if len(candidates) <= 1:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                best_swap = None
                best_delta = 0
                
                for b_prime in cand_list[:80]:
                    if b_prime in selected:
                        continue
                    
                    # Compute delta for swap: remove b, add b_prime
                    # Loss from removing b: triples with cov==1 in b not covered by b_prime
                    # Gain from adding b_prime: triples with cov==0 in b_prime
                    loss = 0
                    gain = 0
                    for tri_idx2 in block_triples[b]:
                        if coverage_count[tri_idx2] == 1:
                            # Check if b_prime also covers it
                            if not (block_masks[b_prime] & (1 << tri_idx2)):
                                loss += 1
                    for tri_idx2 in block_triples[b_prime]:
                        if coverage_count[tri_idx2] == 0:
                            gain += 1
                    
                    delta = loss - gain  # net change in uncovered count (we want negative)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = b_prime
                
                if best_swap is not None and best_delta < 0:
                    b_prime = best_swap
                    # Perform swap
                    for tri_idx2 in block_triples[b]:
                        coverage_count[tri_idx2] -= 1
                    selected.remove(b)
                    
                    for tri_idx2 in block_triples[b_prime]:
                        coverage_count[tri_idx2] += 1
                    selected.add(b_prime)
                    
                    move_done = True
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                    current_obj = compute_objective(selected, coverage_count)
                    no_improve_time = time.time()
                    break
            
            if move_done:
                continue
            
            # Try additions of blocks that cover many uncovered triples
            # (useful if we're in a local minimum with some uncovered triples)
            uncovered_tris = [i for i in range(n_triples) if coverage_count[i] == 0]
            if uncovered_tris and len(selected) < best_size + 3:
                # Find block covering most uncovered
                best_add = None
                best_gain = 0
                for _ in range(50):  # sample
                    idx = rng.randint(0, n_blocks - 1)
                    if idx in selected:
                        continue
                    gain = 0
                    for tri_idx2 in block_triples[idx]:
                        if coverage_count[tri_idx2] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_add = idx
                
                if best_add is not None and best_gain > penalty_weight:
                    for tri_idx2 in block_triples[best_add]:
                        coverage_count[tri_idx2] += 1
                    selected.add(best_add)
                    move_done = True
                    improved = True
                    current_obj = compute_objective(selected, coverage_count)
                    no_improve_time = time.time()
            
            if not improved:
                if time.time() - no_improve_time > 8.0:
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
        ls_deadline = min(time.time() + min(12.0, remaining - 2.0), deadline - 3.0)
        
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
