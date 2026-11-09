import itertools
import random
import time
import sys

_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # block_triple_indices: list of triple indices for each block
    block_triple_indices = [None] * n_blocks
    # triple_to_blocks: list of block indices for each triple
    triple_to_blocks = [[] for _ in range(n_triples)]
    # block_masks: bitmask of triples covered by each block
    block_masks = [0] * n_blocks
    
    for i, block in enumerate(all_blocks):
        indices = []
        mask = 0
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            indices.append(idx)
            triple_to_blocks[idx].append(i)
            mask |= (1 << idx)
        block_triple_indices[i] = indices
        block_masks[i] = mask
        
    full_mask = (1 << n_triples) - 1
    
    _CACHE[key] = (all_blocks, block_triple_indices, triple_to_blocks, block_masks, full_mask, n_triples, n_blocks)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, block_masks, full_mask, n_triples, seed):
    rng = random.Random(seed)
    uncovered_mask = full_mask
    blocks_used = []
    
    while uncovered_mask:
        # Pick a random uncovered triple
        # Find an index of a set bit in uncovered_mask
        # We can just pick a random triple from the list of all triples and check if it's in uncovered_mask
        # But that's inefficient.
        # Better: maintain a list of uncovered triple indices?
        # Or just pick a random block from those covering a random uncovered triple.
        
        # To pick a random uncovered triple efficiently:
        # We can store the list of uncovered indices.
        # But updating it is costly.
        # Alternative: Pick a random block index? No.
        
        # Let's just pick a random triple from the global list until we find an uncovered one?
        # N_triples is 2024.
        # On average, we might need a few tries.
        
        # Even better: maintain a list of uncovered triples.
        # Initially, list(range(n_triples)).
        # When we cover a block, we remove its triples from the list.
        # This requires efficient removal.
        
        # Let's stick to the bitmask and just iterate through a small subset of candidates?
        # Actually, the standard greedy picks the triple with the fewest covering blocks (Min Candidates).
        # This is often better than random.
        
        # Let's try Min Candidates heuristic.
        # For each uncovered triple, count how many blocks cover it.
        # This is expensive to compute from scratch.
        
        # Let's go back to Random Uncovered Triple, but implement it well.
        # We can keep a list `uncovered_list`.
        # When we add a block, we filter `uncovered_list` to remove covered triples.
        # `uncovered_list = [i for i in uncovered_list if not (uncovered_mask >> i) & 1]`?
        # No, we update `uncovered_mask`.
        # We can just rebuild `uncovered_list` from `uncovered_mask`?
        # `bin(uncovered_mask).count('1')` is small.
        
        # Let's just pick a random index from 0 to n_triples-1, and check if bit is set.
        # If not, try again.
        # This is O(1) expected if coverage is high.
        
        target_tri = None
        while target_tri is None:
            tri_idx = rng.randint(0, n_triples - 1)
            if (uncovered_mask >> tri_idx) & 1:
                target_tri = tri_idx
                
        candidate_blocks = triple_to_blocks[target_tri]
        
        best_block = None
        max_gain = -1
        
        for b_idx in candidate_blocks:
            gain = (block_masks[b_idx] & uncovered_mask).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx
            elif gain == max_gain and best_block is not None:
                if rng.random() < 0.5:
                    best_block = b_idx
                    
        if best_block is None:
            break
            
        blocks_used.append(best_block)
        uncovered_mask &= ~block_masks[best_block]
        
    return blocks_used

def _local_search(v, k, t, initial_blocks, block_triple_indices, triple_to_blocks, block_masks, full_mask, n_triples, n_blocks, deadline):
    current_blocks = list(initial_blocks)
    
    # Initialize counts
    triple_counts = [0] * n_triples
    for b in current_blocks:
        for idx in block_triple_indices[b]:
            triple_counts[idx] += 1
            
    # Verify
    if any(c == 0 for c in triple_counts):
        return current_blocks
        
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    rng = random.Random(12345)
    
    while time.time() < deadline:
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # 1. Try Deletion
        if current_blocks:
            pos = rng.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[pos]
            
            redundant = True
            for idx in block_triple_indices[b_out]:
                if triple_counts[idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                del current_blocks[pos]
                for idx in block_triple_indices[b_out]:
                    triple_counts[idx] -= 1
                continue
        
        # 2. Try 2-out-1-in
        if len(current_blocks) >= 2:
            # Pick two distinct random blocks
            pos1 = rng.randint(0, len(current_blocks) - 1)
            pos2 = rng.randint(0, len(current_blocks) - 1)
            while pos2 == pos1:
                pos2 = rng.randint(0, len(current_blocks) - 1)
                
            b1 = current_blocks[pos1]
            b2 = current_blocks[pos2]
            
            # Temporarily remove
            for idx in block_triple_indices[b1]:
                triple_counts[idx] -= 1
            for idx in block_triple_indices[b2]:
                triple_counts[idx] -= 1
                
            # Find uncovered triples
            req_mask = 0
            for idx in range(n_triples):
                if triple_counts[idx] == 0:
                    req_mask |= (1 << idx)
                    
            if req_mask == 0:
                # Both were redundant together
                # Remove both
                # Ensure pos1 < pos2 for safe deletion
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1
                del current_blocks[pos2]
                del current_blocks[pos1]
            else:
                # Try to find a block covering req_mask
                found = False
                # Iterate all blocks? Or just candidates?
                # Candidates must cover the triple with min count?
                # All req triples have count 0.
                # We need a block that contains ALL req triples.
                # This is a very strong constraint.
                # Let's check all blocks.
                for b_idx in range(n_blocks):
                    if (block_masks[b_idx] & req_mask) == req_mask:
                        # Check if b_idx is already in current_blocks
                        if b_idx not in current_blocks:
                            # Perform swap
                            # Remove b1, b2
                            if pos1 > pos2:
                                pos1, pos2 = pos2, pos1
                            del current_blocks[pos2]
                            del current_blocks[pos1]
                            # Add b_idx
                            current_blocks.append(b_idx)
                            for idx in block_triple_indices[b_idx]:
                                triple_counts[idx] += 1
                            found = True
                            break
                            
                if not found:
                    # Restore counts
                    for idx in block_triple_indices[b1]:
                        triple_counts[idx] += 1
                    for idx in block_triple_indices[b2]:
                        triple_counts[idx] += 1
                    continue # Try another move
            
        # 3. Try 1-out-1-in
        if current_blocks:
            pos = rng.randint(0, len(current_blocks) - 1)
            b_out = current_blocks[pos]
            
            # Temporarily remove
            for idx in block_triple_indices[b_out]:
                triple_counts[idx] -= 1
                
            req_mask = 0
            for idx in range(n_triples):
                if triple_counts[idx] == 0:
                    req_mask |= (1 << idx)
                    
            if req_mask == 0:
                del current_blocks[pos]
            else:
                found = False
                for b_idx in range(n_blocks):
                    if (block_masks[b_idx] & req_mask) == req_mask:
                        if b_idx not in current_blocks:
                            del current_blocks[pos]
                            current_blocks.append(b_idx)
                            for idx in block_triple_indices[b_idx]:
                                triple_counts[idx] += 1
                            found = True
                            break
                            
                if not found:
                    for idx in block_triple_indices[b_out]:
                        triple_counts[idx] += 1
                    continue

    return best_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_blocks, block_triple_indices, triple_to_blocks, block_masks, full_mask, n_triples, n_blocks = _precompute(v, k, t)
    
    best_blocks_global = None
    best_len_global = float('inf')
    
    seed = 0
    # Run multiple instances
    while time.time() < deadline - 10:
        greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, block_masks, full_mask, n_triples, seed)
        seed += 1
        
        if not greedy_blocks:
            continue
            
        ls_deadline = time.time() + 8
        ls_blocks = _local_search(
            v, k, t, 
            greedy_blocks, 
            block_triple_indices, 
            triple_to_blocks, 
            block_masks, 
            full_mask, 
            n_triples, 
            n_blocks, 
            ls_deadline
        )
        
        if len(ls_blocks) < best_len_global:
            best_len_global = len(ls_blocks)
            best_blocks_global = ls_blocks
            
    if best_blocks_global is None:
        greedy_blocks = _greedy_construct(v, k, t, all_blocks, block_triple_indices, triple_to_blocks, block_masks, full_mask, n_triples, 999)
        best_blocks_global = greedy_blocks

    result = [list(all_blocks[idx]) for idx in best_blocks_global]
    return result
