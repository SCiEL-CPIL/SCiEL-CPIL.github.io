import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute for each block (k-subset) the list of triple indices it covers
    # This allows O(1) lookup when we need to update coverage
    all_blocks = list(itertools.combinations(range(v), k))
    num_all_blocks = len(all_blocks)
    
    # For each block, precompute the triple indices it covers
    block_triples = []
    for blk in all_blocks:
        tris = [triple_to_idx[tr] for tr in itertools.combinations(blk, t)]
        block_triples.append(tris)
    
    # Map from block tuple to its index
    block_to_idx = {blk: i for i, blk in enumerate(all_blocks)}
    
    # Precompute: for each point x and each pair (a,b), the triple index
    # triple_idx[x, a, b] = index of sorted triple containing x, a, b
    triple_idx = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                triple_idx[a, b, c] = triple_to_idx[(t1, t2, t3)]
    
    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.uint8)
        num_uncovered = num_triples
        blocks = []
        points = list(range(v))
        
        while num_uncovered > 0:
            # Find first uncovered triple
            start_idx = int(np.argmax(uncovered))
            if uncovered[start_idx] == 0:
                break
                
            target_triple = triples[start_idx]
            block = list(target_triple)
            block_set = set(block)
            
            # Mark the target triple as covered
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k - t points one by one
            while len(block) < k:
                candidates = [x for x in points if x not in block_set]
                if not candidates:
                    break
                    
                # For each candidate, compute gain: number of uncovered triples it would cover
                # A candidate x covers triples (x, a, b) for all pairs (a,b) in current block
                block_pairs = list(itertools.combinations(block, 2))
                
                best_gain = -1
                best_point = -1
                best_gain_idx = -1
                
                for x in candidates:
                    gain = 0
                    for (p1, p2) in block_pairs:
                        idx = triple_idx[x, p1, p2]
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        best_gain_idx = x
                        
                if best_point == -1:
                    break
                
                block.append(best_point)
                block_set.add(best_point)
                
                # Update uncovered: mark all triples (best_point, p1, p2) as covered
                for (p1, p2) in itertools.combinations(block, 2):
                    if best_point in (p1, p2):
                        continue
                    idx = triple_idx[best_point, p1, p2]
                    if idx >= 0 and uncovered[idx]:
                        uncovered[idx] = 0
                        num_uncovered -= 1
                        
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def trim_blocks(blocks):
        counts = np.zeros(num_triples, dtype=np.int32)
        
        for block in blocks:
            for tr in itertools.combinations(block, t):
                idx = triple_to_idx[tr]
                counts[idx] += 1
                
        active_blocks = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i, block in enumerate(blocks):
                if not active_blocks[i]:
                    continue
                essential = False
                for tr in itertools.combinations(block, t):
                    idx = triple_to_idx[tr]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active_blocks[i] = False
                    changed = True
                    for tr in itertools.combinations(block, t):
                        idx = triple_to_idx[tr]
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active_blocks[i]]

    def local_search_trim(blocks, random_state, time_limit=5.0):
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        counts = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for tr in itertools.combinations(block, t):
                idx = triple_to_idx[tr]
                counts[idx] += 1
                
        active = [True] * len(blocks)
        
        while time.time() - start_time < time_limit:
            active_indices = [i for i in range(len(blocks)) if active[i]]
            if not active_indices:
                break
            rng.shuffle(active_indices)
            
            found_removal = False
            for i in active_indices:
                if not active[i]:
                    continue
                essential = False
                for tr in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[tr]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    found_removal = True
                    for tr in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[tr]
                        counts[idx] -= 1
            if not found_removal:
                break
                
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def swap_local_search(blocks, random_state, time_limit=10.0):
        """
        Try to replace a block with another that covers all its essential triples
        and possibly more, reducing total block count.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute which triples each block covers
        block_cover = []
        for block in blocks:
            tris = set()
            for tr in itertools.combinations(block, t):
                tris.add(tr)
            block_cover.append(tris)
        
        # Compute triple counts
        counts = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for tr in itertools.combinations(block, t):
                idx = triple_to_idx[tr]
                counts[idx] += 1
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(len(blocks)):
                if time.time() - start_time >= time_limit:
                    break
                
                # Find essential triples of block i
                essential_triples = set()
                for tr in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[tr]
                    if counts[idx] == 1:
                        essential_triples.add(tr)
                
                if not essential_triples:
                    # Block can be removed
                    del blocks[i]
                    del block_cover[i]
                    for tr in essential_triples:
                        idx = triple_to_idx[tr]
                        counts[idx] -= 1
                    improved = True
                    break
                
                # Try to find a replacement block that covers all essential triples
                # and has more total coverage
                best_replacement = -1
                best_score = -1
                
                for j, new_block in enumerate(all_blocks):
                    if j == i:
                        continue
                    new_tris = set()
                    for tr in itertools.combinations(new_block, t):
                        new_tris.add(tr)
                    
                    # Check if new block covers all essential triples
                    if not essential_triples.issubset(new_tris):
                        continue
                    
                    # Score: number of additional triples covered
                    additional = len(new_tris - block_cover[i])
                    if additional > best_score:
                        best_score = additional
                        best_replacement = j
                
                if best_replacement != -1:
                    # Update counts
                    for tr in block_cover[i]:
                        idx = triple_to_idx[tr]
                        counts[idx] -= 1
                    for tr in all_blocks[best_replacement]:
                        pass  # Will handle below
                    
                    new_block_tris = set()
                    for tr in itertools.combinations(all_blocks[best_replacement], t):
                        new_block_tris.add(tr)
                        idx = triple_to_idx[tr]
                        counts[idx] += 1
                    
                    blocks[i] = all_blocks[best_replacement]
                    block_cover[i] = new_block_tris
                    improved = True
        
        return blocks

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    seed_counter = 0
    while time.time() < deadline:
        if time.time() + 3 > deadline:
            break
            
        blocks = greedy_cover(seed_counter)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
        
        if time.time() < deadline - 2:
            trimmed = trim_blocks(blocks)
            if len(trimmed) < best_len:
                best_len = len(trimmed)
                best_blocks = list(trimmed)
            
            if best_len <= 76 and time.time() < deadline - 5:
                ls_blocks = local_search_trim(trimmed, seed_counter, time_limit=2.0)
                if len(ls_blocks) < best_len:
                    best_len = len(ls_blocks)
                    best_blocks = list(ls_blocks)
            
            if best_len <= 78 and time.time() < deadline - 10:
                sw_blocks = swap_local_search(trimmed, seed_counter, time_limit=5.0)
                if len(sw_blocks) < best_len:
                    best_len = len(sw_blocks)
                    best_blocks = list(sw_blocks)
                    
        if best_len == 69:
            break
            
        seed_counter += 1
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
