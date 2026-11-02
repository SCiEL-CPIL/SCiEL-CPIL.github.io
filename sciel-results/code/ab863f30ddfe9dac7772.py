import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    # Parameters for the task: v=24, k=7, t=3
    # But we should use the passed arguments.
    
    start_time = time.time()
    # Hard deadline to ensure we return before 120s
    deadline = start_time + 100.0

    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute 3D index lookup for speed (vectorized access)
    # idx_3d[a, b, c] is the index of the sorted triple (a,b,c)
    # We only care about distinct a,b,c.
    idx_3d = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a, b, c] = triple_to_idx[(t1, t2, t3)]
                
    def get_triple_idx(a, b, c):
        return idx_3d[a, b, c]

    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.uint8)
        num_uncovered = num_triples
        blocks = []
        
        # List of points
        points = list(range(v))
        
        while num_uncovered > 0:
            # Find first uncovered triple
            start_idx = int(np.argmax(uncovered))
            if uncovered[start_idx] == 0:
                break
                
            # Get the triple
            target_triple = triples[start_idx]
            block = list(target_triple)
            block_set = set(block)
            
            # Mark as covered
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                candidates = [x for x in points if x not in block_set]
                
                if not candidates:
                    break
                    
                # Shuffle candidates for random tie-breaking
                rng.shuffle(candidates)
                
                # Compute gains for all candidates using vectorized ops
                best_gain = -1
                best_point = -1
                
                # Precompute pairs in current block
                block_pairs = list(itertools.combinations(block, 2))
                
                # For each candidate, count how many uncovered triples it covers
                for x in candidates:
                    gain = 0
                    for (p1, p2) in block_pairs:
                        idx = get_triple_idx(x, p1, p2)
                        if uncovered[idx]:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        
                if best_point == -1:
                    break
                
                block.append(best_point)
                block_set.add(best_point)
                
                # Update uncovered status for all triples covered by the new point
                # for each pair in block that includes best_point
                for i in range(len(block)):
                    if block[i] == best_point:
                        continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point:
                            continue
                        p1 = block[i]
                        p2 = block[j]
                        idx = get_triple_idx(best_point, p1, p2)
                        if uncovered[idx]:
                            uncovered[idx] = 0
                            num_uncovered -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def trim_blocks(blocks):
        # Count frequency of each triple
        counts = np.zeros(num_triples, dtype=np.int32)
        
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active_blocks = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i, block in enumerate(blocks):
                if not active_blocks[i]:
                    continue
                essential = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active_blocks[i] = False
                    changed = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active_blocks[i]]

    def local_search_trim(blocks, random_state, time_limit=5.0):
        """
        Attempt to further reduce blocks by random reordering and deletion.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute triple counts for current blocks
        counts = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
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
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    found_removal = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
            if not found_removal:
                break
                
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    seed_counter = 0
    while time.time() < deadline:
        # Check if we have enough time for a full iteration
        if time.time() + 2 > deadline:
            break
            
        blocks = greedy_cover(seed_counter)
        
        # Quick check: if we already have a very good solution, try to trim
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        # Attempt to trim
        if time.time() < deadline - 1:
            trimmed = trim_blocks(blocks)
            if len(trimmed) < best_len:
                best_len = len(trimmed)
                best_blocks = list(trimmed)
                
            # Attempt local search if we are close to the known best
            if best_len <= 76 and time.time() < deadline - 2:
                ls_blocks = local_search_trim(trimmed, seed_counter, time_limit=1.0)
                if len(ls_blocks) < best_len:
                    best_len = len(ls_blocks)
                    best_blocks = list(ls_blocks)
                    
        if best_len == 69:
            break
            
        seed_counter += 1
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback: return empty if something went wrong, though greedy should always produce something
        return []
