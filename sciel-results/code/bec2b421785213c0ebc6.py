import itertools
import time
import random
import sys
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin for safety
    
    # Precompute
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # idx_3d for fast lookup
    idx_3d = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a != b and a != c and b != c:
                    t1, t2, t3 = sorted((a, b, c))
                    idx_3d[a, b, c] = triple_to_idx[(t1, t2, t3)]

    def get_triple_idx(a, b, c):
        return idx_3d[a, b, c]

    # Precompute all pairs for each block size? No, block size varies.
    # But we can precompute the list of triple indices for a specific set of points.
    # Let's stick to on-the-fly for gain, but optimize the inner loop.
    
    # For speed, precompute the list of triple indices for each pair (i, j)
    # triple_indices_for_pair[i][j] = list of (idx, k) such that (i, j, k) is a triple
    # This helps in checking if a triple is covered? No, we need to check if a specific triple is uncovered.
    
    # Alternative: Precompute for each point x, the list of triples containing x.
    # triples_containing_point[x] = list of triple indices
    
    # Let's try a different optimization for the greedy step.
    # Instead of scanning all candidates, can we be smarter?
    # For now, let's stick to the current structure but try to make it faster.
    
    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.uint8)
        num_uncovered = num_triples
        blocks = []
        points = list(range(v))
        
        while num_uncovered > 0:
            # Pick random uncovered triple
            while True:
                idx = rng.randint(0, num_triples - 1)
                if uncovered[idx]:
                    break
            
            target_triple = triples[idx]
            block = list(target_triple)
            block_set = set(block)
            
            uncovered[idx] = 0
            num_uncovered -= 1
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in points if x not in block_set]
                
                # Shuffle for tie-breaking
                rng.shuffle(candidates)
                
                # Calculate gains
                # Optimization: Only consider candidates that might have high gain?
                # Hard to know without checking.
                
                for x in candidates:
                    gain = 0
                    # Pairs in block
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            p1 = block[i]
                            p2 = block[j]
                            t_idx = get_triple_idx(x, p1, p2)
                            if uncovered[t_idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1:
                    break
                
                block.append(best_point)
                block_set.add(best_point)
                
                # Update uncovered
                for i in range(len(block)):
                    if block[i] == best_point: continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point: continue
                        p1 = block[i]
                        p2 = block[j]
                        t_idx = get_triple_idx(best_point, p1, p2)
                        if uncovered[t_idx]:
                            uncovered[t_idx] = 0
                            num_uncovered -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def trim_blocks(blocks):
        counts = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not active[i]: continue
                essential = False
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    changed = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def local_search(blocks, random_state, time_limit=2.0):
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute counts
        counts = np.zeros(num_triples, dtype=np.int32)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active = [True] * len(blocks)
        
        # First pass: trim
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not active[i]: continue
                essential = False
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    changed = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                        
        blocks = [blocks[i] for i in range(len(blocks)) if active[i]]
        
        best_len = len(blocks)
        
        # Local Search: Try to swap points to reduce redundancy
        while time.time() - start_time < time_limit:
            if len(blocks) == 0: break
            
            # Pick a random block to perturb
            b_idx = rng.randint(0, len(blocks) - 1)
            block = list(blocks[b_idx])
            
            # Pick a random point in block to replace
            p_idx = rng.randint(0, len(block) - 1)
            old_point = block[p_idx]
            
            # Pick a new point
            new_point = rng.randint(0, v - 1)
            if new_point == old_point: continue
            if new_point in block: continue
            
            new_block = block[:]
            new_block[p_idx] = new_point
            new_block.sort()
            
            old_block = blocks[b_idx]
            blocks[b_idx] = tuple(new_block)
            
            trimmed = trim_blocks(blocks)
            
            if len(trimmed) < best_len:
                best_len = len(trimmed)
                blocks = trimmed
            else:
                blocks[b_idx] = old_block
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    seed_counter = 0
    
    while time.time() < deadline:
        if time.time() + 5 > deadline: break
        
        blocks = greedy_cover(seed_counter)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        trimmed = trim_blocks(blocks)
        if len(trimmed) < best_len:
            best_len = len(trimmed)
            best_blocks = list(trimmed)
            
        if best_len <= 76 and time.time() < deadline - 2:
            ls_blocks = local_search(trimmed, seed_counter, time_limit=1.0)
            if len(ls_blocks) < best_len:
                best_len = len(ls_blocks)
                best_blocks = list(ls_blocks)
                
        if best_len == 69: break
        seed_counter += 1
        
    if best_blocks:
        return [list(b) for b in best_blocks]
    return []
