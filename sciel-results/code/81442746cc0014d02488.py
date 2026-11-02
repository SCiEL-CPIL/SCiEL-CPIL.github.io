import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute triple index lookup for fast access
    # triple_idx[x][a][b] = index of t-subset {x, a, b} or -1 if invalid/not a subset
    triple_idx = np.full((v, v, v), -1, dtype=np.int32)
    for i, ts in enumerate(all_tsubsets):
        a, b, c = ts
        # Store for all permutations to allow arbitrary order lookup
        triple_idx[a, b, c] = i
        triple_idx[a, c, b] = i
        triple_idx[b, a, c] = i
        triple_idx[b, c, a] = i
        triple_idx[c, a, b] = i
        triple_idx[c, b, a] = i

    def greedy_cover_weighted(seed, time_limit):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsubsets, dtype=bool)
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        blocks = []
        
        while np.any(uncovered) and time.time() < time_limit:
            # Pick a random uncovered t-subset to start
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            start_combo = all_tsubsets[start_idx]
            block = list(start_combo)
            block_set = set(block)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1.0
                
                # Calculate weighted gain for each candidate x
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0.0
                    # Iterate over pairs in current block
                    # Since block size is small (<7), a simple loop is fine
                    for i in range(len(block)):
                        for j in range(i + 1, len(block)):
                            a = block[i]
                            b = block[j]
                            idx = triple_idx[x, a, b]
                            if idx >= 0:
                                if cover_count[idx] == 0:
                                    gain += 1.0
                                else:
                                    gain += 1.0 / cover_count[idx]
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_gain > 0:
                        if rng.random() < 0.5:
                            best_x = x
                
                if best_x < 0:
                    break
                
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) < k:
                continue
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
                    if cover_count[idx] == 1:
                        uncovered[idx] = False
        
        if not np.any(uncovered):
            return [list(b) for b in blocks]
        return None

    def remove_redundant(blocks, time_limit):
        if not blocks:
            return []
        if time.time() > time_limit:
            return blocks
        
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        block_tsubs = []
        
        # Precompute t-subsets for each block and update counts
        for b in blocks:
            ts_list = list(itertools.combinations(b, t))
            block_tsubs.append(ts_list)
            for ts in ts_list:
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
        
        result_blocks = list(blocks)
        result_tsubs = list(block_tsubs)
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(result_blocks)):
                if time.time() > time_limit:
                    break
                
                redundant = True
                # Check if any triple in this block is uniquely covered by it
                for ts in result_tsubs[i]:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block
                    for ts in result_tsubs[i]:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    
                    result_blocks.pop(i)
                    result_tsubs.pop(i)
                    changed = True
                    break
        
        return [list(b) for b in result_blocks]

    best_blocks = None
    best_count = float('inf')
    
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Allocate time for greedy and reduction
        # Greedy is fast, reduction is fast. Allow several iterations.
        attempt_time = min(remaining - 2, 8.0)
        
        blocks = greedy_cover_weighted(seed, time.time() + attempt_time)
        
        if blocks is not None:
            red_time = min(remaining - 2, 5.0)
            blocks = remove_redundant(blocks, time.time() + red_time)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        seed += 1
    
    if best_blocks is None:
        # Fallback: simple greedy without weighting, just to ensure a valid solution if weighted fails
        # This is unlikely if weighted works, but good for safety.
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []
        while np.any(uncovered):
            start_idx = np.where(uncovered)[0][0]
            start_combo = all_tsubsets[start_idx]
            block = list(start_combo)
            block_set = set(block)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for p1, p2 in itertools.combinations(block, 2):
                        idx = triple_idx[x, p1, p2]
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered[idx] = False
        
        # Try to reduce
        blocks = remove_redundant(blocks, time.time() + 1.0)
        return blocks

    return best_blocks
