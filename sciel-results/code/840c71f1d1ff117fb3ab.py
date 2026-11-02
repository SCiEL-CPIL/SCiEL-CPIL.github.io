import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    def greedy_cover(seed, time_limit):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered) and time.time() < time_limit:
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            start_combo = all_tsubsets[start_idx]
            block = list(start_combo)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                block_pairs = list(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for p1, p2 in block_pairs:
                        triple = tuple(sorted((x, p1, p2)))
                        idx = tsubset_to_idx.get(triple, -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_gain > 0:
                        if rng.random() < 0.5:
                            best_x = x
                
                if best_x < 0:
                    break
                block.append(best_x)
            
            if len(block) < k:
                continue
            
            block.sort()
            blocks.append(tuple(block))
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
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
                for ts in result_tsubs[i]:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    for ts in result_tsubs[i]:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    result_blocks.pop(i)
                    result_tsubs.pop(i)
                    changed = True
                    break
        
        return [list(b) for b in result_blocks]

    def local_search(blocks, time_limit, seed):
        """Try block swaps and removals to reduce block count."""
        rng = random.Random(seed)
        blocks = [list(b) for b in blocks]
        
        # Build coverage map
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        block_tsubs = []
        for b in blocks:
            ts_list = list(itertools.combinations(b, t))
            block_tsubs.append(ts_list)
            for ts in ts_list:
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
        
        # First pass: remove redundant blocks
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(blocks)):
                if time.time() > time_limit:
                    return [list(b) for b in blocks]
                redundant = True
                for ts in block_tsubs[i]:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts in block_tsubs[i]:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    blocks.pop(i)
                    block_tsubs.pop(i)
                    changed = True
                    break
        
        # Second pass: try swapping blocks to find better covers
        # For each pair of blocks, try replacing one with a new block that covers
        # more uncovered t-subsets while maintaining coverage
        best_blocks = [list(b) for b in blocks]
        best_count = len(blocks)
        
        while time.time() < time_limit:
            # Try random block replacement
            if len(blocks) == 0:
                break
            
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_tsubs = block_tsubs[i]
            
            # Temporarily remove block i
            for ts in old_tsubs:
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] -= 1
            
            # Find t-subsets that are now uncovered (cover_count == 0)
            newly_uncovered = [ts for ts in old_tsubs 
                              if tsubset_to_idx.get(ts, -1) >= 0 and cover_count[tsubset_to_idx[ts]] == 0]
            
            if not newly_uncovered:
                # Block was redundant, keep it removed
                blocks.pop(i)
                block_tsubs.pop(i)
                continue
            
            # Try to find a new block that covers all newly_uncovered t-subsets
            # plus some additional uncovered ones
            best_new_block = None
            best_gain = -1
            
            # Sample candidate blocks
            for _ in range(20):
                # Build a candidate block starting from points in newly_uncovered
                candidate = list(rng.sample(range(v), min(k, 3)))
                while len(candidate) < k:
                    # Add point that maximizes coverage of newly_uncovered
                    best_x = -1
                    best_g = -1
                    for x in range(v):
                        if x in candidate:
                            continue
                        g = 0
                        for ts in newly_uncovered:
                            if x in ts and all(p in candidate or p == x for p in ts if p != x):
                                # Check if adding x helps cover this t-subset
                                pass
                        # Simpler: just check if x is in any newly_uncovered t-subset
                        if any(x in ts for ts in newly_uncovered):
                            g += 1
                        if g > best_g:
                            best_g = g
                            best_x = x
                    if best_x >= 0:
                        candidate.append(best_x)
                    else:
                        break
                
                if len(candidate) == k:
                    candidate.sort()
                    # Check if this candidate covers all newly_uncovered
                    cand_set = set(candidate)
                    covers_all = all(all(p in cand_set for p in ts) for ts in newly_uncovered)
                    if covers_all:
                        # Count additional coverage
                        extra = 0
                        for ts in itertools.combinations(candidate, t):
                            idx = tsubset_to_idx.get(ts, -1)
                            if idx >= 0 and cover_count[idx] == 0:
                                extra += 1
                        if extra > best_gain:
                            best_gain = extra
                            best_new_block = candidate
            
            if best_new_block is not None:
                # Replace old block with new block
                blocks[i] = best_new_block
                new_tsubs = list(itertools.combinations(best_new_block, t))
                block_tsubs[i] = new_tsubs
                for ts in new_tsubs:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0:
                        cover_count[idx] += 1
            else:
                # Restore old block
                for ts in old_tsubs:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0:
                        cover_count[idx] += 1
        
        return [list(b) for b in blocks]

    best_blocks = None
    best_count = float('inf')
    
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        greedy_time = min(remaining - 5, 8)
        local_time = min(remaining - greedy_time - 2, 15)
        
        blocks = greedy_cover(seed, time.time() + greedy_time)
        
        if blocks is not None:
            # Remove redundant blocks
            red_time = min(remaining - greedy_time - 2, 5)
            blocks = remove_redundant(blocks, time.time() + red_time)
            
            # Local search with block swaps
            if time.time() < deadline - 3:
                blocks = local_search(blocks, time.time() + local_time, seed)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        seed += 1
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []
        while np.any(uncovered):
            start_idx = np.where(uncovered)[0][0]
            start_combo = all_tsubsets[start_idx]
            block = list(start_combo)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                block_pairs = list(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for p1, p2 in block_pairs:
                        triple = tuple(sorted((x, p1, p2)))
                        idx = tsubset_to_idx.get(triple, -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered[idx] = False
        
        return blocks

    return best_blocks
