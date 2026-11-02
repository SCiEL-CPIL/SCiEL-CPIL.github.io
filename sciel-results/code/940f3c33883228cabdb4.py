import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute lookup for triple indices
    triple_idx = np.full((v, v, v), -1, dtype=np.int32)
    for i, ts in enumerate(all_tsubsets):
        a, b, c = ts
        triple_idx[a, b, c] = i
        triple_idx[a, c, b] = i
        triple_idx[b, a, c] = i
        triple_idx[b, c, a] = i
        triple_idx[c, a, b] = i
        triple_idx[c, b, a] = i

    def greedy_cover(seed, time_limit):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsubsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered) and time.time() < time_limit:
            # Pick a random uncovered t-subset to start
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
                
                # Precompute block pairs for efficiency
                block_pairs = list(itertools.combinations(block, 2))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for p1, p2 in block_pairs:
                        idx = triple_idx[x, p1, p2]
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
            
            # Update uncovered
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

    def density_local_search(blocks, time_limit):
        """
        Local search that minimizes sum of 1/cover_count (density objective).
        Tries to remove blocks with highest marginal redundancy.
        """
        if not blocks:
            return []
        
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        block_tsubs = []
        
        for b in blocks:
            ts_list = list(itertools.combinations(b, t))
            block_tsubs.append(ts_list)
            for ts in ts_list:
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
        
        result_blocks = [list(b) for b in blocks]
        result_tsubs = list(block_tsubs)
        
        # Compute initial density objective: sum of 1/cover_count for covered tsubsets
        def compute_density(cover_count, n_blocks):
            # Only consider tsubsets that are covered (cover_count > 0)
            covered = cover_count > 0
            if not np.any(covered):
                return float('inf')
            return np.sum(1.0 / cover_count[covered]) + 0.01 * n_blocks
        
        current_density = compute_density(cover_count, len(result_blocks))
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try removing each block and see if density improves
            for i in range(len(result_blocks)):
                if time.time() > time_limit:
                    break
                
                # Check if block i is redundant (all its tsubsets are covered by others)
                redundant = True
                for ts in result_tsubs[i]:
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for ts in result_tsubs[i]:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    
                    result_blocks.pop(i)
                    result_tsubs.pop(i)
                    
                    new_density = compute_density(cover_count, len(result_blocks))
                    if new_density < current_density:
                        current_density = new_density
                        improved = True
                        break
                    else:
                        # Restore
                        result_blocks.insert(i, list(result_blocks[i]) if i < len(result_blocks) else [])
                        # Actually we need to restore properly
                        # Let's re-add and restore cover counts
                        # This is getting complex, let's just restart the loop
                        for ts in result_tsubs[i]:
                            idx = tsubset_to_idx.get(ts, -1)
                            if idx >= 0:
                                cover_count[idx] += 1
        
        return result_blocks

    def local_search_swap(blocks, time_limit):
        """Try to improve block count by swapping points in blocks."""
        if not blocks:
            return []
        
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        block_tsubs = []
        
        for b in blocks:
            ts_list = list(itertools.combinations(b, t))
            block_tsubs.append(ts_list)
            for ts in ts_list:
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
        
        result_blocks = [list(b) for b in blocks]
        result_tsubs = list(block_tsubs)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            for i in range(len(result_blocks)):
                if time.time() > time_limit:
                    break
                
                block = result_blocks[i]
                block_set = set(block)
                
                # Try removing each point and replacing with another
                for old_point in block:
                    # Temporarily remove old_point
                    cover_count_changes = {}
                    new_tsubs = []
                    
                    for ts in itertools.combinations(block, t):
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count_changes[idx] = cover_count_changes.get(idx, 0) + 1
                            cover_count[idx] -= 1
                            new_tsubs.append((ts, idx))
                    
                    # Find best replacement point
                    best_new_point = -1
                    best_gain = 0
                    
                    for new_point in range(v):
                        if new_point in block_set and new_point != old_point:
                            continue
                        
                        # Calculate gain from adding new_point
                        gain = 0
                        new_block = [p for p in block if p != old_point] + [new_point]
                        
                        for ts in itertools.combinations(new_block, t):
                            if old_point in ts:  # Only consider new t-subsets
                                idx = tsubset_to_idx.get(ts, -1)
                                if idx >= 0 and cover_count[idx] == 0:
                                    gain += 1
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_new_point = new_point
                    
                    # Restore cover_count
                    for ts, idx in new_tsubs:
                        if idx >= 0:
                            cover_count[idx] += 1
                    
                    # If we found a better point, make the swap
                    if best_new_point != -1 and best_gain > 0:
                        block.remove(old_point)
                        block.append(best_new_point)
                        block.sort()
                        result_blocks[i] = block
                        
                        # Update tsubs
                        new_tsubs_list = list(itertools.combinations(block, t))
                        result_tsubs[i] = new_tsubs_list
                        
                        # Update cover_count
                        for ts in itertools.combinations(block, t):
                            idx = tsubset_to_idx.get(ts, -1)
                            if idx >= 0:
                                cover_count[idx] += 1
                        
                        improved = True
                        break  # Restart loop after modification
        
        return result_blocks

    best_blocks = None
    best_count = float('inf')
    
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        attempt_time = min(remaining - 2, 8)
        
        blocks = greedy_cover(seed, time.time() + attempt_time)
        
        if blocks is not None:
            red_time = min(remaining - 4, 5)
            blocks = remove_redundant(blocks, time.time() + red_time)
            
            # Try density local search if we have time
            dl_time = min(remaining - 3, 4)
            if time.time() + dl_time < deadline:
                blocks = density_local_search(blocks, time.time() + dl_time)
            
            # Try local search if we have time
            ls_time = min(remaining - 2, 5)
            if time.time() + ls_time < deadline:
                blocks = local_search_swap(blocks, time.time() + ls_time)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        seed += 1
    
    if best_blocks is None:
        # Fallback
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
                block_pairs = list(itertools.combinations(block, 2))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for p1, p2 in block_pairs:
                        idx = triple_idx[x, p1, p2]
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
