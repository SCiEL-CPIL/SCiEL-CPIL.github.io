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
            ts_list = [tsubset_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_tsubs.append(ts_list)
            for idx in ts_list:
                cover_count[idx] += 1

        result_blocks = [list(b) for b in blocks]
        result_tsubs = list(block_tsubs)

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            for i in range(len(result_blocks)):
                if time.time() > time_limit:
                    break

                # Check if block i is redundant (all its tsubsets are covered by others)
                redundant = True
                for idx in result_tsubs[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break

                if redundant:
                    # Remove block i
                    for idx in result_tsubs[i]:
                        cover_count[idx] -= 1

                    result_blocks.pop(i)
                    result_tsubs.pop(i)
                    improved = True
                    break

        return result_blocks

    def local_search_swap(blocks, time_limit):
        """Try to improve block count by swapping points in blocks."""
        if not blocks:
            return []

        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        block_tsubs = []

        for b in blocks:
            ts_list = [tsubset_to_idx[ts] for ts in itertools.combinations(b, t)]
            block_tsubs.append(ts_list)
            for idx in ts_list:
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

                for old_point in block:
                    # Temporarily remove old_point
                    old_indices = []
                    for ts in itertools.combinations(block, t):
                        idx = tsubset_to_idx[ts]
                        old_indices.append(idx)
                        cover_count[idx] -= 1

                    # Find best replacement point
                    best_new_point = -1
                    best_gain = 0
                    other_points = [p for p in block if p != old_point]

                    for new_point in range(v):
                        if new_point in block_set:
                            continue

                        # Gain: new tsubsets containing new_point that were uncovered
                        gain = 0
                        for ts in itertools.combinations(other_points, t - 1):
                            idx = triple_idx[new_point, ts[0], ts[1]]
                            if idx >= 0 and cover_count[idx] == 0:
                                gain += 1

                        if gain > best_gain:
                            best_gain = gain
                            best_new_point = new_point

                    # Restore cover_count
                    for idx in old_indices:
                        cover_count[idx] += 1

                    # If we found a better point, make the swap
                    if best_new_point != -1 and best_gain > 0:
                        # Remove old_point from block
                        block.remove(old_point)
                        block.append(best_new_point)
                        block.sort()
                        result_blocks[i] = block

                        # Update tsubs and cover_count
                        new_ts_list = [tsubset_to_idx[ts] for ts in itertools.combinations(block, t)]
                        result_tsubs[i] = new_ts_list
                        for idx in new_ts_list:
                            cover_count[idx] += 1

                        improved = True
                        break  # Restart loop after modification

        return result_blocks

    # Precompute gain table: gain_table[x][p1][p2] = number of uncovered triples (x,p1,p2)
    # Actually, we need a faster greedy. Let's precompute for each point x,
    # which triples it participates in, so we can quickly compute gain.
    
    # For each point x, list of (p1, p2) pairs such that (x,p1,p2) is a valid triple
    # and the triple index. We'll use this for fast gain computation.
    point_triples = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        a, b, c = ts
        point_triples[a].append((b, c, i))
        point_triples[b].append((a, c, i))
        point_triples[c].append((a, b, i))
    
    def fast_greedy_cover(seed, time_limit):
        """Faster greedy using precomputed point-triple lists."""
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
            block_set = set(block)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_pairs = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # For each pair in block, check if (x, p1, p2) is uncovered
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
                block_set.add(best_x)
            
            if len(block) < k:
                continue
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                uncovered[idx] = False
        
        if not np.any(uncovered):
            return [list(b) for b in blocks]
        return None
    
    best_blocks = None
    best_count = float('inf')
    
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        attempt_time = min(remaining - 2, 10)
        
        blocks = fast_greedy_cover(seed, time.time() + attempt_time)
        
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
        # Fallback: deterministic greedy
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
                block_pairs = list(itertools.combinations(block, t - 1))
                
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
                block_set.add(best_x)
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                uncovered[idx] = False
        
        return blocks

    return best_blocks
