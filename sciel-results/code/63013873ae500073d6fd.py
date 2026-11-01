"""Covering design C(24,7,3) via fast randomized greedy + redundancy removal + local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # 118s to be safe under 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute: for each point x, list of t-subsets containing x (as indices)
    # And for each (point x, (t-1)-tuple), the index of the t-subset
    # For t=3: for each point x and each pair (a,b), precompute idx of (a,b,x)
    
    # Precompute t-subset indices for fast gain computation
    # For t=3: precompute pair_to_idx[x][a][b] = index of t-subset {a,b,x}
    if t == 3:
        # pair_idx[x][a][b] where a < b, a != x, b != x
        # Use a 3D array: pair_idx[x, a, b] = index of t-subset {a,b,x}
        pair_idx = np.full((v, v, v), -1, dtype=np.int32)
        for a in range(v):
            for b in range(a+1, v):
                for x in range(v):
                    if x != a and x != b:
                        ts = tuple(sorted((a, b, x)))
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            pair_idx[x, a, b] = idx
                            pair_idx[x, b, a] = idx
    else:
        pair_idx = None
    
    best_blocks = None
    best_count = num_tsubsets  # worst case upper bound
    
    def greedy_cover_fast(seed, time_limit):
        """Run a single greedy covering attempt with given random seed, using fast gain computation."""
        rng = random.Random(seed)
        uncovered_arr = np.ones(num_tsubsets, dtype=np.uint8)  # 1 = uncovered
        uncovered_set = set(range(num_tsubsets))
        blocks = []
        
        while uncovered_set and time.time() < time_limit:
            # Pick a random uncovered t-subset to start the block
            start_ts = rng.choice(tuple(uncovered_set))
            start_combo = all_tsubsets[start_ts]
            block = list(start_combo)
            block_set = set(block)
            
            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                if t == 3 and pair_idx is not None:
                    # Fast path: for each candidate x, count uncovered t-subsets
                    # that would be covered by adding x to block
                    gains = np.zeros(v, dtype=np.int32)
                    for a in block:
                        for b in block:
                            if a == b:
                                continue
                            # For each x not in block, check pair_idx[x, a, b]
                            # We need x not in block
                            for x in range(v):
                                if x in block_set:
                                    continue
                                idx = pair_idx[x, a, b]
                                if idx >= 0 and uncovered_arr[idx]:
                                    gains[x] += 1
                    
                    # Find best x
                    valid_mask = np.ones(v, dtype=bool)
                    for x in block_set:
                        valid_mask[x] = False
                    gains[~valid_mask] = -1
                    
                    # Add some randomness for tie-breaking
                    # Find max gain
                    max_gain = gains.max()
                    if max_gain <= 0:
                        # Pick random valid point
                        valid_pts = np.where(valid_mask)[0]
                        best_x = rng.choice(valid_pts.tolist())
                    else:
                        # Randomly pick among points with max gain
                        max_pts = np.where(gains == max_gain)[0]
                        best_x = rng.choice(max_pts.tolist())
                else:
                    # General path (slower)
                    best_x = -1
                    best_gain = -1
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            new_ts = tuple(sorted(rest + (x,)))
                            idx = tsubset_to_idx.get(new_ts, -1)
                            if idx >= 0 and idx in uncovered_set:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                        elif gain == best_gain and best_x >= 0:
                            if rng.random() < 0.5:
                                best_x = x
                    
                    if best_x < 0:
                        break
                
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Remove covered t-subsets
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered_arr[idx] = 0
                    uncovered_set.discard(idx)
        
        if not uncovered_set:
            return [list(b) for b in blocks]
        return None
    
    def remove_redundant(blocks, time_limit):
        """Try to remove redundant blocks from a covering."""
        if time.time() > time_limit:
            return blocks
        
        # Count how many blocks cover each t-subset
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
        
        # Try to remove each block if it doesn't make any t-subset uncovered
        result = list(blocks)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(result)):
                if time.time() > time_limit:
                    break
                block = result[i]
                # Check if removing this block would leave any t-subset uncovered
                can_remove = True
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0 and cover_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove block i
                    for ts in itertools.combinations(block, t):
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] -= 1
                    result.pop(i)
                    changed = True
                    break  # Restart from beginning
        return [list(b) for b in result]
    
    def local_search_swap(blocks, time_limit):
        """Try to reduce block count by swapping out one block for another.
        For each block, try removing it and see if we can cover the newly-uncovered
        t-subsets with a single new block."""
        if time.time() > time_limit:
            return blocks
        
        # Compute cover counts
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
        
        result = list(blocks)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(result)):
                if time.time() > time_limit:
                    break
                block_i = result[i]
                # Temporarily remove block_i
                newly_uncovered = []
                for ts in itertools.combinations(block_i, t):
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx >= 0:
                        cover_count[idx] -= 1
                        if cover_count[idx] == 0:
                            newly_uncovered.append(ts)
                
                if not newly_uncovered:
                    # Can just remove
                    result.pop(i)
                    improved = True
                    break
                
                # Try to find a single block that covers all newly_uncovered t-subsets
                # The new block must contain all points from each newly_uncovered t-subset
                # So it must be a superset of the union... but that's too restrictive.
                # Instead: the new block must contain each t-subset in newly_uncovered.
                # For t=3, each newly_uncovered is a triple. The new block (size k=7)
                # must contain each triple.
                
                # Find points that are in all... no, the block must contain each triple.
                # So for each triple (a,b,c), the block must contain a, b, c.
                # Union of all points in newly_uncovered triples must be a subset of the new block.
                required_points = set()
                for ts in newly_uncovered:
                    for p in ts:
                        required_points.add(p)
                
                if len(required_points) > k:
                    # Can't cover with one block
                    for ts in newly_uncovered:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] += 1
                    continue
                
                # Try to complete required_points to a k-subset that maximizes coverage
                # of other uncovered t-subsets (those with cover_count == 1 that this block would help)
                # Actually, we just need to verify the covering is still valid.
                # Let's try a few random completions.
                remaining_needed = k - len(required_points)
                available = [x for x in range(v) if x not in required_points]
                
                found = False
                if remaining_needed == 0:
                    candidate = sorted(required_points)
                    # Verify: does this block cover all newly_uncovered?
                    ok = True
                    for ts in newly_uncovered:
                        if not all(p in candidate for p in ts):
                            ok = False
                            break
                    if ok:
                        result[i] = candidate
                        # Update cover counts for the new block
                        for ts in itertools.combinations(candidate, t):
                            idx = tsubset_to_idx.get(ts, -1)
                            if idx >= 0:
                                cover_count[idx] += 1
                        improved = True
                        found = True
                        break
                else:
                    # Try a few random completions
                    for attempt in range(20):
                        if time.time() > time_limit:
                            break
                        extra = random.sample(available, remaining_needed)
                        candidate = sorted(required_points | set(extra))
                        # Verify: does this block cover all newly_uncovered?
                        ok = True
                        for ts in newly_uncovered:
                            if not all(p in candidate for p in ts):
                                ok = False
                                break
                        if ok:
                            result[i] = candidate
                            # Update cover counts for the new block
                            for ts in itertools.combinations(candidate, t):
                                idx = tsubset_to_idx.get(ts, -1)
                                if idx >= 0:
                                    cover_count[idx] += 1
                            improved = True
                            found = True
                            break
                
                if not found:
                    # Restore cover counts
                    for ts in newly_uncovered:
                        idx = tsubset_to_idx.get(ts, -1)
                        if idx >= 0:
                            cover_count[idx] += 1
        
        return result
    
    def point_swap_local_search(blocks, time_limit):
        """Try to reduce block count by swapping points in blocks, then removing redundant blocks.
        This is inspired by the local search approach from inspiration 1."""
        if time.time() > time_limit:
            return blocks
        
        # Compute cover counts
        cover_count = np.zeros(num_tsubsets, dtype=np.int32)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    cover_count[idx] += 1
        
        result = list(blocks)
        rng = random.Random(12345)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try to delete blocks in random order
            indices = list(range(len(result)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > time_limit:
                    break
                block = result[idx]
                # Check if all triples in this block are covered by other blocks
                can_remove = True
                for ts in itertools.combinations(block, t):
                    ti = tsubset_to_idx.get(ts, -1)
                    if ti >= 0 and cover_count[ti] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Delete it
                    for ts in itertools.combinations(block, t):
                        ti = tsubset_to_idx.get(ts, -1)
                        if ti >= 0:
                            cover_count[ti] -= 1
                    result.pop(idx)
                    improved = True
                    break
            
            # If no deletion worked, try perturbations
            if not improved and len(result) > 0:
                # Try swapping a point in a random block
                for _ in range(100):
                    if time.time() > time_limit:
                        break
                    idx = rng.randint(0, len(result)-1)
                    old_x = rng.choice(result[idx])
                    new_x = rng.randint(0, v-1)
                    if new_x == old_x or new_x in result[idx]:
                        continue
                    
                    # Try the swap
                    other_points = [p for p in result[idx] if p != old_x]
                    new_block = sorted([new_x] + other_points)
                    
                    # Check if swap is valid
                    old_block_triples = set(itertools.combinations(result[idx], t))
                    new_block_triples = set(itertools.combinations(new_block, t))
                    
                    valid = True
                    for ts in old_block_triples:
                        ti = tsubset_to_idx.get(ts, -1)
                        if ti >= 0 and cover_count[ti] <= 1 and ts not in new_block_triples:
                            valid = False
                            break
                    
                    if valid:
                        # Apply swap
                        for ts in old_block_triples:
                            ti = tsubset_to_idx.get(ts, -1)
                            if ti >= 0:
                                cover_count[ti] -= 1
                        for ts in new_block_triples:
                            ti = tsubset_to_idx.get(ts, -1)
                            if ti >= 0:
                                cover_count[ti] += 1
                        result[idx] = new_block
                        
                        # After swap, try to delete some blocks
                        indices2 = list(range(len(result)))
                        rng.shuffle(indices2)
                        for idx2 in indices2:
                            block2 = result[idx2]
                            can_remove = True
                            for ts in itertools.combinations(block2, t):
                                ti = tsubset_to_idx.get(ts, -1)
                                if ti >= 0 and cover_count[ti] <= 1:
                                    can_remove = False
                                    break
                            if can_remove:
                                for ts in itertools.combinations(block2, t):
                                    ti = tsubset_to_idx.get(ts, -1)
                                    if ti >= 0:
                                        cover_count[ti] -= 1
                                result.pop(idx2)
                                improved = True
                                break
                        
                        if improved:
                            break
        
        return result
    
    # Try multiple randomized greedy attempts
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Allocate time for this attempt
        attempt_time = min(remaining - 2, 20)  # Leave 2s buffer
        
        blocks = greedy_cover_fast(seed, time.time() + attempt_time)
        if blocks is not None:
            # Try to remove redundant blocks
            blocks = remove_redundant(blocks, time.time() + min(remaining - 2, 8))
            
            # Try local search swap
            blocks = local_search_swap(blocks, time.time() + min(remaining - 2, 8))
            
            # Try point swap local search
            blocks = point_swap_local_search(blocks, time.time() + min(remaining - 2, 8))
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
            seed += 1
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(range(num_tsubsets))
        blocks = []
        while uncovered:
            start_ts = min(uncovered)
            start_combo = all_tsubsets[start_ts]
            block = list(start_combo)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(rest + (x,)))
                        idx = tsubset_to_idx.get(new_ts, -1)
                        if idx >= 0 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered.discard(idx)
        return blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
