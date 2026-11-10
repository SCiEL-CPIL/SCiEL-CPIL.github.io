# EVOLVE-BLOCK-START
import itertools
import time
import random
from math import comb
from collections import defaultdict

def _greedy_cover(v, k, t, rng, deadline):
    """
    Greedy set cover with hardness-based seed selection.
    Selects the uncovered t-subset that shares the fewest points with other uncovered t-subsets.
    """
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    uncovered = [True] * n_t
    n_uncovered = n_t
    
    # Precompute point to t-subset indices for hardness calculation
    point_to_ts = defaultdict(list)
    for i, ts in enumerate(all_t):
        for p in ts:
            point_to_ts[p].append(i)
            
    blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            break
            
        # 1. Select seed t-subset
        # We want to find an uncovered t-subset with minimal "interaction" with other uncovered ones.
        # Interaction = number of other uncovered t-subsets sharing at least one point.
        best_seed = None
        best_score = float('inf')
        
        # To keep it fast, we sample candidates if the list is large, but here n_t is ~2024, so we can iterate or sample.
        # Let's sample up to 100 candidates from the uncovered set to find a good seed.
        # If n_uncovered is small, check all.
        candidates = []
        if n_uncovered < 200:
            candidates = [i for i in range(n_t) if uncovered[i]]
        else:
            # Random sampling is faster and usually sufficient
            # We need to sample from indices where uncovered is True
            # Efficient way: just pick random indices and filter
            attempts = 0
            while len(candidates) < 100 and attempts < 500:
                idx = rng.randint(0, n_t - 1)
                if uncovered[idx] and idx not in candidates:
                    candidates.append(idx)
                attempts += 1
            if not candidates:
                # Fallback: find any uncovered
                for i in range(n_t):
                    if uncovered[i]:
                        candidates.append(i)
                        break

        for i in candidates:
            ts = all_t[i]
            # Count unique uncovered t-subsets sharing a point with ts
            neighbors = set()
            for p in ts:
                for j in point_to_ts[p]:
                    if j != i and uncovered[j]:
                        neighbors.add(j)
            
            score = len(neighbors)
            if score < best_score:
                best_score = score
                best_seed = i
            elif score == best_score and rng.random() < 0.2:
                best_seed = i
                
        if best_seed is None:
            # Should not happen if n_uncovered > 0
            break
            
        block = list(all_t[best_seed])
        
        # 2. Grow block to size k
        while len(block) < k:
            if time.time() > deadline:
                break
            in_block = set(block)
            best_x = None
            best_gain = -1
            best_tie_count = 0
            best_tie_list = []
            
            # Precompute (t-1) subsets of current block for gain calculation
            rests = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in in_block:
                    continue
                gain = 0
                for rest in rests:
                    # Create t-subset: rest + x
                    # rest is tuple of t-1, x is int
                    # We need to check if this specific t-subset is uncovered
                    # Optimization: sort the tuple to match t_index keys
                    ts_new = tuple(sorted(rest + (x,)))
                    idx = t_index.get(ts_new)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_tie_count = 1
                    best_tie_list = [x]
                elif gain == best_gain and gain > 0:
                    best_tie_count += 1
                    best_tie_list.append(x)
            
            if best_x is None or best_gain <= 0:
                # No gain found, pick a random unused point to fill the block
                unused = [x for x in range(v) if x not in in_block]
                if not unused:
                    break
                best_x = rng.choice(unused)
            
            block.append(best_x)
            
        block.sort()
        blocks.append(block)
        
        # 3. Mark covered t-subsets
        for ts in itertools.combinations(block, t):
            idx = t_index[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                n_uncovered -= 1
                
    return blocks

def _is_valid(blocks, v, k, t):
    total = comb(v, t)
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    return len(covered) == total

def _local_search(blocks, v, k, t, rng, deadline):
    """
    Local search to reduce block count.
    1. Remove redundant blocks.
    2. Replace a block with another that covers its unique t-subsets and maximizes other coverage.
    """
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)
    blocks = [sorted(list(b)) for b in blocks]
    
    # Coverage count per t-subset
    cov = [0] * n_t
    for b in blocks:
        for ts in itertools.combinations(b, t):
            cov[t_index[ts]] += 1
            
    def recompute_cov():
        nonlocal cov
        cov = [0] * n_t
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov[t_index[ts]] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        order = list(range(n))
        rng.shuffle(order)
        
        for bi in order:
            if time.time() > deadline:
                return blocks
            if bi >= len(blocks):
                continue
                
            b = blocks[bi]
            # Identify t-subsets uniquely covered by b
            unique_idxs = []
            for ts in itertools.combinations(b, t):
                idx = t_index[ts]
                if cov[idx] == 1:
                    unique_idxs.append(idx)
                    
            if not unique_idxs:
                # Block is fully redundant
                blocks.pop(bi)
                recompute_cov()
                improved = True
                break
                
            # Candidate points: union of points in unique t-subsets
            cand_set = set()
            for idx in unique_idxs:
                cand_set.update(all_t[idx])
            cand_list = list(cand_set)
            
            if len(cand_list) < k:
                continue
                
            # Try to construct a new block that covers all unique_idxs
            # and ideally covers other t-subsets as well
            newblock = []
            remaining_unique = set(unique_idxs)
            
            # Greedily pick points from cand_list to cover remaining_unique
            # We must pick points that help cover the unique sets.
            # If a point doesn't help cover any remaining unique set, it's only useful for general gain,
            # but we MUST cover all unique sets first.
            
            for _ in range(k):
                if not remaining_unique:
                    break
                    
                best_x = None
                best_count = -1
                
                # Prioritize points in cand_list that cover the most remaining unique t-subsets
                for x in cand_list:
                    if x in newblock:
                        continue
                    count = 0
                    for idx in remaining_unique:
                        if x in all_t[idx]:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_x = x
                        
                if best_x is None or best_count == 0:
                    # No point in cand_list covers a remaining unique set?
                    # This implies we can't cover all unique sets with points from cand_list?
                    # Wait, cand_list is the union of points in unique sets.
                    # If we picked points carefully, we should be able to.
                    # If we are stuck, we might need to allow points outside cand_list?
                    # No, the new block MUST contain all points of the unique t-subsets?
                    # No, the new block must COVER the unique t-subsets.
                    # A t-subset is covered if ALL its points are in the block.
                    # So the new block MUST contain the union of all unique t-subsets?
                    # Yes. If a unique t-subset is {a,b,c}, the block must contain a,b,c.
                    # So the new block MUST be a superset of the union of unique t-subsets.
                    # If len(union) > k, we cannot replace b with a single block.
                    # Actually, if len(cand_set) > k, we can't form a block containing all of them.
                    # So we can only do this replacement if len(cand_set) <= k.
                    if len(cand_set) > k:
                        continue
                    
                if best_x is None:
                    break
                    
                newblock.append(best_x)
                
                # Remove covered unique t-subsets
                to_remove = set()
                for idx in remaining_unique:
                    if all(p in newblock for p in all_t[idx]):
                        to_remove.add(idx)
                remaining_unique -= to_remove
                
            if remaining_unique:
                # Could not cover all unique t-subsets with k points
                continue
                
            # We have a valid newblock covering all unique t-subsets.
            # Now fill remaining spots (if any) to maximize total coverage gain
            # compared to the original block b.
            
            # Current coverage of newblock
            # We want to maximize the number of t-subsets covered by newblock
            # that are NOT already covered by other blocks (i.e., cov[idx] == 1 after removing b? No, cov is current state including b).
            # Let's calculate the "net gain" or just "total unique contribution".
            # Actually, simpler: just maximize the number of t-subsets in newblock that have cov[idx] == 1 (unique to b) 
            # PLUS t-subsets that have cov[idx] > 1 but might become unique if we remove b? No.
            # The goal is to keep the covering valid.
            # If we replace b with newblock:
            # - Any t-subset covered ONLY by b must be covered by newblock. (Handled by unique_idxs constraint)
            # - Any other t-subset covered by b is still covered by other blocks, so it doesn't matter if newblock covers it or not.
            # - However, if newblock covers a t-subset that was NOT covered by b, it might help future reductions.
            # - Also, if newblock covers a t-subset that WAS covered by b and others, it maintains coverage.
            
            # So, we just need to fill the remaining k - len(newblock) slots with points that maximize 
            # the coverage of t-subsets that are currently covered by b but might be at risk?
            # No, the only risk is the unique ones.
            # To be safe and potentially allow further reductions, we should pick points that cover the most 
            # t-subsets that are currently covered by EXACTLY one other block? No.
            # Let's just pick points that maximize the number of t-subsets in newblock that are 
            # "hard" (low coverage count) or just random.
            # Actually, a good heuristic is to maximize the number of t-subsets covered by newblock
            # that are covered by b. This minimizes the "change" in coverage profile.
            
            # Let's just fill with random points from v that are not in newblock, 
            # prioritizing those that complete t-subsets already in b.
            
            while len(newblock) < k:
                in_block = set(newblock)
                best_x = None
                best_gain = -1
                
                # Precompute rests of newblock
                rests = list(itertools.combinations(newblock, t - 1))
                
                for x in range(v):
                    if x in in_block:
                        continue
                    gain = 0
                    for rest in rests:
                        ts_new = tuple(sorted(rest + (x,)))
                        idx = t_index.get(ts_new)
                        if idx is not None:
                            # Prefer t-subsets that were covered by b
                            # We can check if ts_new is in the set of t-subsets of b
                            # For speed, just count any covered t-subset
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x is None:
                    break
                newblock.append(best_x)
                
            newblock.sort()
            
            # Check if newblock is different from b and not duplicate
            if newblock != b and newblock not in blocks:
                blocks.pop(bi)
                blocks.append(newblock)
                recompute_cov()
                improved = True
                break
                
    return blocks

def construct(v, k, t):
    start = time.time()
    deadline = start + 112.0  # Safety margin
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple iterations with different random seeds
    # The first one is the most important, so we give it more time or just run it first.
    
    # Iteration 1: Baseline
    rng = random.Random(12345)
    b = _greedy_cover(v, k, t, rng, deadline)
    if _is_valid(b, v, k, t):
        best_blocks = b
        best_len = len(b)
        
    if best_blocks is not None and time.time() < deadline - 10:
        ls = _local_search(best_blocks, v, k, t, rng, deadline - 10)
        if _is_valid(ls, v, k, t) and len(ls) < best_len:
            best_blocks = ls
            best_len = len(ls)
            
    # Iteration 2+: Randomized restarts
    attempt = 2
    while time.time() < deadline - 15:
        rng = random.Random(1000 + attempt)
        b = _greedy_cover(v, k, t, rng, deadline - 15)
        if not _is_valid(b, v, k, t):
            attempt += 1
            continue
            
        if len(b) < best_len:
            best_len = len(b)
            best_blocks = b
            
        if time.time() < deadline - 5:
            ls = _local_search(b, v, k, t, rng, deadline - 5)
            if _is_valid(ls, v, k, t) and len(ls) < best_len:
                best_len = len(ls)
                best_blocks = ls
                
        attempt += 1
        
    if best_blocks is None:
        # Fallback: should not happen
        return []
        
    return best_blocks
# EVOLVE-BLOCK-END
