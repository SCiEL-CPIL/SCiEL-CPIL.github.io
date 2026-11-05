import itertools
import time
import random
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    # Precompute all t-subsets
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_to_id = {ts: i for i, ts in enumerate(all_ts)}

    # Precompute for each point, the list of t-subset IDs that contain it
    # This helps in gain calculation
    point_ts = [[] for _ in range(v)]
    for ts in all_ts:
        for p in ts:
            point_ts[p].append(ts_to_id[ts])

    # For a given block (list of k points), compute the set of t-subset IDs it covers
    def block_cover_ids(block):
        """Return a set of t-subset IDs covered by this block."""
        ids = set()
        for ts in itertools.combinations(block, t):
            ids.add(ts_to_id[ts])
        return ids

    # For a given block, return a frozenset of its points (for dedup)
    def block_key(block):
        return frozenset(block)

    def greedy_construct(v, k, t, seed=0, bias=0.0, time_limit=None):
        """
        Bitset-optimized greedy construction.
        """
        rng = random.Random(seed)
        
        # Use a numpy array for coverage counts (0 = uncovered, >0 = covered)
        # But for greedy, we just need to know if uncovered
        covered = np.zeros(n_ts, dtype=np.int8)
        
        blocks = []
        seen_blocks = set()
        
        while True:
            # Check if all covered
            if np.all(covered > 0):
                break
            
            # Pick a starting triplet: prefer uncovered ones
            uncovered_mask = covered == 0
            if not np.any(uncovered_mask):
                break
            
            # Randomly pick an uncovered triplet to start
            uncovered_ids = np.where(uncovered_mask)[0]
            start_id = uncovered_ids[rng.randint(0, len(uncovered_ids) - 1)]
            start_ts = all_ts[start_id]
            
            block = list(start_ts)
            block_set = set(block)
            
            # Mark the starting triplet as "tentatively covered" for gain calc?
            # No, we compute gain based on currently uncovered.
            
            while len(block) < k:
                # Calculate gain for each candidate point
                # gain(p) = number of currently uncovered t-subsets that would be covered
                #          by adding p to the current block.
                # A t-subset is covered by block+p iff it is a subset of block+p.
                # Since |block| = j < k, and t=3:
                # The new t-subsets are those of the form (a, b, p) where {a,b} ⊆ block.
                
                best_point = -1
                max_gain = -1
                second_best_gain = -1
                second_best_point = -1
                
                # Pairs in current block
                if t == 3:
                    pairs = list(itertools.combinations(block, 2))
                else:
                    pairs = list(itertools.combinations(block, t - 1))
                
                # For each candidate point p not in block
                # We need to count how many (pair, p) combos are uncovered.
                
                # Optimization: Precompute for each pair, the list of t-subset IDs
                # Actually, for each pair (a,b) and point p, the t-subset is {a,b,p}.
                # We can precompute pair_point_to_id: {(a,b,p): id}
                # But that's C(v,3) entries which is 2024 for v=24. Small enough.
                
                # Let's just iterate candidates and check coverage.
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs:
                        ts = tuple(sorted(pair + (p,)))
                        tid = ts_to_id[ts]
                        if covered[tid] == 0:
                            gain += 1
                    if gain > max_gain:
                        second_best_gain = max_gain
                        second_best_point = best_point
                        max_gain = gain
                        best_point = p
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = p
                
                if max_gain == 0:
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    r = rng.random()
                    if r < (1.0 - bias):
                        pass
                    elif second_best_point != -1 and r < (1.0 - bias) + bias * 0.5:
                        best_point = second_best_point
                    else:
                        candidates = [p for p in range(v) if p not in block_set]
                        best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            bkey = block_key(block)
            if bkey in seen_blocks:
                # Duplicate block, skip adding but still mark coverage?
                # No, if it's a duplicate, it adds no new coverage.
                # But we already started with an uncovered triplet, so this shouldn't happen
                # unless the block was already added. Let's just skip.
                # Actually, we should still mark coverage in case the block wasn't added
                # but covers new triplets? No, if it's a duplicate, it covers the same triplets.
                pass
            
            blocks.append(block)
            seen_blocks.add(bkey)
            
            # Update coverage
            for ts in itertools.combinations(block, t):
                tid = ts_to_id[ts]
                covered[tid] = 1  # Just mark as covered (binary is enough for greedy)
            
            if time_limit and time.time() > time_limit:
                break
        
        return blocks

    # Phase 1: Multiple greedy runs with different seeds and biases
    best_blocks = None
    best_count = float('inf')
    
    # Pure greedy
    for seed in range(30):
        if time.time() > deadline - 50:
            break
        blocks = greedy_construct(v, k, t, seed, bias=0.0)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Randomized greedy
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        bias = 0.1 + 0.03 * seed
        blocks = greedy_construct(v, k, t, seed, bias=bias)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        best_blocks = greedy_construct(v, k, t, 0, bias=0.0)

    # Phase 2: Local search with bitset coverage
    # Use a count array for proper redundancy detection
    count = np.zeros(n_ts, dtype=np.int16)
    
    def update_count(blocks_list, delta):
        for b in blocks_list:
            for ts in itertools.combinations(b, t):
                tid = ts_to_id[ts]
                count[tid] += delta
    
    def is_redundant(block):
        for ts in itertools.combinations(block, t):
            if count[ts_to_id[ts]] == 1:
                return False
        return True
    
    def remove_redundant(blocks_list):
        new_blocks = []
        removed = False
        for b in blocks_list:
            if is_redundant(b):
                update_count([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed
    
    current_blocks = list(best_blocks)
    update_count(current_blocks, 1)
    
    # Local search loop
    progress = True
    while progress and time.time() < deadline - 3:
        progress = False
        
        # 1. Remove redundants
        current_blocks, removed = remove_redundant(current_blocks)
        if removed:
            progress = True
        
        # 2. Try 2-for-1 swap
        if not progress:
            n = len(current_blocks)
            swapped = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                b1 = current_blocks[i]
                ids1 = block_cover_ids(b1)
                
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break
                    b2 = current_blocks[j]
                    ids2 = block_cover_ids(b2)
                    
                    # Find critical t-subsets (those covered only by b1 or b2)
                    critical_points = set()
                    for tid in ids1:
                        if count[tid] == 1:
                            critical_points.update(all_ts[tid])
                        elif tid in ids2 and count[tid] == 2:
                            critical_points.update(all_ts[tid])
                    
                    for tid in ids2:
                        if tid not in ids1 and count[tid] == 1:
                            critical_points.update(all_ts[tid])
                    
                    if not critical_points:
                        # Both redundant together
                        update_count([b1, b2], -1)
                        current_blocks = [b for idx, b in enumerate(current_blocks) if idx != i and idx != j]
                        swapped = True
                        break
                    elif len(critical_points) <= k:
                        new_block = list(critical_points)
                        rng = random.Random()
                        while len(new_block) < k:
                            p = rng.randint(0, v - 1)
                            if p not in new_block:
                                new_block.append(p)
                        new_block.sort()
                        
                        # Verify validity: all previously covered t-subsets remain covered
                        # Critical ones are in new_block by construction.
                        # Others had count >= 2, so removing b1,b2 leaves count >= 1.
                        # Adding new_block only increases counts.
                        # So it's valid.
                        
                        update_count([b1, b2], -1)
                        update_count([new_block], 1)
                        current_blocks = [b for idx, b in enumerate(current_blocks) if idx != i and idx != j]
                        current_blocks.append(new_block)
                        swapped = True
                        break
                if swapped:
                    break
            
            if swapped:
                progress = True
    
    # Final cleanup
    current_blocks, _ = remove_redundant(current_blocks)
    
    return current_blocks
