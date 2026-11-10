import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using greedy initialization 
    and aggressive local search (redundancy removal and block swaps).
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Map each point to the list of t-subset indices it belongs to
    # This helps in quick lookups if needed, but for now we iterate block t-subsets
    
    def is_valid(blocks):
        """Check if blocks cover all t-subsets."""
        if not blocks:
            return n_tsub == 0
        covered = bytearray([0] * n_tsub)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(tuple(sorted(ts)))
                if idx is not None:
                    covered[idx] = 1
        # Check if all are 1
        for i in range(n_tsub):
            if not covered[i]:
                return False
        return True

    def greedy_build(seed):
        """Fast greedy covering."""
        rng = random.Random(seed)
        uncovered = bytearray([1] * n_tsub)
        uncovered_count = n_tsub
        blocks = []
        
        # Precompute t-subsets for each potential block extension?
        # Simpler: just greedy add points
        
        while uncovered_count > 0:
            if time.time() > deadline - 20:
                break
                
            # Find an uncovered t-subset
            # Random sampling is faster than scanning if sparse
            found_idx = -1
            for _ in range(100):
                idx = rng.randint(0, n_tsub - 1)
                if uncovered[idx]:
                    found_idx = idx
                    break
            
            if found_idx == -1:
                # Fallback scan
                for idx in range(n_tsub):
                    if uncovered[idx]:
                        found_idx = idx
                        break
                if found_idx == -1:
                    break

            seed_ts = all_tsubsets[found_idx]
            block = list(seed_ts)
            block_set = set(block)
            rng.shuffle(block)
            
            # Greedily add points
            while len(block) < k:
                if time.time() > deadline - 20:
                    break
                best_gain = 0
                best_x = None
                # Current block size is s, we add 1 point.
                # New t-subsets formed are those containing the new point and t-1 from existing block.
                # We want to maximize uncovered t-subsets covered.
                
                # Optimization: Precompute (t-1) subsets of current block
                block_t1 = list(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_t1:
                        # Form t-subset: rest + x
                        # Need to check if it's a valid t-subset (sorted tuple)
                        # And if it's uncovered
                        # Constructing tuple every time is slow.
                        # Let's precompute a lookup for gain calculation?
                        # Or just do it simply first.
                        new_ts = list(rest)
                        new_ts.append(x)
                        new_ts.sort()
                        idx = tsubset_to_idx.get(tuple(new_ts))
                        if idx is not None and uncovered[idx]:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x is None:
                    # No gain, pick random
                    remaining = [x for x in range(v) if x not in block_set]
                    if not remaining:
                        break
                    best_x = rng.choice(remaining)
                    
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(tuple(sorted(ts)))
                if idx is not None and uncovered[idx]:
                    uncovered[idx] = 0
                    uncovered_count -= 1

        return blocks

    def local_search(blocks, max_time=30):
        """
        Aggressive local search:
        1. Remove redundant blocks.
        2. Try 2-for-1 swaps.
        3. Try 1-for-1 replacements.
        """
        start_ls = time.time()
        blocks = [list(b) for b in blocks] # Deep copy
        
        def get_cover_counts(blocks):
            counts = [0] * n_tsub
            block_tsubs = []
            for b in blocks:
                bts = []
                for ts in itertools.combinations(b, t):
                    idx = tsubset_to_idx.get(tuple(sorted(ts)))
                    if idx is not None:
                        bts.append(idx)
                        counts[idx] += 1
                block_tsubs.append(bts)
            return counts, block_tsubs

        def try_construct_block(uniq_tsubs, rng):
            """
            Try to find an 8-subset that covers all t-subsets in uniq_tsubs.
            Returns the block as a list, or None.
            """
            if not uniq_tsubs:
                return list(rng.sample(range(v), k))
            
            # Union of all points in uniq_tsubs
            must_have = set()
            for ts in uniq_tsubs:
                must_have.update(ts)
                
            if len(must_have) > k:
                return None
                
            # Start with must_have
            block = list(must_have)
            block_set = set(block)
            
            # Check if current block covers all uniq_tsubs
            def covers_all(block_set):
                for ts in uniq_tsubs:
                    if not all(x in block_set for x in ts):
                        return False
                return True
                
            if covers_all(block_set):
                # Fill rest randomly
                remaining = [x for x in range(v) if x not in block_set]
                rng.shuffle(remaining)
                block.extend(remaining[:k-len(block)])
                return sorted(block)
            
            # Greedily add points to cover remaining t-subsets
            while len(block) < k:
                # Find uncovered t-subsets in uniq_tsubs
                uncovered_ts = []
                for ts in uniq_tsubs:
                    if not all(x in block_set for x in ts):
                        uncovered_ts.append(ts)
                        
                if not uncovered_ts:
                    break
                    
                best_gain = 0
                best_x = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for ts in uncovered_ts:
                        if x in ts:
                            # Check if adding x covers this ts?
                            # It covers ts if all other points in ts are in block_set
                            if all(y in block_set for y in ts if y != x):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x is None:
                    # Try random
                    remaining = [x for x in range(v) if x not in block_set]
                    if not remaining:
                        return None
                    best_x = rng.choice(remaining)
                    
                block.append(best_x)
                block_set.add(best_x)
                
            if covers_all(block_set):
                return sorted(block)
            return None

        # Main loop
        changed = True
        while changed and (time.time() - start_ls) < max_time:
            changed = False
            counts, block_tsubs = get_cover_counts(blocks)
            
            # 1. Redundancy Removal
            new_blocks = []
            new_block_tsubs = []
            for i, b in enumerate(blocks):
                is_critical = False
                for idx in block_tsubs[i]:
                    if counts[idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    # Remove
                    for idx in block_tsubs[i]:
                        counts[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
                    new_block_tsubs.append(block_tsubs[i])
                    
            blocks = new_blocks
            block_tsubs = new_block_tsubs
            
            # 2. 2-for-1 Swap
            if time.time() > start_ls + max_time - 5:
                break
                
            n_blocks = len(blocks)
            rng = random.Random(int(time.time()*1000) % 10000)
            
            # Iterate over pairs
            swapped = False
            for i in range(n_blocks):
                for j in range(i+1, n_blocks):
                    # Calculate unique t-subsets for i and j
                    # A t-subset is unique to i if count==1 and it's in block i
                    # But we need the union of unique i and unique j
                    # Actually, we want to replace i and j with a new block C
                    # such that C covers all t-subsets that are NOT covered by any other block.
                    
                    # Let U be the set of t-subset indices such that:
                    # count[idx] == 1 and (idx in block_tsubs[i] or idx in block_tsubs[j])
                    # Wait, if count[idx] == 1, it is covered by exactly one block.
                    # If we remove i and j, we lose coverage for all idx in block_tsubs[i] and block_tsubs[j].
                    # We need the new block C to cover all idx that were covered ONLY by i or ONLY by j.
                    
                    unique_to_i = set(idx for idx in block_tsubs[i] if counts[idx] == 1)
                    unique_to_j = set(idx for idx in block_tsubs[j] if counts[idx] == 1)
                    
                    # The new block must cover unique_to_i union unique_to_j
                    # Note: If a t-subset is covered by both i and j, its count is at least 2.
                    # So unique_to_i and unique_to_j are disjoint.
                    
                    target_tsubs = list(unique_to_i | unique_to_j)
                    
                    if not target_tsubs:
                        # We can just remove one of them? No, we are doing 2-for-1.
                        # If target is empty, we can replace i and j with a random block?
                        # Or just remove i? That's 1-for-0.
                        # Let's just skip if empty, or try to remove one.
                        # Actually, if unique sets are empty, we can remove i entirely (1-for-0).
                        # But redundancy pass should have caught that.
                        # Unless counts changed? No, counts are consistent.
                        continue
                    
                    # Try to construct C
                    # To speed up, limit the number of target t-subsets
                    if len(target_tsubs) > 10: 
                        continue
                        
                    new_block = try_construct_block(target_tsubs, rng)
                    if new_block is not None:
                        # Check if new_block is already in blocks
                        # And check if it's valid (it is by construction)
                        # We need to ensure new_block doesn't duplicate existing blocks
                        # For simplicity, we'll allow duplicates for a moment and clean up later?
                        # No, blocks must be distinct.
                        
                        # Check distinctness
                        new_block_set = frozenset(new_block)
                        exists = False
                        for b in blocks:
                            if frozenset(b) == new_block_set:
                                exists = True
                                break
                                # Also check if it's i or j
                                
                        if exists and new_block_set not in (frozenset(blocks[i]), frozenset(blocks[j])):
                            continue
                            
                        # Replace i and j with new_block
                        # Remove i and j
                        # We need to update counts
                        # This is expensive if done inside the loop.
                        # Let's just mark that a swap happened and restart the outer loop?
                        # Or do it carefully.
                        
                        # Let's do it carefully.
                        # Decrement counts for i and j
                        for idx in block_tsubs[i]:
                            counts[idx] -= 1
                        for idx in block_tsubs[j]:
                            counts[idx] -= 1
                            
                        # Increment counts for new_block
                        new_bts = []
                        for ts in itertools.combinations(new_block, t):
                            idx = tsubset_to_idx.get(tuple(sorted(ts)))
                            if idx is not None:
                                new_bts.append(idx)
                                counts[idx] += 1
                                
                        # Update blocks list
                        # Remove i and j (indices shift)
                        # Easier: build new list
                        new_blocks_list = []
                        new_tsubs_list = []
                        for m, b in enumerate(blocks):
                            if m == i or m == j:
                                continue
                            new_blocks_list.append(b)
                            new_tsubs_list.append(block_tsubs[m])
                            
                        new_blocks_list.append(new_block)
                        new_tsubs_list.append(new_bts)
                        
                        blocks = new_blocks_list
                        block_tsubs = new_tsubs_list
                        changed = True
                        swapped = True
                        break # Restart pair loop with new state?
                            # Actually, we should restart the whole while loop to be safe with indices
                            # But breaking inner loops and letting outer while handle it is fine.
                            # We need to break out of pair loop and go to next while iteration.
                if swapped:
                    break
            
            # 3. 1-for-1 Replacement
            if not swapped and (time.time() - start_ls) < max_time - 5:
                # Similar logic for single block replacement
                # Find block i with unique t-subsets U_i
                # Try to find C covering U_i
                # If C != i, replace i with C
                replaced = False
                for i in range(len(blocks)):
                    unique_i = set(idx for idx in block_tsubs[i] if counts[idx] == 1)
                    if not unique_i:
                        continue
                    if len(unique_i) > 10:
                        continue
                        
                    new_block = try_construct_block(list(unique_i), rng)
                    if new_block is not None:
                        new_block_set = frozenset(new_block)
                        if new_block_set == frozenset(blocks[i]):
                            continue
                            
                        # Check distinctness
                        exists = False
                        for m, b in enumerate(blocks):
                            if m != i and frozenset(b) == new_block_set:
                                exists = True
                                break
                        if exists:
                            continue
                        
                        # Replace
                        for idx in block_tsubs[i]:
                            counts[idx] -= 1
                            
                        new_bts = []
                        for ts in itertools.combinations(new_block, t):
                            idx = tsubset_to_idx.get(tuple(sorted(ts)))
                            if idx is not None:
                                new_bts.append(idx)
                                counts[idx] += 1
                                
                        blocks[i] = new_block
                        block_tsubs[i] = new_bts
                        changed = True
                        replaced = True
                        break
                if replaced:
                    pass # Continue while loop

        return blocks

    # Main search
    best_result = None
    best_len = float('inf')
    
    # Phase 1: Greedy + Local Search
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        result = greedy_build(seed)
        if result:
            result = local_search(result, max_time=10)
            if is_valid(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result
    
    # Phase 2: More seeds if time permits
    for seed in range(20, 50):
        if time.time() > deadline - 10:
            break
        result = greedy_build(seed)
        if result:
            result = local_search(result, max_time=5)
            if is_valid(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result

    if best_result is None:
        # Fallback
        best_result = greedy_build(0)
        
    return best_result

