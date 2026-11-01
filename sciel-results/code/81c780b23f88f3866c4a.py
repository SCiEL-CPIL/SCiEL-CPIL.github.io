import itertools
import random
import time
import sys

def construct(v, k, t):
    # Precompute all t-subsets and their IDs
    t_subs = list(itertools.combinations(range(v), t))
    sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    num_subs = len(t_subs)
    
    def get_sub_ids(block):
        """Get IDs of all t-subsets contained in block."""
        return [sub_to_id[sub] for sub in itertools.combinations(block, t)]

    def greedy_cover(v, k, t, sub_to_id, t_subs, num_subs, seed):
        random.seed(seed)
        uncovered = set(range(num_subs))
        blocks = []
        
        while uncovered:
            start_sub_id = random.choice(list(uncovered))
            start_sub = t_subs[start_sub_id]
            block = list(start_sub)
            
            while len(block) < k:
                gains = []
                for x in range(v):
                    if x in block:
                        continue
                    remaining_points = [p for p in range(v) if p not in block and p != x]
                    need = t - len(block) - 1
                    count = 0
                    if need >= 0:
                        for rest in itertools.combinations(remaining_points, need):
                            s = tuple(sorted(tuple(block) + (x,) + rest))
                            sid = sub_to_id.get(s)
                            if sid is not None and sid in uncovered:
                                count += 1
                    gains.append((count, x))
                
                if not gains or max(g[0] for g in gains) == 0:
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        block.append(random.choice(candidates))
                    else:
                        break
                else:
                    gains.sort(reverse=True)
                    top_k = min(5, len(gains))
                    candidates = [g[1] for g in gains[:top_k]]
                    block.append(random.choice(candidates))
            
            block.sort()
            blocks.append(block)
            for sid in get_sub_ids(block):
                uncovered.discard(sid)
        
        return blocks

    # 1. Generate a pool of candidate blocks
    # We use multiple greedy runs with different seeds to create a diverse pool
    pool = []
    pool_set = set()
    
    # Add blocks from several greedy runs
    for s in range(10):
        b = greedy_cover(v, k, t, sub_to_id, t_subs, num_subs, seed=1000 + s)
        for blk in b:
            if tuple(blk) not in pool_set:
                pool_set.add(tuple(blk))
                pool.append(blk)
        
        # Stop if pool is large enough
        if len(pool) > 2000:
            break
            
    # If pool is still small, add random blocks? 
    # For v=19, k=8, random blocks are rare to be useful, but let's add some to ensure diversity
    if len(pool) < 3000:
        random.seed(42)
        while len(pool) < 3000:
            blk = tuple(sorted(random.sample(range(v), k)))
            if blk not in pool_set:
                pool_set.add(blk)
                pool.append(list(blk))

    # Precompute subset IDs for each block in the pool
    pool_sub_ids = [get_sub_ids(blk) for blk in pool]
    num_blocks = len(pool)
    
    # 2. Optimization: Randomized Local Search (Add-Remove)
    best_size = float('inf')
    best_selected = None
    
    start_time = time.time()
    timeout = 110.0
    
    # Initialize with the first greedy cover (from pool)
    # We need to identify which pool blocks form a valid cover.
    # The blocks from the first greedy run (seed 1000) are in the pool.
    # Let's just start with a random valid cover.
    # We can reconstruct a cover by greedily picking from the pool?
    # Or just use the result of greedy_cover directly as initial state.
    
    initial_blocks = greedy_cover(v, k, t, sub_to_id, t_subs, num_subs, seed=2000)
    
    # Map initial blocks to pool indices
    # Note: initial_blocks might not be in the pool if pool generation stopped early or seeds differ.
    # To be safe, let's ensure the initial cover is in the pool or just work with the pool indices directly.
    # It's easier to just run the optimization on the pool.
    # We need a valid starting set.
    
    # Let's find indices in pool that form a cover.
    # We can use a simple greedy selection from the pool to get a valid starting set.
    def greedy_select_from_pool(pool, pool_sub_ids, sub_to_id, t_subs, num_subs):
        uncovered = set(range(num_subs))
        selected_indices = []
        # Sort pool by coverage of uncovered? No, just iterate or random.
        # Let's just pick blocks that cover the most uncovered subsets.
        # Since pool is static, we can compute "potential" gain dynamically.
        
        # For efficiency, just iterate through pool and add if it covers any new subset.
        # This might not be optimal but ensures validity.
        for i in range(num_blocks):
            if not uncovered:
                break
            for sid in pool_sub_ids[i]:
                if sid in uncovered:
                    selected_indices.append(i)
                    for s in pool_sub_ids[i]:
                        uncovered.discard(s)
                    break
            # Optimization: if we add a block, we should check if it's the "best" add.
            # For initialization, a simple pass is fine.
        
        # If we didn't cover everything (unlikely with good pool), add more
        while uncovered:
             # Pick random block from pool that covers at least one uncovered
             found = False
             for i in range(num_blocks):
                 if i in selected_indices: continue
                 for sid in pool_sub_ids[i]:
                     if sid in uncovered:
                         selected_indices.append(i)
                         for s in pool_sub_ids[i]:
                             uncovered.discard(s)
                         found = True
                         break
                 if found: break
             if not found:
                 break # Should not happen
        
        return selected_indices

    selected_indices = greedy_select_from_pool(pool, pool_sub_ids, sub_to_id, t_subs, num_subs)
    selected_set = set(selected_indices)
    
    # Initialize cover counts
    cover_count = [0] * num_subs
    for i in selected_set:
        for sid in pool_sub_ids[i]:
            cover_count[sid] += 1
            
    current_size = len(selected_set)
    if current_size < best_size:
        best_size = current_size
        best_selected = list(selected_set)

    while time.time() - start_time < timeout:
        # Phase 1: Deletion
        deleted_any = False
        # Iterate through a copy of the selected set to avoid modification during iteration issues
        # We check each block if it's redundant
        for i in list(selected_set):
            redundant = True
            for sid in pool_sub_ids[i]:
                if cover_count[sid] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for sid in pool_sub_ids[i]:
                    cover_count[sid] -= 1
                selected_set.discard(i)
                deleted_any = True
                
        if deleted_any:
            current_size = len(selected_set)
            if current_size < best_size:
                best_size = current_size
                best_selected = list(selected_set)
            continue # Restart deletion loop as removals might make others redundant

        # Phase 2: Escape Local Minimum (Addition)
        # If no blocks were deleted, we are in a local minimum.
        # Add a random block from the pool that is not currently selected.
        # This might allow other blocks to become redundant.
        
        # Find indices not in selected_set
        # To avoid O(N) scan every time, we can maintain a list of available indices?
        # Or just pick a random index and check.
        # Pool size is ~3000, Selected is ~300.
        # Probability of picking an unselected block is high.
        
        attempts = 0
        added = False
        while attempts < 100 and not added:
            j = random.randint(0, num_blocks - 1)
            if j not in selected_set:
                # Add block j
                selected_set.add(j)
                for sid in pool_sub_ids[j]:
                    cover_count[sid] += 1
                added = True
            attempts += 1
        
        if not added:
            # All blocks are selected? Unlikely.
            break

    # Convert best_selected indices back to blocks
    best_blocks = [pool[i] for i in best_selected]
    return best_blocks
