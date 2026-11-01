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

    # 1. Generate a diverse pool of candidate blocks
    pool = []
    pool_set = set()
    
    # Add blocks from several greedy runs to ensure good coverage
    for s in range(15):
        b = greedy_cover(v, k, t, sub_to_id, t_subs, num_subs, seed=1000 + s)
        for blk in b:
            if tuple(blk) not in pool_set:
                pool_set.add(tuple(blk))
                pool.append(list(blk))
        
        if len(pool) > 5000:
            break
            
    # Fill with random blocks to increase diversity for local search
    random.seed(42)
    target_pool_size = 8000
    while len(pool) < target_pool_size:
        blk = tuple(sorted(random.sample(range(v), k)))
        if blk not in pool_set:
            pool_set.add(blk)
            pool.append(list(blk))
            
    num_blocks = len(pool)
    
    # Precompute subset IDs for each block in the pool
    pool_sub_ids = [get_sub_ids(blk) for blk in pool]
    
    # Build inverted index: for each t-subset, which blocks cover it?
    # This is crucial for the WalkSAT "pick variable to satisfy clause" step
    sub_to_blocks = [[] for _ in range(num_subs)]
    for i in range(num_blocks):
        for sid in pool_sub_ids[i]:
            sub_to_blocks[sid].append(i)

    # 2. Local Search (WalkSAT-style)
    best_size = float('inf')
    best_selected = None
    start_time = time.time()
    timeout = 110.0
    
    # Run multiple iterations with different random seeds/initial states
    num_iterations = 5
    
    for iter_seed in range(num_iterations):
        if time.time() - start_time > timeout:
            break
            
        random.seed(2000 + iter_seed * 100)
        
        # Initialize with a greedy cover from the pool? 
        # Or just start empty and let the search build it?
        # Starting empty is slow. Let's start with a quick greedy selection from the pool.
        
        # Quick Greedy Initialization
        uncovered = set(range(num_subs))
        selected_set = set()
        cover_count = [0] * num_subs
        
        # Greedy pick from pool
        # Sort pool by size of coverage? No, just pick randomly or by index.
        # To be fast, we can just iterate and add if it covers new stuff.
        # But we want a good start. Let's use the first few greedy blocks if they are in the pool?
        # It's easier to just run a fast greedy on the pool indices.
        
        # For initialization, we can just use the result of one greedy_cover run 
        # and map it to pool indices if possible, or just use a random subset of the pool 
        # that forms a cover.
        
        # Let's use a simple greedy selection from the pool for initialization.
        # This is O(Pool * Subsets) which might be slow if Pool is 8000 and Subsets is 11628.
        # 8000 * 11628 is ~92M ops. Might be too slow for 5 iterations.
        
        # Alternative: Use the blocks from `greedy_cover` directly as the initial state, 
        # but we need them in the pool to use the `sub_to_blocks` index efficiently?
        # Actually, if we use the `sub_to_blocks` index, we MUST use pool indices.
        
        # Let's try a faster initialization:
        # Pick a random block from pool, add it. Repeat until covered.
        # This is random but valid.
        
        while uncovered:
            # Pick a random uncovered subset
            if not uncovered:
                break
            target_sub_id = random.choice(list(uncovered))
            
            # Get blocks covering this subset
            candidates = sub_to_blocks[target_sub_id]
            if not candidates:
                # Should not happen if pool is good, but if it does, we are stuck
                # Add a random block from pool
                candidates = [random.randint(0, num_blocks - 1)]
            
            # Pick the candidate that covers the most other uncovered subsets
            best_cand = None
            best_gain = -1
            for cand_idx in candidates:
                gain = 0
                for sid in pool_sub_ids[cand_idx]:
                    if sid in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand_idx
            
            if best_cand is not None:
                selected_set.add(best_cand)
                for sid in pool_sub_ids[best_cand]:
                    cover_count[sid] += 1
                    if cover_count[sid] == 1:
                        uncovered.discard(sid)
                
                # After adding, try to remove redundant blocks (Local Search step)
                # This keeps the solution tight from the start
                for i in list(selected_set):
                    redundant = True
                    for sid in pool_sub_ids[i]:
                        if cover_count[sid] == 1:
                            redundant = False
                            break
                    if redundant:
                        for sid in pool_sub_ids[i]:
                            cover_count[sid] -= 1
                            if cover_count[sid] == 0:
                                uncovered.add(sid)
                        selected_set.discard(i)

        # Now we have a valid cover. Let's run the main WalkSAT loop to optimize size.
        current_size = len(selected_set)
        
        # WalkSAT Loop
        stall_count = 0
        while time.time() - start_time < timeout:
            # If we found a better solution, record it
            if current_size < best_size:
                best_size = current_size
                best_selected = list(selected_set)
                
            # If no uncovered subsets, we are at a local minimum (w.r.t deletion)
            # Try to delete more? The initialization and post-add deletion should handle this.
            # If we are here, no block is redundant.
            
            # To escape local minimum, we need to add a block.
            # WalkSAT strategy: 
            # 1. If there are unsatisfied clauses (uncovered subsets), pick one and fix it.
            # 2. If all satisfied, we are in a local minimum. We must "flip" a variable 
            #    that doesn't immediately violate constraints but might lead to a better state.
            #    In Set Cover, "flipping" means adding a block (since removing violates 
            #    constraints if no redundancy). 
            #    So, add a random block (or one guided by some heuristic) and then 
            #    aggressively delete redundant blocks.
            
            # Heuristic for which block to add:
            # Pick a random block from the pool that is NOT currently selected.
            # Or, pick a block that covers many subsets currently covered exactly once?
            # No, those are critical. 
            # Let's just pick a random unselected block.
            
            # Find an unselected block
            # To avoid O(N) scan, we can keep a list of unselected indices? 
            # Or just pick random indices until we find one not in selected_set.
            # With 8000 pool and ~300 selected, probability of hit is high.
            
            added = False
            for _ in range(20): # Max 20 attempts to find an unselected block
                j = random.randint(0, num_blocks - 1)
                if j not in selected_set:
                    # Add block j
                    selected_set.add(j)
                    for sid in pool_sub_ids[j]:
                        cover_count[sid] += 1
                        if cover_count[sid] == 1:
                            uncovered.discard(sid)
                    added = True
                    break
            
            if not added:
                # All blocks selected? Unlikely.
                break
            
            # Now, aggressively delete redundant blocks
            # We can do multiple passes
            deleted_any = True
            while deleted_any:
                deleted_any = False
                # Shuffle to avoid bias? Or just iterate.
                # Iterating through a copy of the set
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
                            if cover_count[sid] == 0:
                                uncovered.add(sid)
                        selected_set.discard(i)
                        deleted_any = True
                        current_size -= 1

            # If we didn't delete anything, we are still in a local minimum.
            # If we added a block and deleted nothing, we are worse off.
            # We should probably revert? Or just continue?
            # WalkSAT usually just continues. But for minimization, if we get worse,
            # we might want to accept it with some probability (Simulated Annealing style)
            # or just revert if we don't improve within X steps.
            
            # For now, let's just continue. The random addition might eventually
            # create a configuration where deletion yields a net gain.

    # Convert best_selected indices back to blocks
    if best_selected is None:
        # Fallback: return a simple greedy cover
        return greedy_cover(v, k, t, sub_to_id, t_subs, num_subs, seed=9999)
        
    best_blocks = [pool[i] for i in best_selected]
    return best_blocks
