import time
import random
import itertools
import math
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0

    # Precompute
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    idx = {s: i for i, s in enumerate(all_t)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    B = len(all_blocks)
    
    # block_cover[bi] = list of t-subset indices covered by block bi
    block_cover = []
    for b in all_blocks:
        covered = [idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # t_to_blocks[ti] = list of block indices that cover t-subset ti
    t_to_blocks = [[] for _ in range(N)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Precompute cover sets as numpy arrays for fast operations
    # block_cover_np[bi] = np.array of t-subset indices
    block_cover_np = [np.array(cov, dtype=np.int32) for cov in block_cover]
    
    # Precompute for each block, the count of t-subsets it covers (always C(8,5)=56)
    # Not strictly needed but good to know.

    def greedy_cover():
        """Standard greedy set cover to get a valid initial solution."""
        uncovered = set(range(N))
        blocks = []
        while uncovered:
            seed_idx = next(iter(uncovered))
            candidates = t_to_blocks[seed_idx]
            best_b = -1
            max_cov = -1
            for bi in candidates:
                cov = 0
                for ti in block_cover[bi]:
                    if ti in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
            if best_b == -1:
                break
            blocks.append(best_b)
            for ti in block_cover[best_b]:
                uncovered.discard(ti)
        return blocks

    def prune_redundant(blocks):
        """Remove blocks that are not the sole cover for any t-subset."""
        cnt = np.zeros(N, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                cnt[ti] += 1
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in blocks:
                essential = False
                for ti in block_cover[bi]:
                    if cnt[ti] == 1:
                        essential = True
                        break
                if not essential:
                    changed = True
                    for ti in block_cover[bi]:
                        cnt[ti] -= 1
                else:
                    new_blocks.append(bi)
            blocks = new_blocks
            if time.time() > deadline - 2.0:
                break
        return blocks

    # 1. Initial Solution
    init_blocks = greedy_cover()
    init_blocks = prune_redundant(init_blocks)
    
    best_blocks = list(init_blocks)
    best_score = len(best_blocks)
    
    rng = random.Random(42)
    
    # 2. Dual-Guided Local Search
    # State: current set of blocks
    # We maintain:
    # - current_blocks: list of block indices
    # - cnt: np.array of size N, count of covers for each t-subset
    # - in_set: np.array of size B, boolean if block is in current set
    
    current_blocks = list(best_blocks)
    current_cost = len(current_blocks)
    
    cnt = np.zeros(N, dtype=np.int32)
    in_set = np.zeros(B, dtype=bool)
    for bi in current_blocks:
        in_set[bi] = True
        for ti in block_cover[bi]:
            cnt[ti] += 1
            
    # Dual variables for uncovered subsets (conceptually)
    # In this heuristic, we use "uncovered count" or "deficiency" to guide selection.
    # When we drop a block, some subsets become "critical" (cnt drops).
    # If they become 0, they are uncovered.
    
    # Strategy:
    # 1. Pick a block to drop (random, or weighted by "importance" = number of unique covers it provides).
    # 2. Determine the set U of t-subsets that become uncovered (cnt becomes 0).
    # 3. If U is empty, accept (cost -1).
    # 4. Else, find a set of replacement blocks R to cover U.
    #    - Use a greedy repair that prioritizes blocks covering the most "critical" subsets.
    #    - "Critical" here means subsets in U.
    #    - To make it "dual-guided", we can weight the gain of a block by the sum of duals of the subsets it covers.
    #    - Let dual[ti] = 1 if ti is in U, else 0. (Simple primal-dual step)
    #    - Actually, to escape local minima, we might want to add blocks that cover subsets *not* in U but are "weakly covered" (cnt=1)?
    #    - Let's stick to standard repair first, but with a twist: 
    #      Instead of just covering U, we might swap. 
    #      But the directive asks for "dual-guided ... select high-value blocks".
      
    # Let's implement a specific "Drop and Repair" with a smarter repair.
    # The repair should minimize the number of added blocks.
    # Standard greedy does this well.
    # To improve, we can try multiple random drops and keep the best result.
    
    # Optimization: Precompute for each block, the list of t-subsets it covers.
    # We have block_cover.
    
    # To speed up repair:
    # When looking for a block to cover a subset of U, we check candidates from t_to_blocks[seed].
    # We need to count how many of U it covers.
    
    steps = 0
    max_steps = 20000
    T = 5.0 # Temperature for simulated annealing acceptance of cost-increasing moves (rarely needed if repair is good, but good for escape)
    cooling = 0.9995
    
    # Precompute block_cover as a list of lists for faster iteration in Python?
    # block_cover is already list of lists.
    
    while time.time() < deadline - 2.0 and steps < max_steps:
        steps += 1
        
        if len(current_blocks) == 0:
            break
            
        # Select block to drop
        # Bias towards blocks that are "less essential"?
        # Essentiality = number of t-subsets where cnt[ti] == 1.
        # Let's just pick randomly for now, or pick the block with the lowest essentiality.
        
        # To speed up, let's just pick randomly.
        idx_drop = rng.randrange(len(current_blocks))
        block_drop = current_blocks[idx_drop]
        
        # Tentatively remove
        U = []
        for ti in block_cover[block_drop]:
            cnt[ti] -= 1
            if cnt[ti] == 0:
                U.append(ti)
        
        if not U:
            # Valid drop
            current_blocks.pop(idx_drop)
            in_set[block_drop] = False
            current_cost -= 1
            if current_cost < best_score:
                best_score = current_cost
                best_blocks = list(current_blocks)
            T *= cooling
            continue
            
        # Need to repair U
        # Dual-guided repair:
        # We want to find a set of blocks R such that union(cover(b) for b in R) covers U.
        # Minimize |R|.
        # Greedy:
        # 1. Let U_set = set(U).
        # 2. While U_set is not empty:
        #    a. Find block b not in current_blocks that covers the most elements of U_set.
        #    b. If no such block, fail (should not happen).
        #    c. Add b to R.
        #    d. Remove covered elements from U_set.
        
        U_set = set(U)
        R = []
        
        # To speed up finding the best block, we can iterate through candidates of a random element in U_set.
        # But this might miss a block that covers many elements but doesn't cover the specific random element?
        # No, if it covers many elements, it covers at least one.
        # However, picking the *best* block globally requires checking all blocks not in set.
        # B is ~75k. U_set size is small (<=56).
        # Checking all 75k blocks per repair step is expensive.
        
        # Better: Iterate through candidates of a seed element.
        # But a block covering 50% of U might not cover the seed.
        # So we should check candidates of *all* elements in U? That's expensive.
        
        # Alternative:
        # Since |U| is small (max 56), and each block covers 56 t-subsets.
        # We can precompute for each block, the intersection size with U? No, U changes.
        
        # Let's use a heuristic:
        # Pick 3 random seeds from U_set.
        # For each seed, get candidates.
        # For each candidate, compute coverage of U_set.
        # Pick the best.
        
        best_b = -1
        max_cov = -1
        candidates_checked = set()
        
        # To ensure we don't miss good blocks, we can check candidates of a few seeds.
        seeds = []
        if len(U_set) > 0:
            seeds = list(rng.sample(list(U_set), min(3, len(U_set))))
        
        for seed in seeds:
            for bi in t_to_blocks[seed]:
                if bi in candidates_checked:
                    continue
                if in_set[bi]:
                    continue
                candidates_checked.add(bi)
                
                cov = 0
                for ti in block_cover[bi]:
                    if ti in U_set:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
                    
        if best_b == -1:
            # Fallback: check all candidates of the first seed
            seed = next(iter(U_set))
            for bi in t_to_blocks[seed]:
                if in_set[bi]:
                    continue
                cov = 0
                for ti in block_cover[bi]:
                    if ti in U_set:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_b = bi
                    
        if best_b == -1:
            # Should not happen, but handle it
            # Restore state
            for ti in block_cover[block_drop]:
                cnt[ti] += 1
            continue
            
        R.append(best_b)
        
        # Update U_set
        for ti in block_cover[best_b]:
            U_set.discard(ti)
            
        # We might need more blocks if U_set is not empty.
        # Repeat the process.
        # To limit time, we'll do this in a loop.
        
        while U_set and time.time() < deadline - 2.0:
            best_b = -1
            max_cov = -1
            candidates_checked = set()
            
            seeds = list(rng.sample(list(U_set), min(3, len(U_set))))
            for seed in seeds:
                for bi in t_to_blocks[seed]:
                    if bi in candidates_checked:
                        continue
                    if in_set[bi]:
                        continue
                    candidates_checked.add(bi)
                    
                    cov = 0
                    for ti in block_cover[bi]:
                        if ti in U_set:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_b = bi
                        
            if best_b == -1:
                break
                
            R.append(best_b)
            for ti in block_cover[best_b]:
                U_set.discard(ti)
                
        # If U_set is not empty, repair failed. Reject move.
        if U_set:
            # Restore cnt
            for ti in block_cover[block_drop]:
                cnt[ti] += 1
            T *= cooling
            continue
            
        # Apply Repair
        # Add blocks in R
        for bi in R:
            current_blocks.append(bi)
            in_set[bi] = True
            for ti in block_cover[bi]:
                cnt[ti] += 1
                
        # Remove dropped block from list
        # Note: We already decremented cnt for block_drop.
        # We need to remove it from current_blocks and in_set.
        current_blocks.pop(idx_drop)
        in_set[block_drop] = False
        
        new_cost = len(current_blocks)
        delta = new_cost - current_cost
        
        # Acceptance
        # If delta < 0, accept.
        # If delta == 0, accept with prob 0.5 (to explore).
        # If delta > 0, accept with prob exp(-delta/T).
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = rng.random() < 0.5
        else:
            accept = rng.random() < math.exp(-delta / T)
            
        if accept:
            current_cost = new_cost
            if new_cost < best_score:
                best_score = current_cost
                best_blocks = list(current_blocks)
        else:
            # Reject: Undo
            # Remove R
            for bi in R:
                # Find and remove from current_blocks
                # Since we just added them, they are at the end.
                # But order might be mixed? No, we appended.
                # However, if we did multiple swaps, order changes.
                # Safer to remove by value.
                if bi in current_blocks:
                    current_blocks.remove(bi)
                    in_set[bi] = False
                    for ti in block_cover[bi]:
                        cnt[ti] -= 1
            # Re-add block_drop
            current_blocks.insert(idx_drop, block_drop)
            in_set[block_drop] = True
            for ti in block_cover[block_drop]:
                cnt[ti] += 1
                
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Final Prune
    best_blocks = prune_redundant(best_blocks)
    
    # Verify
    cnt_final = np.zeros(N, dtype=np.int32)
    for bi in best_blocks:
        for ti in block_cover[bi]:
            cnt_final[ti] += 1
            
    if np.min(cnt_final) < 1:
        # Fallback
        return [list(all_blocks[bi]) for bi in greedy_cover()]
        
    return [list(all_blocks[bi]) for bi in best_blocks]
