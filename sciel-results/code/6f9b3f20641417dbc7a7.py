import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Covers design C(v, k, t) using a Destructive Set-Cover approach with Simulated Annealing.
    
    Strategy:
    1. Generate a large pool of random k-subsets (blocks).
    2. Select a subset of these blocks that covers all t-subsets (a valid covering).
       This is done by a greedy set-cover selection on the random pool, which creates 
       significant redundancy (over-coverage).
    3. Use Simulated Annealing to minimize the number of blocks in this valid covering.
       Moves: Remove a block (if coverage remains valid).
       Cost: Number of blocks.
       Since we start with a valid covering, we only need to check if removal breaks validity.
       We do not add blocks, ensuring validity is maintained. The search space is the 
       hypercube of included/excluded blocks from the initial pool.
    """
    deadline = time.time() + 140.0  # Safety margin
    start_time = time.time()
    
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute the t-subsets covered by every possible k-block
    # There are C(24, 7) = 346,104 possible blocks.
    # Each covers C(7, 3) = 35 t-subsets.
    # We will generate a random pool of candidate blocks.
    
    # Step 1: Generate a random pool of candidate blocks
    # To ensure we can find a covering, we need enough blocks.
    # C(24, 3) = 2024 t-subsets.
    # Each block covers 35.
    # If we pick N random blocks, expected coverage is high.
    # Let's pick a pool of size P.
    # We want P to be large enough to allow for many redundant choices, 
    # so that the SA has room to optimize.
    # Let's aim for an initial covering of ~150 blocks from a pool of ~500.
    
    rng = random.Random(42)
    pool_size = 600
    all_possible_blocks = list(itertools.combinations(range(v), k))
    
    # Randomly sample the pool
    # Using set to ensure distinct blocks
    pool_blocks = set()
    while len(pool_blocks) < pool_size:
        pool_blocks.add(rng.choice(all_possible_blocks))
    pool_blocks = list(pool_blocks)
    
    # Map each pool block to its covered t-subset indices
    # pool_block_cover[i] is a frozenset of t-subset indices covered by pool_blocks[i]
    pool_block_cover = []
    for b in pool_blocks:
        covered = frozenset(t_subset_to_idx[ts] for ts in itertools.combinations(b, t))
        pool_block_cover.append(covered)
    
    # Step 2: Greedy Set-Cover to find an initial valid covering from the pool
    # We select blocks from the pool to cover all n_t t-subsets.
    uncovered = set(range(n_t))
    selected_indices = []
    selected_covers = []
    
    # To make the initial covering "dense" (redundant), we don't just pick the absolute best,
    # but we can pick the best. The redundancy comes from the fact that many pool blocks
    # cover similar areas, and the greedy might pick one, leaving others as candidates 
    # for SA to potentially swap/remove if we allowed swaps. 
    # However, pure removal SA on a greedy solution often gets stuck.
    # Better: Initialize with a Random Valid Covering.
    # How? 
    # 1. Pick a random block from pool that covers an uncovered t-subset.
    # 2. This tends to produce a covering with more "slack" than optimal greedy, 
    #    allowing SA to find a smaller subset.
    
    # Let's try a hybrid: Greedy for speed, but we can also try a "Randomized Greedy"
    # where we pick from the top few candidates.
    
    def get_initial_covering(mode="random_greedy"):
        unc = set(range(n_t))
        sel_idx = []
        sel_cov = []
        
        # Precompute for each t-subset, which pool blocks cover it
        # This is O(n_t * pool_size) which is 2024 * 600 ~ 1.2M ops. Acceptable.
        ts_to_blocks = [[] for _ in range(n_t)]
        for i, cov in enumerate(pool_block_cover):
            for idx in cov:
                ts_to_blocks[idx].append(i)
        
        while unc:
            if time.time() > deadline - 20.0:
                # Fallback: just return what we have (might be invalid if time ran out, 
                # but we should ensure validity in main loop)
                break
            
            if mode == "greedy":
                # Find block covering max unc
                best_i = -1
                best_count = -1
                # Pick a random unc t-subset to break ties or just scan all?
                # Scanning all unc is slow. 
                # Let's pick a random unc t-subset, and look at blocks covering it.
                # This is a standard randomized set cover heuristic.
                target_ts = rng.choice(list(unc))
                candidates = ts_to_blocks[target_ts]
                if not candidates:
                    # Should not happen if pool is good, but if it does, we are stuck.
                    # For now, break.
                    break
                for i in candidates:
                    count = len(pool_block_cover[i] & unc)
                    if count > best_count:
                        best_count = count
                        best_i = i
            else: # random_greedy
                # Pick a random unc t-subset
                target_ts = rng.choice(list(unc))
                candidates = ts_to_blocks[target_ts]
                if not candidates:
                    break
                # Pick a random candidate? Or pick from top 3?
                # Let's pick a random one to ensure diversity and redundancy.
                best_i = rng.choice(candidates)
                best_count = len(pool_block_cover[best_i] & unc)
            
            if best_i == -1:
                break
                
            sel_idx.append(best_i)
            sel_cov.append(pool_block_cover[best_i])
            unc -= pool_block_cover[best_i]
            
        return sel_idx, sel_cov
    
    # We will run multiple initializations and SA on the best one
    best_final_blocks = None
    best_final_count = float('inf')
    
    # Run a few different initializations
    num_initials = 5
    
    for init_i in range(num_initials):
        if time.time() > deadline - 10.0:
            break
            
        # Use random_greedy to get a denser, more redundant start
        init_mode = "random_greedy" if init_i % 2 == 0 else "greedy"
        sel_idx, sel_cov = get_initial_covering(init_mode)
        
        if not sel_idx:
            continue
            
        # Verify validity
        covered = set()
        for cov in sel_cov:
            covered |= cov
        if covered != set(range(n_t)):
            continue
            
        current_count = len(sel_idx)
        
        # Simulated Annealing
        # State: set of selected indices (from pool)
        # We represent state as a boolean array of size pool_size? 
        # No, pool_size is 600. 
        # It's better to keep a list of selected indices.
        # But removal is easy. 
        # The problem is: if we remove a block, we might need to add a different one 
        # from the pool to maintain coverage? 
        # NO. The task is to find a subset of the POOL that covers everything.
        # If we remove a block and it breaks coverage, we reject the move.
        # This is a pure minimization of a subset.
        # This is equivalent to: Given a universe U and a collection of sets S (pool),
        # find the smallest sub-collection that covers U.
        # This is Set Cover.
        # SA Move: 
        # 1. Pick a random selected block. Try to remove it.
        #    If valid, accept (cost decreases by 1).
        # 2. This is a "hill climbing" towards local minima of size.
        #    It will get stuck at a minimal cover (no single block can be removed).
        #    To escape, we need to allow "add" moves? 
        #    If we allow adding a block, cost increases. 
        #    Standard SA accepts worse states with probability exp(-delta/T).
        #    So we can allow adding a block (from pool, not currently selected) 
        #    to enable future removals of two blocks?
        #    Move types:
        #    A. Remove a selected block. (Cost -1)
        #    B. Add a non-selected block from pool. (Cost +1)
        #    C. Swap a selected block with a non-selected block. (Cost 0)
        
        # Let's implement A, B, and C.
        
        selected_set = set(sel_idx)
        # For fast coverage check, we maintain cover_count array
        cover_count = np.zeros(n_t, dtype=np.int32)
        for i in sel_idx:
            for idx in pool_block_cover[i]:
                cover_count[idx] += 1
        
        # Check if any count is 0 -> invalid (shouldn't happen if init valid)
        
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        max_steps = 5000
        steps = 0
        
        while T > T_min and steps < max_steps:
            if time.time() > deadline - 5.0:
                break
                
            steps += 1
            move_type = rng.random()
            
            if move_type < 0.7:
                # Move A: Remove
                if not selected_set:
                    continue
                i = rng.choice(list(selected_set))
                # Check if removal is valid
                # We need to check if any t-subset covered by block i has count 1
                # We can precompute the min count for block i? 
                # Or just iterate its covers.
                min_count = n_t
                for idx in pool_block_cover[i]:
                    if cover_count[idx] < min_count:
                        min_count = cover_count[idx]
                    if min_count == 1:
                        break
                if min_count == 1:
                    continue # Cannot remove
                
                # Apply removal
                for idx in pool_block_cover[i]:
                    cover_count[idx] -= 1
                selected_set.remove(i)
                current_count -= 1
                
                # Always accept improvement
                # T stays same or cools? Usually cool on accept.
                T *= cooling
                
            elif move_type < 0.85:
                # Move B: Add
                # Pick a random block from pool not in selected_set
                # To speed up, pick random i in range(pool_size)
                # If it's in selected_set, skip
                i = rng.randint(0, pool_size - 1)
                if i in selected_set:
                    continue
                    
                # Apply add
                for idx in pool_block_cover[i]:
                    cover_count[idx] += 1
                selected_set.add(i)
                current_count += 1
                
                # Accept with probability?
                # Delta = +1
                # p = exp(-1/T)
                if rng.random() < np.exp(-1.0 / T):
                    T *= cooling
                else:
                    # Reject
                    for idx in pool_block_cover[i]:
                        cover_count[idx] -= 1
                    selected_set.remove(i)
                    current_count -= 1
                    T *= cooling # Cool anyway? Or not? Usually cool on every step.
                    
            else:
                # Move C: Swap
                # Pick i in selected_set, j not in selected_set
                if not selected_set:
                    continue
                i = rng.choice(list(selected_set))
                j = rng.randint(0, pool_size - 1)
                if j in selected_set:
                    continue
                    
                # Check validity of swap:
                # Remove i, Add j.
                # We need to ensure that after removing i, all t-subsets covered by i 
                # are either covered by j OR covered by other blocks.
                # i.e., for all idx in cover(i), (count[idx] - 1 + (1 if idx in cover(j) else 0)) > 0
                # Simplifies to: if idx in cover(i) and idx NOT in cover(j), then count[idx] > 1.
                
                valid_swap = True
                for idx in pool_block_cover[i]:
                    if idx not in pool_block_cover[j]:
                        if cover_count[idx] == 1:
                            valid_swap = False
                            break
                            
                if not valid_swap:
                    continue
                
                # Apply swap
                for idx in pool_block_cover[i]:
                    cover_count[idx] -= 1
                for idx in pool_block_cover[j]:
                    cover_count[idx] += 1
                selected_set.remove(i)
                selected_set.add(j)
                # current_count unchanged
                
                T *= cooling
                
        # End SA loop
        if current_count < best_final_count:
            best_final_count = current_count
            best_final_blocks = [pool_blocks[i] for i in selected_set]
            
    # Fallback if SA failed or didn't run
    if best_final_blocks is None:
        # Use a simple greedy on the full space (slow but safe)
        # Or just return the first valid init we found
        # Let's re-run a deterministic greedy on the full space as a last resort
        # But we must fit in time.
        # Actually, the initializations in the loop should have produced valid coverings.
        # If best_final_blocks is None, it means no valid covering was found in the pool?
        # That implies the pool was bad.
        # Let's just do a standard greedy on the fly.
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            if time.time() > deadline - 2.0:
                break
            # Pick random uncovered t-subset
            ts = rng.choice(list(uncovered))
            # Find a random k-block containing this t-subset
            # This is hard to do randomly.
            # Let's just use the pool again.
            # Find a block in pool that covers ts
            found = False
            for i, cov in enumerate(pool_block_cover):
                if t_subset_to_idx[ts] in cov:
                    blocks.append(pool_blocks[i])
                    for idx in cov:
                        uncovered.discard(idx)
                    found = True
                    break
            if not found:
                # If pool doesn't cover it, we are stuck. 
                # This shouldn't happen if pool is large enough.
                break
        
        if uncovered:
             # Last resort: construct manually
             # This is very slow. 
             # But we must return something valid.
             # Let's assume the pool is good.
             pass
        best_final_blocks = blocks

    # Final verification
    if best_final_blocks is None:
        return []
        
    covered = set()
    for b in best_final_blocks:
        for ts in itertools.combinations(b, t):
            covered.add(t_subset_to_idx[ts])
            
    if covered != set(range(n_t)):
        # Invalid. Return a valid one from a quick greedy
        # This is a safety net.
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            ts = min(uncovered) # Deterministic
            # Find block in pool
            found = False
            for i, cov in enumerate(pool_block_cover):
                if ts in cov:
                    blocks.append(pool_blocks[i])
                    for idx in cov:
                        uncovered.discard(idx)
                    found = True
                    break
            if not found:
                break
        best_final_blocks = blocks

    return best_final_blocks

# To ensure the function signature is correct and handles edge cases
if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Blocks: {len(blocks)}")
