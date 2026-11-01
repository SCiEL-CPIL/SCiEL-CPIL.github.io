import itertools
import random
import time
import sys
import numpy as np

# EVOLVE-BLOCK-START

def _get_all_tsets(v, t):
    return list(itertools.combinations(range(v), t))

def _compute_block_masks(blocks, tset_to_idx, t):
    """
    Precompute bitmasks for each block.
    Returns list of integers where bit i is set if tset i is covered by the block.
    """
    masks = []
    for b in blocks:
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tset_to_idx[ts]
            mask |= (1 << idx)
        masks.append(mask)
    return masks

def _count_active_blocks(blocks, masks, full_mask):
    """
    Count the number of blocks that are not redundant.
    A block is redundant if removing it does not change the union of all masks.
    To do this efficiently, we can check if (OR of all masks) == (OR of all masks except this one).
    However, recalculating the OR is O(N).
    For N~100, this is 100 * 100 = 10k ops, which is fast enough.
    
    Returns (active_count, uncovered_count).
    """
    n = len(blocks)
    if n == 0:
        return 0, full_mask.bit_count()
    
    # Calculate total union
    total_union = 0
    for m in masks:
        total_union |= m
        
    # Identify redundant blocks
    active_count = 0
    for i in range(n):
        # Check if block i is essential
        # If total_union == (total_union & ~masks[i]), then block i is redundant
        # Because total_union already contains everything block i covers (covered by others)
        # Wait, logic:
        # If we remove block i, the new union is total_union & ~masks[i]? NO.
        # The union of others is (total_union & ~masks[i]) ONLY IF no other block covers any bit in masks[i].
        # This is not generally true.
        
        # Correct check:
        # Is there any tset covered ONLY by block i?
        # If yes, block i is essential.
        # If no, block i is redundant.
        
        # To check if block i is essential:
        # We need to know if masks[i] contains any bit that is NOT present in (OR of all other masks).
        
        # Let U_others = OR of masks[j] for j != i.
        # If (masks[i] & ~U_others) != 0, then block i is essential.
        
        # Calculating U_others for each i is O(N^2).
        # With N=100, 10,000 operations. Very fast.
        
        u_others = 0
        for j in range(n):
            if i != j:
                u_others |= masks[j]
                
        if (masks[i] & ~u_others) != 0:
            active_count += 1
            
    uncovered_count = full_mask.bit_count() - total_union.bit_count()
    return active_count, uncovered_count

def _construct_pool(v, k, t, rng, size=5000):
    """
    Generate a diverse pool of candidate blocks.
    """
    pool = set()
    
    # 1. Greedy blocks
    # We use a fast greedy to seed the pool
    all_tsets = _get_all_tsets(v, t)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    uncovered = np.ones(len(all_tsets), dtype=bool)
    
    greedy_blocks = []
    while np.any(uncovered):
        # Pick random uncovered tset
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        rand_idx = unc_idx[rng.randint(0, len(unc_idx)-1)]
        seed_ts = all_tsets[rand_idx]
        
        # Complete to block
        rem = [p for p in range(v) if p not in seed_ts]
        combo = tuple(sorted(rng.sample(rem, k-t)))
        block = tuple(sorted(seed_ts + combo))
        
        # Add to pool
        pool.add(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            uncovered[tset_to_idx[ts]] = False
        greedy_blocks.append(block)
        
        if len(greedy_blocks) > 100: # Cap greedy size for pool generation
             break

    # 2. Random blocks
    while len(pool) < size:
        b = tuple(sorted(rng.sample(range(v), k)))
        pool.add(b)
        
    return list(pool)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    rng = random.Random(42)
    
    # 1. Generate Pool
    pool = _construct_pool(v, k, t, rng, size=3000)
    n_pool = len(pool)
    
    # Precompute masks for pool
    all_tsets = _get_all_tsets(v, t)
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    full_mask = (1 << n_tsets) - 1
    
    pool_masks = _compute_block_masks(pool, tset_to_idx, t)
    
    # 2. Initialize Solution
    # Start with a greedy solution, then pad with random pool blocks to a fixed size
    # Size should be around the expected optimum + slack
    # Optimum is ~76. Let's use 80.
    
    # Get a quick greedy solution to start
    current_blocks = []
    uncovered = np.ones(n_tsets, dtype=bool)
    
    # Simple greedy to get initial valid covering
    while np.any(uncovered) and time.time() < deadline:
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0: break
        rand_idx = unc_idx[rng.randint(0, len(unc_idx)-1)]
        seed_ts = all_tsets[rand_idx]
        rem = [p for p in range(v) if p not in seed_ts]
        combo = tuple(sorted(rng.sample(rem, k-t)))
        block = tuple(sorted(seed_ts + combo))
        current_blocks.append(block)
        for ts in itertools.combinations(block, t):
            uncovered[tset_to_idx[ts]] = False
            
    # If we didn't cover everything (shouldn't happen), add randoms
    while np.any(uncovered):
        b = tuple(sorted(rng.sample(range(v), k)))
        current_blocks.append(b)
        for ts in itertools.combinations(b, t):
            uncovered[tset_to_idx[ts]] = False
            
    # Pad to fixed size N_FIXED
    N_FIXED = 80
    while len(current_blocks) < N_FIXED:
        # Add random blocks from pool
        b = pool[rng.randint(0, n_pool-1)]
        current_blocks.append(b)
        
    # Ensure we have exactly N_FIXED blocks
    current_blocks = current_blocks[:N_FIXED]
    
    # Map current blocks to pool indices?
    # Actually, for SA, it's easier to work with indices into the pool.
    # But the initial greedy blocks might not be in the pool.
    # Let's add current blocks to the pool if they aren't there.
    
    # Rebuild pool to include current blocks
    pool_set = set(pool)
    for b in current_blocks:
        if b not in pool_set:
            pool.append(b)
            pool_set.add(b)
            
    n_pool = len(pool)
    pool_masks = _compute_block_masks(pool, tset_to_idx, t)
    
    # Map current blocks to indices in pool
    block_to_idx = {b: i for i, b in enumerate(pool)}
    current_indices = [block_to_idx[b] for b in current_blocks]
    
    # 3. Simulated Annealing
    best_indices = current_indices[:]
    best_score = float('inf')
    
    # Objective: Minimize (active_count * 100 + uncovered_count)
    # We want to minimize active_count first, then uncovered.
    # But we need a valid covering (uncovered=0) to score.
    # During search, we allow uncovered > 0 to escape local minima,
    # but we only update 'best' if uncovered == 0.
    
    def get_score(indices):
        masks = [pool_masks[i] for i in indices]
        active, unc = _count_active_blocks(indices, masks, full_mask)
        # Weights: Active blocks are the cost. Uncovered are a penalty to ensure validity.
        # If unc > 0, the solution is invalid, so we don't update best, but we might accept the move.
        # Score for acceptance: active * 1000 + unc
        return active * 1000 + unc

    current_score = get_score(current_indices)
    
    T = 1000.0
    T_min = 1.0
    alpha = 0.995 # Cooling rate
    
    step = 0
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        # Swap one block index with a random other pool index
        pos = rng.randint(0, N_FIXED - 1)
        old_idx = current_indices[pos]
        new_idx = rng.randint(0, n_pool - 1)
        
        if old_idx == new_idx:
            continue
            
        current_indices[pos] = new_idx
        
        new_score = get_score(current_indices)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            # Check if valid and better
            # We need to recompute active/unc to check validity
            # get_score returns combined. Let's unpack.
            masks = [pool_masks[i] for i in current_indices]
            active, unc = _count_active_blocks(current_indices, masks, full_mask)
            if unc == 0:
                if active < best_score:
                    best_score = active
                    best_indices = current_indices[:]
        else:
            # Revert
            current_indices[pos] = old_idx
            
        T *= alpha
        step += 1
        
        # Periodic restart if stuck?
        # If no improvement for a long time, maybe re-randomize?
        # For now, simple SA.
        
    # 4. Final Polish
    # Take best_indices, convert to blocks
    best_blocks = [pool[i] for i in best_indices]
    
    # Remove duplicates and redundancies to get the minimal set
    # Re-run a quick reduction on the best set found
    best_blocks = list(set(best_blocks)) # Remove exact duplicates
    
    # Redundancy removal
    all_tsets = _get_all_tsets(v, t)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    n_tsets = len(all_tsets)
    full_mask = (1 << n_tsets) - 1
    
    # Compute masks for best_blocks
    best_masks = _compute_block_masks(best_blocks, tset_to_idx, t)
    
    # Greedy removal of redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(best_blocks)
        for i in range(n):
            if time.time() > deadline:
                break
            # Check if block i is redundant
            u_others = 0
            for j in range(n):
                if i != j:
                    u_others |= best_masks[j]
            if (best_masks[i] & ~u_others) == 0:
                # Redundant
                best_blocks.pop(i)
                best_masks.pop(i)
                changed = True
                break # Restart loop as indices changed
                
    return best_blocks

# EVOLVE-BLOCK-END
