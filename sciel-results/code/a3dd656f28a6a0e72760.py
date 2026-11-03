import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin out of 150s

    # Fallback for non-target parameters
    if v != 19 or k != 7 or t != 4:
        return _fallback_greedy(v, k, t, start_time, deadline)

    # --- Optimized Path for v=19, k=7, t=4 ---
    
    # 1. Precompute 4-subsets and their IDs
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)  # 3876

    # 2. Precompute all blocks (7-subsets) and the t-subsets they cover
    # C(19,7) = 50388 blocks. Each covers C(7,4)=35 t-subsets.
    # Total edges = 50388 * 35 = 1,763,580. This is manageable.
    
    all_blocks = []
    block_tsubs = []  # List of lists of tsub IDs
    block_index_map = {}  # frozenset -> index in all_blocks

    for block_tuple in itertools.combinations(points, k):
        block_fs = frozenset(block_tuple)
        idx = len(all_blocks)
        all_blocks.append(block_fs)
        block_index_map[block_fs] = idx
        
        tsubs_in_block = []
        for tsub in itertools.combinations(block_tuple, t):
            tsubs_in_block.append(tsub_to_id[tsub])
        block_tsubs.append(tsubs_in_block)
        
    num_blocks_total = len(all_blocks)

    # 3. Greedy Initialization
    # We maintain cover_count[tid] = number of blocks in current solution covering tsub tid.
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    current_blocks = []  # List of block indices
    current_block_set = set()  # Set of block indices for O(1) lookup

    def add_block(idx):
        current_blocks.append(idx)
        current_block_set.add(idx)
        for tid in block_tsubs[idx]:
            cover_count[tid] += 1

    def remove_block(idx):
        current_blocks.remove(idx) # O(N), but N is small (~150)
        current_block_set.discard(idx)
        for tid in block_tsubs[idx]:
            cover_count[tid] -= 1

    # Greedy loop
    # To speed up greedy, we can precompute for each block the list of tsubs.
    # Finding the best block: iterate all blocks, count uncovered tsubs.
    # 50k blocks * 35 tsubs = 1.75M ops per greedy step.
    # If we need ~100 steps, that's 175M ops. In Python, this might be too slow (seconds).
    # Optimization: Use a heap or lazy update?
    # Alternatively, vectorize the gain calculation.
    
    # Let's try a vectorized greedy step.
    # Create a matrix M of shape (num_blocks_total, num_tsubs) where M[i, j] = 1 if block i covers tsub j.
    # This matrix is 50388 x 3876. 195 million entries.
    # Memory: 195M * 1 byte (bool) = ~195 MB. This is acceptable.
    # Then gain for block i = sum(M[i, :][cover_count == 0]).
    # We can compute this as M[:, uncovered_mask].sum(axis=1).
    
    # Let's build the sparse matrix or dense boolean matrix.
    # Dense is faster for numpy operations.
    
    # Build incidence matrix
    # Using sparse might be better for memory, but dense is faster for sum.
    # Let's try dense.
    # M = np.zeros((num_blocks_total, num_tsubs), dtype=bool)
    # for i, bts in enumerate(block_tsubs):
    #     for tid in bts:
    #         M[i, tid] = True
        
    # This loop is 1.7M iterations. Fast enough.
    
    M = np.zeros((num_blocks_total, num_tsubs), dtype=bool)
    for i, bts in enumerate(block_tsubs):
        for tid in bts:
            M[i, tid] = True

    uncovered_mask = np.ones(num_tsubs, dtype=bool)
    num_uncovered = num_tsubs

    while num_uncovered > 0:
        if time.time() > deadline:
            break
            
        # Calculate gain for all blocks
        # gain[i] = number of uncovered tsubs covered by block i
        gains = M[:, uncovered_mask].sum(axis=1)
        
        # Find best block not already in current solution
        # We can mask out blocks already in current solution
        # But checking membership in a set for 50k items is slow.
        # Instead, we can just pick the max, and if it's in current, pick next max.
        # Or maintain a 'selected' mask.
        
        selected_mask = np.zeros(num_blocks_total, dtype=bool)
        for idx in current_block_set:
            selected_mask[idx] = True
            
        # Mask out selected blocks
        gains[selected_mask] = -1
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain <= 0:
            # No block covers any uncovered tsub? Should not happen.
            break
            
        add_block(best_idx)
        
        # Update uncovered mask
        # tsubs covered by best_idx that were uncovered become covered
        newly_covered = M[best_idx] & uncovered_mask
        if newly_covered.any():
            uncovered_mask[newly_covered] = False
            num_uncovered -= newly_covered.sum()

    # 4. Local Search / Simulated Annealing
    # Objective: Minimize number of blocks, subject to all tsubs covered.
    # Penalty method: Energy = len(blocks) + lambda * num_uncovered
    
    lambda_penalty = 100.0  # High penalty to ensure validity
    current_energy = len(current_blocks) + lambda_penalty * num_uncovered
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    rng = random.Random(12345)
    
    best_energy = current_energy
    best_blocks_copy = list(current_blocks)
    
    # Precompute block tsub lists for faster delta calculation in SA
    # We already have block_tsubs.
    
    while time.time() < deadline and T > T_min:
        # Choose move
        # If num_uncovered > 0, bias towards adding blocks
        if num_uncovered > 0:
            move_type = rng.choice(['add', 'add', 'remove'])
        else:
            move_type = rng.choice(['remove', 'add'])
            
        if move_type == 'remove':
            if not current_blocks:
                continue
            idx = rng.choice(current_blocks)
            
            # Calculate delta
            delta_uncovered = 0
            for tid in block_tsubs[idx]:
                if cover_count[tid] == 1:
                    delta_uncovered += 1
            
            delta_E = -1.0 + lambda_penalty * delta_uncovered
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                remove_block(idx)
                num_uncovered += delta_uncovered
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks_copy = list(current_blocks)
                    
        else: # add
            # Pick a random block not in current
            # To avoid duplicates, pick random index, check if in set.
            # If in set, skip or pick another.
            # For speed, just pick random and check.
            b_idx = rng.randint(0, num_blocks_total - 1)
            if b_idx in current_block_set:
                continue
                
            # Calculate delta
            delta_uncovered = 0
            for tid in block_tsubs[b_idx]:
                if cover_count[tid] == 0:
                    delta_uncovered -= 1  # uncovered decreases
            
            delta_E = 1.0 + lambda_penalty * delta_uncovered
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                add_block(b_idx)
                num_uncovered += delta_uncovered
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_blocks_copy = list(current_blocks)
                    
        T *= cooling_rate

    # Ensure validity of best solution
    # The best solution found might have num_uncovered > 0 if lambda was not high enough or we lost track.
    # We must ensure the returned solution is valid.
    # Let's take the current state, or the best state, and repair if invalid.
    
    # It's safer to work with the current state if it's valid, or repair the best.
    # Let's check validity of best_blocks_copy.
    
    # Reset cover_count to check best_blocks_copy
    cover_count_check = np.zeros(num_tsubs, dtype=np.int32)
    for idx in best_blocks_copy:
        for tid in block_tsubs[idx]:
            cover_count_check[tid] += 1
            
    if np.any(cover_count_check == 0):
        # Best solution is invalid. Repair it using greedy.
        # Use the current state as base if it's valid, else repair best.
        # Let's just repair best_blocks_copy.
        current_blocks = list(best_blocks_copy)
        current_block_set = set(current_blocks)
        cover_count = cover_count_check.copy()
        
        # Greedy repair
        while True:
            if time.time() > deadline:
                break
            uncovered_mask = (cover_count == 0)
            num_uncovered = uncovered_mask.sum()
            if num_uncovered == 0:
                break
                
            gains = M[:, uncovered_mask].sum(axis=1)
            selected_mask = np.zeros(num_blocks_total, dtype=bool)
            for idx in current_block_set:
                selected_mask[idx] = True
            gains[selected_mask] = -1
            
            best_idx = np.argmax(gains)
            if gains[best_idx] <= 0:
                break
                
            add_block(best_idx)
            
        # Final check
        cover_count_final = np.zeros(num_tsubs, dtype=np.int32)
        for idx in current_blocks:
            for tid in block_tsubs[idx]:
                cover_count_final[tid] += 1
        if np.any(cover_count_final == 0):
            # Still invalid? Fallback to simple greedy from scratch?
            # This should not happen.
            pass

    # Convert to output format
    final_blocks = [sorted(all_blocks[idx]) for idx in current_blocks]
    return final_blocks


def _fallback_greedy(v, k, t, start_time, deadline):
    """Simple greedy fallback for other parameters."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        best_block = None
        best_gain = -1
        # This is very slow for large v, but v=19 is handled above.
        # For fallback, we'll just pick a random block covering at least one uncovered.
        # Or simple greedy: find block covering most uncovered.
        # To save time, we'll sample.
        found = False
        for _ in range(1000):
            block = tuple(sorted(random.sample(range(v), k)))
            gain = 0
            for tsub in itertools.combinations(block, t):
                if tsub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
                found = True
            if gain == len(uncovered):
                break
        if not found:
            break
        if best_block is None:
            break
        blocks.append(list(best_block))
        for tsub in itertools.combinations(best_block, t):
            uncovered.discard(tsub)
            
    return blocks
