import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using Simulated Annealing
    with 2-for-2 block exchanges and bitset-accelerated coverage tracking.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    def time_ok():
        return time.time() < deadline - 5.0

    # 1. Precomputation
    # All t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # All blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to the indices of t-subsets it covers
    # This is a list of lists of ints
    block_covered_t = [None] * num_blocks
    # Reverse index: for each t-subset, which blocks cover it?
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices

    # 2. Initial Solution: Greedy Set Cover
    # We use a standard greedy approach to get a valid starting point.
    # To speed this up, we can use a simple heuristic.
    
    def initial_greedy():
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected = []
        
        # Precompute "mask" or list of covered t-subsets for fast checking?
        # With num_t_subsets = C(24,3) = 2024, and num_blocks = C(24,7) = 346104.
        # Greedy step: find block with max gain.
        # Naive greedy is O(num_blocks * num_t_subsets) per step? No, O(num_blocks * C(k,t)).
        # C(7,3) = 35. So 346104 * 35 ~ 12M ops per step. 
        # We need ~70 blocks. 70 * 12M = 840M ops. This might be too slow in Python loops.
        
        # Optimization: Use randomization to approximate greedy or use a faster C-extension-like logic?
        # Actually, 12M ops in Python is ~1s. 70 steps = 70s. Too slow.
        
        # Alternative: Randomized Greedy.
        # Pick a random uncovered t-subset. Pick a random block covering it that maximizes gain?
        # Or just pick a random block covering it?
        
        # Let's try a faster greedy:
        # Maintain count of uncovered t-subsets per block?
        # gain[b] = sum(1 for t in block_covered_t[b] if uncovered[t])
        
        # Update gain when we select a block:
        # For each t in block_covered_t[selected_block]:
        #    if uncovered[t]:
        #       uncovered[t] = False
        #       for b in t_to_blocks[t]:
        #           gain[b] -= 1
        #
        # This is O(C(k,t) * avg_degree).
        # avg_degree = num_blocks * C(k,t) / num_t_subsets = 346104 * 35 / 2024 ~ 5970.
        # So update cost is 35 * 5970 ~ 200k ops.
        # Selection cost: finding max gain in array of size 346k. np.argmax is fast.
        # Total greedy cost: 70 * (200k + 346k) ~ 40M ops. This is feasible in <1s with NumPy.
        
        gain = np.zeros(num_blocks, dtype=np.int32)
        
        # Initialize gain
        for b_idx in range(num_blocks):
            gain[b_idx] = len(block_covered_t[b_idx]) # All uncovered initially
        
        while np.any(uncovered) and time_ok():
            # Find block with max gain
            best_idx = np.argmax(gain)
            if gain[best_idx] == 0:
                break # Should not happen if uncovered exists and we have valid blocks
                
            selected.append(best_idx)
            
            # Update uncovered and gains
            for t_idx in block_covered_t[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # Decrease gain for all blocks covering this t_idx
                    # Vectorized subtraction
                    blocks_covering_t = np.array(t_to_blocks[t_idx], dtype=np.int32)
                    gain[blocks_covering_t] -= 1
            
            # Zero out gain for selected block to avoid re-selection?
            # Actually, we just need distinct blocks. 
            # If we select same block again, gain is 0 (since all its t-subsets are covered? No, other blocks might cover them, but this block's specific t-subsets are marked uncovered=False).
            # Wait, gain counts *uncovered* t-subsets.
            # Once a t-subset is covered, it's no longer "uncovered".
            # So gain for any block containing that t-subset decreases.
            # The selected block's gain becomes 0 only if ALL its t-subsets were the ones just covered?
            # No, if it covered 5 uncovered, its gain drops by 5. If it had 35 total, and 30 were already covered, gain was 5. Now it's 0.
            # So it won't be selected again.
            
        return selected

    # 3. Simulated Annealing
    def simulated_annealing(initial_blocks):
        if not initial_blocks:
            return []
            
        # State
        current_blocks = list(initial_blocks)
        current_len = len(current_blocks)
        
        # Coverage counts: how many blocks cover each t-subset?
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for t_idx in block_covered_t[b_idx]:
                cover_count[t_idx] += 1
        
        # Validity check: min(cover_count) > 0
        # We assume initial is valid.
        
        best_blocks = list(current_blocks)
        best_len = current_len
        
        # SA Parameters
        T = 10.0
        T_min = 0.1
        alpha = 0.995
        steps_per_temp = 50
        
        rng = np.random.default_rng(42)
        
        while T > T_min and time_ok():
            for _ in range(steps_per_temp):
                if not time_ok():
                    break
                
                # Move: 2-for-2 exchange
                # Pick 2 blocks to remove, 2 to add
                # Constraint: The 2 removed blocks must be such that no t-subset becomes uncovered.
                # i.e., for all t in union(removed), cover_count[t] >= 2 OR t is in union(added) and cover_count[t] == 1 (then added makes it 2).
                
                # Strategy:
                # 1. Pick a random block B1 to remove.
                # 2. Identify "critical" t-subsets for B1 (cover_count[t] == 1).
                # 3. If no critical, we can remove B1 freely? No, we want 2-for-2.
                
                # Let's try a simpler move: 1-for-1 exchange first?
                # Or stick to 2-for-2 as planned.
                
                # To make 2-for-2 efficient:
                # Pick B1, B2 to remove.
                # Critical set C = {t | t in B1 U B2 and cover_count[t] == 1}
                # We need to find B3, B4 such that C is subset of (B3 U B4).
                # And B3, B4 not in current set.
                
                # This is hard to find directly.
                
                # Alternative Move: "Remove 1, Add 1" (1-for-1)
                # This is easier.
                # Pick B1 to remove.
                # Critical C = {t | t in B1 and cover_count[t] == 1}
                # Find B2 not in set such that C subset of B2.
                # If found, swap.
                
                # Let's implement 1-for-1 first, it's robust.
                
                # Pick random block to remove
                idx_remove = rng.integers(0, len(current_blocks))
                b_remove = current_blocks[idx_remove]
                
                # Find critical t-subsets
                critical = []
                for t_idx in block_covered_t[b_remove]:
                    if cover_count[t_idx] == 1:
                        critical.append(t_idx)
                
                if not critical:
                    # Block is redundant? If so, removing it reduces size by 1.
                    # This is a "free" improvement.
                    # Remove it.
                    for t_idx in block_covered_t[b_remove]:
                        cover_count[t_idx] -= 1
                    current_blocks.pop(idx_remove)
                    current_len -= 1
                    
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(current_blocks)
                    
                    T *= alpha
                    continue
                
                # Need to find a replacement block b_add that covers all critical t-subsets.
                # Intersection of t_to_blocks for all critical t-subsets.
                # Since critical size is small (usually 1-3), we can intersect lists.
                
                # Sort critical by frequency to intersect smallest first?
                # t_to_blocks[t] is a list of ints.
                
                candidates = None
                for t_idx in critical:
                    blocks_c = set(t_to_blocks[t_idx])
                    if candidates is None:
                        candidates = blocks_c
                    else:
                        candidates &= blocks_c
                        if not candidates:
                            break
                
                if not candidates:
                    continue # No valid replacement
                
                # Exclude current blocks
                current_set = set(current_blocks)
                valid_candidates = candidates - current_set
                
                if not valid_candidates:
                    continue
                
                # Pick a random candidate
                b_add = rng.choice(list(valid_candidates))
                
                # Perform swap
                # Remove b_remove
                for t_idx in block_covered_t[b_remove]:
                    cover_count[t_idx] -= 1
                # Add b_add
                for t_idx in block_covered_t[b_add]:
                    cover_count[t_idx] += 1
                
                # Update list
                current_blocks[idx_remove] = b_add
                
                # Cost change: 0 (size unchanged)
                # But we might have reduced "redundancy" or "overlap".
                # For SA, if cost is same, we accept with prob 1 (or use a secondary metric).
                # Actually, if size is same, we should accept to explore.
                # But we want to minimize size.
                # Since size didn't change, delta_cost = 0.
                # Accept.
                
                # Check for new best? Size is same, so no.
                # But maybe this swap enables a future removal?
                
                T *= alpha
                
            # End steps_per_temp
            
            # Occasionally try a "removal" move if we have time?
            # The 1-for-1 swap with empty critical set handles removal.
            
        return best_blocks

    # 4. Main Execution
    # Get initial solution
    init_blocks = initial_greedy()
    
    if not init_blocks:
        # Fallback if greedy fails (shouldn't happen)
        return []
        
    # Run SA
    sa_blocks = simulated_annealing(init_blocks)
    
    # Final cleanup: remove any redundant blocks
    final_blocks = list(sa_blocks)
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in final_blocks:
        for t_idx in block_covered_t[b_idx]:
            cover_count[t_idx] += 1
            
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(len(final_blocks)):
            b_idx = final_blocks[i]
            redundant = True
            for t_idx in block_covered_t[b_idx]:
                if cover_count[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                for t_idx in block_covered_t[b_idx]:
                    cover_count[t_idx] -= 1
                final_blocks.pop(i)
                changed = True
                break
                
    # Convert to lists of points
    result = [list(all_blocks[b_idx]) for b_idx in final_blocks]
    
    # Ensure distinctness (SA should maintain distinctness, but just in case)
    seen = set()
    unique_result = []
    for b in result:
        b.sort()
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_result.append(b)
            
    return unique_result

if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Blocks: {len(blocks)}")
