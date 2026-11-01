# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    # Deadline management: leave 15 seconds for final repair and reduction
    deadline = time.time() + 135.0
    
    # 1. Precomputation
    tsubs = list(itertools.combinations(range(v), t))
    n_tsub = len(tsubs)
    tsub_to_bit = {sub: i for i, sub in enumerate(tsubs)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    # Build block_masks as Python ints (bitmasks)
    block_masks = []
    for comb in all_k_subsets:
        mask = 0
        for sub in itertools.combinations(comb, t):
            mask |= (1 << tsub_to_bit[sub])
        block_masks.append(mask)
    
    # Also store as numpy uint64 array for vectorized gain computation
    # Split each mask into chunks of 64 bits
    n_chunks = (n_tsub + 63) // 64
    block_masks_np = np.zeros((n_ksub, n_chunks), dtype=np.uint64)
    for i, mask in enumerate(block_masks):
        for c in range(n_chunks):
            block_masks_np[i, c] = (mask >> (64 * c)) & 0xFFFFFFFFFFFFFFFF
    
    # Full mask
    full_mask = (1 << n_tsub) - 1
    uncovered = full_mask
    uncovered_np = np.zeros(n_chunks, dtype=np.uint64)
    for c in range(n_chunks):
        uncovered_np[c] = (uncovered >> (64 * c)) & 0xFFFFFFFFFFFFFFFF
    
    # State
    cover_count = [0] * n_tsub
    blocks_selected = []
    selected_set = set()
    
    # Block lookup for repair
    block_to_idx = {comb: i for i, comb in enumerate(all_k_subsets)}
    
    # 2. Greedy Loop with NumPy Vectorized Gain Computation
    while uncovered and time.time() < deadline - 15.0:
        # Vectorized gain: for each block, count uncovered t-subsets it covers
        # gain_i = sum over chunks of popcount(block_masks_np[i,c] & uncovered_np[c])
        # Use numpy to compute this efficiently
        masked = block_masks_np & uncovered_np[np.newaxis, :]
        # Count set bits per row per chunk, then sum
        # numpy doesn't have direct popcount, so we use a lookup or bit trick
        # For speed, use a precomputed popcount table for bytes
        # Actually, let's use a different approach: since we're in Python,
        # let's use the int approach but with better optimization
        
        # Alternative: use the fact that we can compute gains incrementally
        # But for now, let's optimize the Python loop
        best_gain = 0
        best_idx = -1
        uncv = uncovered
        b_masks = block_masks
        
        for i in range(n_ksub):
            g = (b_masks[i] & uncv).bit_count()
            if g > best_gain:
                best_gain = g
                best_idx = i
        
        if best_gain == 0:
            break
        
        # Add the best block
        blocks_selected.append(best_idx)
        selected_set.add(best_idx)
        
        # Update cover_count and uncovered
        mask = block_masks[best_idx]
        m = mask
        while m:
            lsb = m & (-m)
            bit_pos = lsb.bit_length() - 1
            cover_count[bit_pos] += 1
            m ^= lsb
        
        uncovered &= ~mask
    
    # 3. Repair Phase
    if uncovered and time.time() < deadline - 10.0:
        m = uncovered
        while m:
            if time.time() > deadline - 5.0:
                break
            lsb = m & (-m)
            bit_pos = lsb.bit_length() - 1
            m ^= lsb
            
            tsub = tsubs[bit_pos]
            have = set(tsub)
            need = k - len(have)
            
            remaining_points = [p for p in range(v) if p not in have]
            
            if need > 0:
                extra = random.sample(remaining_points, need)
            else:
                extra = []
            
            block = sorted(list(have) + extra)
            block_tuple = tuple(block)
            
            if block_tuple in selected_set:
                continue
            
            b_idx = block_to_idx.get(block_tuple)
            if b_idx is not None and b_idx not in selected_set:
                blocks_selected.append(b_idx)
                selected_set.add(b_idx)
                
                mask = block_masks[b_idx]
                m2 = mask
                while m2:
                    lsb2 = m2 & (-m2)
                    bit_pos2 = lsb2.bit_length() - 1
                    cover_count[bit_pos2] += 1
                    m2 ^= lsb2
                
                uncovered &= ~mask
    
    # 4. Reduction Phase - Remove redundant blocks
    # Process in order of least "critical" blocks first
    # A block is redundant if all its t-subsets have cover_count > 1
    changed = True
    while changed and time.time() < deadline - 3.0:
        changed = False
        for i in range(len(blocks_selected) - 1, -1, -1):
            if time.time() > deadline - 3.0:
                break
            b_idx = blocks_selected[i]
            
            # Check if redundant
            mask = block_masks[b_idx]
            m = mask
            redundant = True
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                m ^= lsb
                if cover_count[bit_pos] <= 1:
                    redundant = False
                    break
            
            if redundant:
                mask = block_masks[b_idx]
                m = mask
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    m ^= lsb
                    cover_count[bit_pos] -= 1
                
                blocks_selected.pop(i)
                selected_set.remove(b_idx)
                changed = True
                break
    
    # 5. Final Conversion
    result = []
    for idx in blocks_selected:
        result.append(list(all_k_subsets[idx]))
        
    return result

# EVOLVE-BLOCK-END
