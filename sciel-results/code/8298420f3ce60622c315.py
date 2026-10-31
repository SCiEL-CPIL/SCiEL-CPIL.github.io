import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Generate all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    n_kblocks = len(all_kblocks)
    
    # Precompute mask for each k-block: bitmask of t-subsets it covers
    # Since n_tsubs = 3876, we can use Python ints for bitmasks.
    # However, for speed in intersection/union, NumPy boolean arrays might be better for the "count" logic.
    # Let's use a list of Python int bitmasks for each block.
    block_masks = [0] * n_kblocks
    for i, b in enumerate(all_kblocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            mask |= (1 << tsub_index[ts])
        block_masks[i] = mask
        
    # Map from bitmask of required t-subsets to list of block indices that cover them?
    # Too large.
    
    # Instead, for pair replacement, we need to find a block that covers a specific set of t-subsets.
    # Let's precompute for each subset of points (size <= k) the list of blocks containing it?
    # No, the requirement is on t-subsets.
    
    # Greedy Construction
    def greedy_construct(num_restarts=5):
        best_blocks = None
        best_count = float('inf')
        
        for _ in range(num_restarts):
            if time.time() > deadline - 20:
                break
                
            uncovered = (1 << n_tsubs) - 1
            blocks = []
            block_indices = []
            
            # Randomize the order of t-subsets to break symmetry
            tsub_order = list(range(n_tsubs))
            random.shuffle(tsub_order)
            
            # We pick blocks greedily.
            # To speed up, we can maintain a list of uncovered t-subsets.
            # But checking gain for all 50k blocks is slow.
            
            # Optimization: 
            # 1. Pick a random uncovered t-subset.
            # 2. Only consider blocks that contain this t-subset.
            # 3. Among those, pick the one with highest gain.
            
            # Precompute: for each t-subset, which blocks contain it?
            # tsub_to_blocks = {ts_idx: [block_idx, ...]}
            tsub_to_blocks = [[] for _ in range(n_tsubs)]
            for i, b in enumerate(all_kblocks):
                for ts in itertools.combinations(b, t):
                    tsub_to_blocks[tsub_index[ts]].append(i)
            
            while uncovered:
                if time.time() > deadline - 10:
                    break
                    
                # Find an uncovered t-subset
                # We can just pick the first one in the shuffled order that is still uncovered
                found = False
                for idx in tsub_order:
                    if (uncovered >> idx) & 1:
                        found = True
                        break
                if not found:
                    break
                
                # Get candidate blocks that cover this t-subset
                candidates = tsub_to_blocks[idx]
                
                best_block_idx = -1
                best_gain = -1
                
                # Calculate gain for each candidate
                # Gain = number of new t-subsets covered
                # new_mask = block_mask & uncovered
                # gain = popcount(new_mask)
                
                for b_idx in candidates:
                    bm = block_masks[b_idx]
                    new_mask = bm & uncovered
                    gain = bin(new_mask).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
                        
                if best_block_idx == -1:
                    # Should not happen
                    break
                    
                blocks.append(all_kblocks[best_block_idx])
                block_indices.append(best_block_idx)
                uncovered &= ~block_masks[best_block_idx]
                
            # Cleanup: remove redundant blocks
            blocks = remove_redundant(blocks, block_indices, block_masks)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                if time.time() > deadline - 5:
                    break
                    
        return best_blocks

    def remove_redundant(blocks, block_indices, block_masks):
        # Maintain cover counts
        # cover_count[s] = number of blocks covering t-subset s
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        
        for b_idx in block_indices:
            bm = block_masks[b_idx]
            # Convert bitmask to indices? 
            # It's faster to just iterate through the t-subsets in the block?
            # No, we have the bitmask.
            # We can extract set bits.
            m = bm
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
        
        # Remove blocks that have no unique t-subsets
        new_blocks = []
        new_indices = []
        for b_idx in block_indices:
            bm = block_masks[b_idx]
            # Check if any t-subset in bm has cover_count == 1
            m = bm
            unique = False
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    unique = True
                    break
                m ^= lsb
            if unique:
                new_blocks.append(blocks[blocks.index(all_kblocks[b_idx])]) # This is slow, fix below
                new_indices.append(b_idx)
            else:
                # Update cover counts
                m = bm
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                    
        # The above loop is O(N * K * T) and uses .index which is O(N).
        # Let's rewrite remove_redundant more carefully.
        
        # Actually, the previous implementation was flawed. Let's do a proper pass.
        pass # Placeholder

    # Let's implement a cleaner local search on the set of block indices.
    
    def optimize(blocks, block_indices):
        # blocks: list of tuples
        # block_indices: list of indices into all_kblocks
        # We will work with a set of indices to allow fast removal.
        # But we need to preserve order? No, order doesn't matter.
        
        current_indices = set(block_indices)
        
        # Calculate cover counts
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in current_indices:
            bm = block_masks[b_idx]
            m = bm
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Remove redundant blocks
            for b_idx in list(current_indices):
                bm = block_masks[b_idx]
                m = bm
                unique = False
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        unique = True
                        break
                    m ^= lsb
                if not unique:
                    # Remove b_idx
                    current_indices.remove(b_idx)
                    m = bm
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    improved = True
                    
            if not current_indices:
                break
                
            # 2. Try pair replacement
            # For each pair in current_indices
            # This is O(N^2) which is ~150^2 = 22500 pairs.
            # For each pair, check if a single block can replace them.
            
            indices_list = list(current_indices)
            for i in range(len(indices_list)):
                if time.time() > deadline - 2:
                    break
                b1 = indices_list[i]
                for j in range(i + 1, len(indices_list)):
                    b2 = indices_list[j]
                    
                    # Calculate unique subsets of b1 and b2
                    # Unique to b1: s in mask(b1) such that cover_count[s] == 1 (and s not in b2? No, cover_count is global)
                    # Wait, cover_count includes b1 and b2.
                    # We want subsets that are covered ONLY by b1 or ONLY by b2 (among all blocks).
                    # If a subset is covered by b1 and b2, its count is at least 2.
                    # If we remove b1 and b2, we lose coverage for subsets that were covered ONLY by b1 or ONLY by b2.
                    # Let U = { s | s in mask(b1) xor mask(b2) and cover_count[s] == 1 }?
                    # No.
                    # Let M1 = mask(b1), M2 = mask(b2).
                    # Subsets covered by b1 but not b2: M1 & ~M2.
                    # Subsets covered by b2 but not b1: M2 & ~M1.
                    # Subsets covered by both: M1 & M2.
                    # If we remove b1 and b2, we lose:
                    # 1. Subsets in M1 & ~M2 where cover_count == 1 (only b1 covers it).
                    # 2. Subsets in M2 & ~M1 where cover_count == 1 (only b2 covers it).
                    # 3. Subsets in M1 & M2 where cover_count == 2 (only b1 and b2 cover it).
                    
                    # So Required = ( (M1 & ~M2) & UniqueMask1 ) | ( (M2 & ~M1) & UniqueMask2 ) | ( (M1 & M2) & DoubleCoveredMask )
                    # Where UniqueMask1 has bits for s where cover_count[s]==1 and s in M1.
                    
                    # Simpler:
                    # Let's just iterate through the bits in M1 and M2.
                    # This might be slow if done in Python loops for 22500 pairs.
                    
                    # Optimization:
                    # Precompute for each block the list of t-subsets it covers?
                    # We have the bitmask.
                    
                    # Let's check the union of points in the required subsets.
                    # If the union of points in the required subsets has size > 7, skip.
                    
                    # To get the required subsets efficiently:
                    # We can precompute for each block a list of t-subset indices it covers.
                    # block_tsubs = {b_idx: [ts_idx, ...]}
                    
                    # Let's assume we have block_tsubs.
                    
                    req_subsets = set()
                    m1 = block_masks[b1]
                    m2 = block_masks[b2]
                    
                    # Subsets only in b1
                    m_only1 = m1 & ~m2
                    m = m_only1
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            req_subsets.add(idx)
                        m ^= lsb
                        
                    # Subsets only in b2
                    m_only2 = m2 & ~m1
                    m = m_only2
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            req_subsets.add(idx)
                        m ^= lsb
                        
                    # Subsets in both
                    m_both = m1 & m2
                    m = m_both
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 2:
                            req_subsets.add(idx)
                        m ^= lsb
                        
                    if not req_subsets:
                        # Both are redundant? Should have been caught by step 1.
                        # If req_subsets is empty, we can remove both.
                        current_indices.remove(b1)
                        current_indices.remove(b2)
                        # Update cover counts
                        m = m1 | m2
                        while m:
                            lsb = m & (-m)
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        improved = True
                        break # Restart pair loop as indices changed
                    
                    # Check if a single block can cover req_subsets
                    # The block must contain all points in all req_subsets.
                    # Union of points in req_subsets.
                    union_points = set()
                    for idx in req_subsets:
                        union_points.update(all_tsubs[idx])
                        
                    if len(union_points) > k:
                        continue
                        
                    # Find a block that contains union_points
                    # We can precompute: for each subset of points (size <= k), which blocks contain it?
                    # Or just check candidates.
                    # Candidates: blocks that contain union_points.
                    
                    # Let's use a heuristic: check blocks that contain the first few points of union_points.
                    # Or simply iterate all blocks? 50k blocks * 22k pairs is too much.
                    
                    # Better: Precompute `point_to_blocks`? No.
                    # Precompute `subset_to_blocks` for subsets of size up to 4?
                    # If union_points has size 7, we just need to check if any block IS exactly union_points?
                    # No, the block just needs to CONTAIN union_points.
                    # If union_points has size 7, the block must be exactly union_points.
                    # If union_points has size < 7, there are multiple candidates.
                    
                    # Let's create a lookup: `tuple(sorted(points)) -> list of block indices containing these points`
                    # This is too big for all subsets.
                    
                    # Alternative:
                    # If len(union_points) == 7:
                    #    Check if tuple(sorted(union_points)) is in all_kblocks.
                    # Else:
                    #    Pick a point p in union_points.
                    #    Candidates are blocks containing p and the other points.
                    
                    # Let's just try to find if any block covers req_subsets.
                    # We can precompute for each block its mask.
                    # We need a block B such that (mask(B) & req_mask) == req_mask.
                    # Where req_mask is the bitmask of req_subsets.
                    
                    req_mask = 0
                    for idx in req_subsets:
                        req_mask |= (1 << idx)
                        
                    # Search for a block
                    # To speed this up, we can precompute a dictionary:
                    # key: bitmask of t-subsets? No, too many.
                    
                    # Let's just iterate through a few random blocks or all blocks if time permits.
                    # Given the tight loop, we should be careful.
                    
                    # Optimization:
                    # If len(union_points) == 7, only 1 candidate.
                    # If len(union_points) == 6, C(19-6, 1) = 13 candidates.
                    # If len(union_points) == 5, C(19-5, 2) = 91 candidates.
                    # If len(union_points) == 4, C(19-4, 3) = 364 candidates.
                    
                    # We can generate candidates efficiently.
                    candidates = []
                    if len(union_points) == k:
                        cand_tuple = tuple(sorted(union_points))
                        if cand_tuple in tsub_index: # No, tsub_index is for t-subsets.
                             # Check if cand_tuple is a valid block
                             # We can map block tuple to index
                             pass
                    else:
                         # Generate combinations
                         pass
                         
                    # This is getting complex. Let's simplify.
                    # If we find a replacement, we add it.
                    
                    # For now, let's just do redundancy removal and simple restarts.
                    # The pair replacement is the key to improvement but hard to optimize in Python quickly.
                    
                    # Let's try a different structural change:
                    # Use a genetic algorithm?
                    # Or just more aggressive restarts with different greedy strategies.
                    
                    pass # End of pair logic placeholder
                    
            if improved:
                continue # Restart the while loop as cover_count changed
                
        return list(current_indices)
        
    # The above logic is too slow/complex for the time limit.
    # Let's stick to:
    # 1. Fast Greedy (vectorized)
    # 2. Redundancy Removal
    # 3. Simple Local Search: Try to replace a block with another that covers all its unique subsets.
    
    def fast_greedy():
        # Use NumPy to calculate gains?
        # Gains depend on uncovered set.
        # Uncovered is a bitmask.
        # We can't easily vectorize "popcount(block_mask & uncovered)" for all blocks at once 
        # without converting to arrays, which is slow for 50k blocks.
        
        # Instead, use the "pick random uncovered t-subset, filter candidates" approach.
        
        uncovered = (1 << n_tsubs) - 1
        blocks = []
        block_indices = []
        
        # Precompute tsub_to_blocks
        tsub_to_blocks = [[] for _ in range(n_tsubs)]
        for i, b in enumerate(all_kblocks):
            for ts in itertools.combinations(b, t):
                tsub_to_blocks[tsub_index[ts]].append(i)
                
        # Randomize t-subset order
        tsub_order = list(range(n_tsubs))
        random.shuffle(tsub_order)
        
        while uncovered:
            if time.time() > deadline - 5:
                break
            found = False
            for idx in tsub_order:
                if (uncovered >> idx) & 1:
                    found = True
                    break
            if not found:
                break
                
            candidates = tsub_to_blocks[idx]
            best_gain = -1
            best_idx = -1
            
            for b_idx in candidates:
                bm = block_masks[b_idx]
                gain = bin(bm & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    
            if best_idx != -1:
                blocks.append(all_kblocks[best_idx])
                block_indices.append(best_idx)
                uncovered &= ~block_masks[best_idx]
                
        return blocks, block_indices

    def cleanup(blocks, block_indices):
        # Remove redundant
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            bm = block_masks[b_idx]
            m = bm
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        new_indices = []
        for b_idx in block_indices:
            bm = block_masks[b_idx]
            m = bm
            unique = False
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    unique = True
                    break
                m ^= lsb
            if unique:
                new_indices.append(b_idx)
            else:
                m = bm
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
        return new_indices

    # Main Loop
    best_count = float('inf')
    best_indices = []
    
    while time.time() < deadline - 10:
        blocks, block_indices = fast_greedy()
        if not blocks:
            continue
            
        block_indices = cleanup(blocks, block_indices)
        
        if len(block_indices) < best_count:
            best_count = len(block_indices)
            best_indices = block_indices
            
        # Local Search: Try to remove more
        # (Already done in cleanup, but maybe we can do it again after some swaps?)
        # Let's just restart for now.
        
    if not best_indices:
        # Fallback: return a valid covering even if not optimal
        blocks, block_indices = fast_greedy()
        return [list(b) for b in blocks]
        
    return [list(all_kblocks[i]) for i in best_indices]

if __name__ == "__main__":
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
