import itertools
import time
import numpy as np
from scipy.sparse import csc_matrix

def construct(v, k, t):
    deadline = time.monotonic() + 118.0
    points = list(range(v))
    
    # 1. Generate all t-subsets and k-blocks
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # 2. Build sparse incidence matrix A (num_tsubs x num_kblocks)
    row_indices = []
    col_indices = []
    data = []
    
    # Precompute t-subset indices for each block
    block_tsub_idxs = []
    for j, block in enumerate(all_kblocks):
        idxs = []
        for ts in itertools.combinations(block, t):
            i = tsub_to_idx[ts]
            idxs.append(i)
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
        block_tsub_idxs.append(idxs)
        
    A = csc_matrix((data, (row_indices, col_indices)), shape=(num_tsubs, num_kblocks))
    
    # 3. Greedy Construction with multiple random restarts
    best_blocks = None
    best_count = num_kblocks
    
    # Run multiple greedy passes with different tie-breaking strategies
    for restart in range(5):
        if time.monotonic() > deadline - 30.0:
            break
            
        covered = np.zeros(num_tsubs, dtype=bool)
        selected = []
        
        # Precompute gains for initial state
        # Use numpy for speed
        remaining = np.ones(num_tsubs, dtype=bool)
        
        while time.monotonic() < deadline - 25.0:
            # Calculate gains: how many new t-subsets each block covers
            # A.T @ remaining gives count of uncovered t-subsets per block
            gains = A.T @ remaining
            
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            # Find blocks with max gain
            # Add some randomness in tie-breaking for diversity
            max_blocks = np.where(gains == max_gain)[0]
            
            if len(max_blocks) == 1:
                best_idx = max_blocks[0]
            else:
                # Random selection among ties for diversity
                best_idx = max_blocks[np.random.randint(len(max_blocks))]
                
            selected.append(int(best_idx))
            
            # Update covered
            for idx in block_tsub_idxs[best_idx]:
                covered[idx] = True
            
            if np.all(covered):
                break
                
        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = selected[:]
            
        if best_count <= 140:  # Good enough, stop early
            break
    
    if best_blocks is None:
        best_blocks = []
        best_count = num_kblocks
    
    # 4. Aggressive Polish (Remove redundant blocks)
    selected_set = set(best_blocks)
    counts = np.zeros(num_tsubs, dtype=int)
    for b_idx in selected_set:
        for idx in block_tsub_idxs[b_idx]:
            counts[idx] += 1
    
    changed = True
    while changed and time.monotonic() < deadline - 15.0:
        changed = False
        # Try removing blocks in random order for better results
        for b_idx in list(selected_set):
            can_remove = True
            for idx in block_tsub_idxs[b_idx]:
                if counts[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                selected_set.discard(b_idx)
                for idx in block_tsub_idxs[b_idx]:
                    counts[idx] -= 1
                changed = True
    
    selected_blocks = sorted(selected_set)
    
    # 5. Local Search / Swap Optimization
    # Try swapping one block for another to reduce count
    if time.monotonic() < deadline - 10.0:
        current_set = set(selected_blocks)
        counts = np.zeros(num_tsubs, dtype=int)
        for b_idx in current_set:
            for idx in block_tsub_idxs[b_idx]:
                counts[idx] += 1
        
        improved = True
        while improved and time.monotonic() < deadline - 10.0:
            improved = False
            # Try removing each block and see if we can cover the gap with one new block
            for b_idx in list(current_set):
                # Temporarily remove b_idx
                for idx in block_tsub_idxs[b_idx]:
                    counts[idx] -= 1
                
                # Find t-subsets that are now uncovered
                uncovered = np.where(counts == 0)[0]
                
                if len(uncovered) == 0:
                    # b_idx was redundant
                    current_set.discard(b_idx)
                    improved = True
                    continue
                
                # Try to find a single block that covers all uncovered t-subsets
                # This is expensive, so only try if uncovered is small
                if len(uncovered) > 20:
                    # Too many uncovered, restore and skip
                    for idx in block_tsub_idxs[b_idx]:
                        counts[idx] += 1
                    continue
                    
                # Check if any block covers all uncovered
                found = False
                for candidate in range(num_kblocks):
                    if candidate in current_set:
                        continue
                    # Check if candidate covers all uncovered
                    covers_all = True
                    for u_idx in uncovered:
                        if A[u_idx, candidate] == 0:
                            covers_all = False
                            break
                    if covers_all:
                        # Swap: remove b_idx, add candidate
                        current_set.discard(b_idx)
                        current_set.add(candidate)
                        for idx in block_tsub_idxs[candidate]:
                            counts[idx] += 1
                        improved = True
                        found = True
                        break
                
                if not found:
                    # Restore b_idx
                    for idx in block_tsub_idxs[b_idx]:
                        counts[idx] += 1
    
    selected_blocks = sorted(current_set)
    
    # 6. Final Polish
    selected_set = set(selected_blocks)
    counts = np.zeros(num_tsubs, dtype=int)
    for b_idx in selected_set:
        for idx in block_tsub_idxs[b_idx]:
            counts[idx] += 1
    
    changed = True
    while changed and time.monotonic() < deadline - 5.0:
        changed = False
        for b_idx in list(selected_set):
            can_remove = True
            for idx in block_tsub_idxs[b_idx]:
                if counts[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                selected_set.discard(b_idx)
                for idx in block_tsub_idxs[b_idx]:
                    counts[idx] -= 1
                changed = True
    
    selected_blocks = sorted(selected_set)
    
    # Convert indices back to lists
    result_blocks = [list(all_kblocks[i]) for i in selected_blocks]
    
    return result_blocks
