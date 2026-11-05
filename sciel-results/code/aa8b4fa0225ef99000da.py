import itertools
import time
import numpy as np
import scipy.sparse
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)

    # 3. Build Sparse Matrix M (n_blocks x n_t)
    row_indices = []
    col_indices = []
    data = []
    
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_sub_to_idx.get(t_sub)
            if s_idx is not None:
                row_indices.append(b_idx)
                col_indices.append(s_idx)
                data.append(1)
                
    M = scipy.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t), dtype=np.int32)
    
    # Pre-compute block coverage as lists for faster access
    block_cols = []
    for b_idx in range(n_blocks):
        start = M.indptr[b_idx]
        end = M.indptr[b_idx + 1]
        block_cols.append(M.indices[start:end].copy())
    
    # 4. Greedy Construction with priority queue
    uncovered = np.ones(n_t, dtype=np.int32)
    active_blocks = []
    active_set = set()
    
    # Initialize heap with (negative_gain, block_idx)
    initial_gains = M @ uncovered
    heap = [(-int(initial_gains[b]), b) for b in range(n_blocks) if initial_gains[b] > 0]
    heapq.heapify(heap)
    
    while uncovered.sum() > 0:
        if time.time() > deadline - 20.0:
            break
            
        # Pop best block from heap
        best_b = None
        while heap:
            neg_gain, b = heapq.heappop(heap)
            if b in active_set:
                continue
            # Recompute gain for this block
            cols = block_cols[b]
            gain = int(np.sum(uncovered[cols]))
            if gain > 0:
                best_b = b
                break
            else:
                # Block no longer useful, skip
                continue
        
        if best_b is None:
            break
        
        active_blocks.append(best_b)
        active_set.add(best_b)
        
        # Update uncovered
        cols = block_cols[best_b]
        uncovered[cols] = 0
        
        # Update heap: add blocks that now have positive gain
        # This is expensive, so we'll do a simpler update
        # Instead, we'll periodically rebuild the heap
        if len(active_blocks) % 10 == 0:
            # Rebuild heap with current uncovered
            current_gains = M @ uncovered
            heap = [(-int(current_gains[b]), b) for b in range(n_blocks) 
                   if current_gains[b] > 0 and b not in active_set]
            heapq.heapify(heap)
    
    # 5. Local Search Phase: Try to reduce block count
    active_arr = np.array(active_blocks, dtype=np.int32)
    
    # Coverage count for each t-subset
    if len(active_arr) > 0:
        sub_M = M[active_arr, :]
        counts = np.asarray(sub_M.sum(axis=0)).flatten()
    else:
        counts = np.zeros(n_t, dtype=np.int32)
    
    # Local search: try removing blocks that have all their covered subsets covered by others
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        
        # Recompute counts
        if len(active_arr) > 0:
            sub_M = M[active_arr, :]
            counts = np.asarray(sub_M.sum(axis=0)).flatten()
        else:
            break
        
        # Try removing each block
        for i in range(len(active_arr)):
            if time.time() > deadline - 10.0:
                break
            
            b_idx = active_arr[i]
            cols = block_cols[b_idx]
            
            # Check if all subsets covered by this block are covered by at least one other block
            if np.all(counts[cols] > 1):
                # Safe to remove
                active_arr = np.delete(active_arr, i)
                improved = True
                break  # Restart loop with new active_arr
    
    # 6. Final redundancy removal with multiple passes
    for _ in range(3):
        if time.time() > deadline - 5.0:
            break
        if len(active_arr) == 0:
            break
            
        sub_M = M[active_arr, :]
        counts = np.asarray(sub_M.sum(axis=0)).flatten()
        
        remove_indices = []
        for i in range(len(active_arr)):
            b_idx = active_arr[i]
            cols = block_cols[b_idx]
            if np.all(counts[cols] > 1):
                remove_indices.append(i)
        
        if remove_indices:
            remove_mask = np.ones(len(active_arr), dtype=bool)
            remove_mask[remove_indices] = False
            active_arr = active_arr[remove_mask]
        else:
            break
    
    # 7. Construct Result
    result = [list(all_k_blocks[b]) for b in active_arr]
    return result
