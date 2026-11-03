import itertools
import numpy as np
import time
import random
import scipy.sparse as sp

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Build sparse matrix M
    # M[i, j] = 1 if block i covers t-subset j
    rows = []
    cols = []
    
    for i, b in enumerate(blocks):
        for sub in itertools.combinations(b, t):
            j = t_sub_to_idx[sub]
            rows.append(i)
            cols.append(j)
            
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n_blocks, n_t_subs))
    
    def run_greedy(randomize_ties=False):
        uncovered = np.ones(n_t_subs, dtype=np.uint8)
        total_uncovered = n_t_subs
        used_blocks = set()
        selected_blocks = []
        
        while total_uncovered > 0:
            if time.time() > deadline:
                return None
            
            # Calculate gains for all blocks
            # M is CSR, uncovered is 1D array
            # Result is 1D array of shape (n_blocks,)
            gains = np.asarray(M @ uncovered).flatten()
            
            # Mask used blocks
            # We can create a mask array, but it's expensive to create every time.
            # Instead, we can just check used_blocks after finding max?
            # Or, we can set gains[used_blocks] = -1.
            # But used_blocks is a set. We need to convert to array indices.
            
            # Optimization: Keep a boolean array `is_used`
            # But we need to pass it or make it global to run_greedy?
            # Let's use a local boolean array.
            
            # Actually, M @ uncovered is the expensive part?
            # Sparse matrix vector product is $O(nnz)$.
            # nnz = n_blocks * 56 = 75582 * 56 = 4.2M.
            # This is very fast.
            
            # Now find max gain among unused blocks.
            # We can iterate through indices? No, that's 75k.
            # We can use np.argmax, but we need to ignore used blocks.
            
            # Simple approach:
            # 1. Get all gains.
            # 2. Set gains for used blocks to -1.
            # 3. Argmax.
            
            # To set gains for used blocks, we need the indices.
            # We can maintain a list of used indices.
            
            if used_blocks:
                used_indices = list(used_blocks)
                gains[used_indices] = -1
                
            best_block_idx = np.argmax(gains)
            max_gain = gains[best_block_idx]
            
            if max_gain <= 0:
                break
                
            selected_blocks.append(list(blocks[best_block_idx]))
            used_blocks.add(best_block_idx)
            
            # Update uncovered
            # Get the row for best_block_idx
            # M[best_block_idx] is a sparse matrix row.
            # We need to find which columns are 1.
            # Or just use the precomputed masks?
            # We didn't precompute masks in this version.
            # Let's precompute the column indices for each block for updating.
            
            # Actually, we can just do:
            # row = M.getrow(best_block_idx)
            # cols = row.indices
            # But this is slow?
            
            # Better: Precompute a list of column indices for each block.
            # block_cols[i] = list of j such that M[i, j] == 1
            
            # Let's add that to precomputation.
            
            # Update uncovered
            cols = block_cols[best_block_idx]
            new_covered_count = np.sum(uncovered[cols])
            if new_covered_count > 0:
                uncovered[cols] = 0
                total_uncovered -= new_covered_count
                
        return selected_blocks

    # Precompute block_cols
    block_cols = []
    for i in range(n_blocks):
        # Extract indices from M row i
        # Since M is built from rows/cols lists, we can just store them.
        # But we built M from a flat list.
        # Let's re-structure precomputation.
        
        pass

    # Re-do precomputation to store block_cols
    block_cols = []
    rows = []
    cols = []
    for i, b in enumerate(blocks):
        c_list = []
        for sub in itertools.combinations(b, t):
            j = t_sub_to_idx[sub]
            c_list.append(j)
        block_cols.append(np.array(c_list, dtype=np.int32))
        rows.extend([i] * len(c_list))
        cols.extend(c_list)
        
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n_blocks, n_t_subs))
    
    best_solution = None
    
    sol1 = run_greedy(randomize_ties=False)
    if sol1:
        best_solution = sol1
        
    if time.time() < deadline - 10:
        sol2 = run_greedy(randomize_ties=True)
        if sol2 and (not best_solution or len(sol2) < len(best_solution)):
            best_solution = sol2

    if best_solution is None:
        return []
        
    return best_solution
