import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(k_subsets)
    
    # Map t-subsets to blocks containing them
    t_to_blocks = [[] for _ in range(num_t)]
    
    # Map blocks to t-subsets they contain
    block_t_indices = []
    
    for i, block in enumerate(k_subsets):
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_to_idx[t_sub]
            indices.append(idx)
            t_to_blocks[idx].append(i)
        block_t_indices.append(indices)
        
    # Build sparse matrix M for greedy
    # M[i, j] = 1 if block i covers t_subset j
    rows = []
    cols = []
    data = []
    for i, indices in enumerate(block_t_indices):
        for j in indices:
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    M = csr_matrix((data, (rows, cols)), shape=(num_k, num_t))
    
    coverage = np.zeros(num_t, dtype=np.int32)
    uncovered_count = num_t
    solution_blocks = []
    
    # 2. Greedy Phase
    while uncovered_count > 0:
        if time.time() > deadline - 20: # Leave time for local search
            break
            
        # Calculate gains
        uncovered_mask = (coverage == 0).astype(np.int32)
        gains = M.dot(uncovered_mask)
        
        # Find max gain
        best_block_idx = np.argmax(gains)
        max_gain = gains[best_block_idx]
        
        if max_gain == 0:
            break
            
        solution_blocks.append(best_block_idx)
        
        # Update coverage
        indices = block_t_indices[best_block_idx]
        for idx in indices:
            if coverage[idx] == 0:
                uncovered_count -= 1
            coverage[idx] += 1
            
    # 3. Local Search Phase
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Pass 1: Remove redundant blocks
        # Iterate in reverse to handle indices safely
        for i in range(len(solution_blocks) - 1, -1, -1):
            block_idx = solution_blocks[i]
            # Check if any t-subset is covered ONLY by this block
            is_redundant = True
            for idx in block_t_indices[block_idx]:
                if coverage[idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                for idx in block_t_indices[block_idx]:
                    coverage[idx] -= 1
                solution_blocks.pop(i)
                improved = True
                
        if time.time() > deadline - 5:
            break
            
        # Pass 2: 2-Exchange (Replace 2 blocks with 1)
        n_sol = len(solution_blocks)
        # To speed up, we might not check all pairs if n_sol is large
        # But n_sol is ~120, so 7000 pairs is okay if inner loop is fast
        
        for i in range(n_sol):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n_sol):
                block_i = solution_blocks[i]
                block_j = solution_blocks[j]
                
                # Temporarily remove i and j
                for idx in block_t_indices[block_i]:
                    coverage[idx] -= 1
                for idx in block_t_indices[block_j]:
                    coverage[idx] -= 1
                    
                # Find uncovered t-subsets
                U_indices = [idx for idx in range(num_t) if coverage[idx] == 0]
                
                if not U_indices:
                    # They were redundant together? Should have been caught in Pass 1?
                    # Or one was redundant given the other?
                    # If U is empty, we can just remove them? No, we need to add a block?
                    # If U is empty, it means all t-subsets are covered by OTHER blocks.
                    # So we can remove both i and j.
                    # But we are in 2-exchange. If we can remove both, that's a -2 improvement.
                    # Let's just remove them.
                    solution_blocks.pop(j)
                    solution_blocks.pop(i)
                    improved = True
                    continue
                else:
                    # Check if a single block can cover U_indices
                    # Pick a t-subset from U
                    s0 = U_indices[0]
                    candidates = t_to_blocks[s0]
                    
                    found_replacement = False
                    for k in candidates:
                        if k == block_i or k == block_j:
                            continue
                        # Check if k covers all in U_indices
                        # Optimization: if k is in solution, we can't add it again? 
                        # Blocks must be distinct.
                        if k in solution_blocks:
                            continue
                            
                        valid = True
                        for idx in U_indices:
                            if idx not in block_t_indices[k]:
                                valid = False
                                break
                                
                        if valid:
                            # Replace i, j with k
                            # Restore coverage for i, j
                            for idx in block_t_indices[block_i]:
                                coverage[idx] += 1
                            for idx in block_t_indices[block_j]:
                                coverage[idx] += 1
                            
                            # Remove i, j
                            # Find indices in solution_blocks
                            idx_i = solution_blocks.index(block_i)
                            idx_j = solution_blocks.index(block_j)
                            if idx_i > idx_j:
                                idx_i, idx_j = idx_j, idx_i
                            solution_blocks.pop(idx_j)
                            solution_blocks.pop(idx_i)
                            
                            # Add k
                            solution_blocks.append(k)
                            for idx in block_t_indices[k]:
                                coverage[idx] += 1
                                
                            improved = True
                            found_replacement = True
                            break
                            
                    if not found_replacement:
                        # Restore coverage
                        for idx in block_t_indices[block_i]:
                            coverage[idx] += 1
                        for idx in block_t_indices[block_j]:
                            coverage[idx] += 1

    # Convert to list of lists
    final_blocks = [list(k_subsets[i]) for i in solution_blocks]
    return final_blocks
