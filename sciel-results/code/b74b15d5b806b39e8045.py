import itertools
import numpy as np
from math import comb
import time
import scipy.sparse as sp

def construct(v, k, t):
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access (dense for updates)
    # Shape: (n_blocks, k_t)
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    
    # Build sparse matrix A: (n_triples, n_blocks)
    # A[i, j] = 1 if triple i is in block j
    row_indices = []
    col_indices = []
    
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            idx = triple_to_idx[sub]
            block_triple_indices[i, j] = idx
            row_indices.append(idx)
            col_indices.append(i)
            
    A = sp.csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(n_triples, n_blocks))
    A = A.tocsr() # Ensure CSR format for fast row slicing/summing
    
    def greedy_with_prune(random_seed):
        rng = np.random.RandomState(random_seed)
        
        # coverage_count[i] is the number of selected blocks covering triple i
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        selected_indices = []
        
        # Track uncovered triples for efficiency? 
        # A[uncovered_indices, :] is efficient if we just pass the mask or indices.
        # Initially all are uncovered.
        
        while True:
            # Find uncovered triples
            uncovered_mask = (coverage_count == 0)
            if not np.any(uncovered_mask):
                break
                
            uncovered_indices = np.where(uncovered_mask)[0]
            
            # Calculate gains: number of uncovered triples in each block
            # A[uncovered_indices, :] is a sparse matrix of shape (len(uncovered_indices), n_blocks)
            # Summing along axis 0 gives a vector of length n_blocks
            # This is O(nnz) in the submatrix
            gains = A[uncovered_indices, :].sum(axis=0)
            
            # gains is a sparse matrix (1, n_blocks) or dense?
            # .sum(axis=0) on CSR usually returns a 1xN sparse matrix or 1D array if dense.
            # Let's ensure it's a 1D array for np.max
            gains_arr = np.asarray(gains.todense()).flatten()
            
            max_gain = np.max(gains_arr)
            if max_gain == 0:
                break
                
            candidates = np.where(gains_arr == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            selected_indices.append(best_idx)
            block_triples = block_triple_indices[best_idx]
            coverage_count[block_triples] += 1
            
        # Pruning Phase
        active_blocks = list(selected_indices)
        
        changed = True
        passes = 0
        max_passes = 20 # Increase passes as it's cheap
        
        while changed and passes < max_passes:
            changed = False
            passes += 1
            # Shuffle active blocks to randomize removal order
            rng.shuffle(active_blocks)
            
            new_active_blocks = []
            for b_idx in active_blocks:
                block_triples = block_triple_indices[b_idx]
                
                # Check if any triple in this block would become uncovered if we remove it
                is_redundant = not np.any(coverage_count[block_triples] == 1)
                
                if is_redundant:
                    coverage_count[block_triples] -= 1
                    changed = True
                else:
                    new_active_blocks.append(b_idx)
            
            active_blocks = new_active_blocks
            
        # Convert indices to blocks
        return [list(all_blocks[i]) for i in active_blocks]

    # Run multiple times with different seeds until deadline
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = 110.0 # Safety margin from 120s limit
    
    seed = 0
    while time.time() < deadline:
        blocks = greedy_with_prune(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        seed += 1
        
    return best_blocks
