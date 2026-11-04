import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication.
    Tries multiple block partitions and returns the best valid one.
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    n, m, p = DIMENSIONS
    
    def build_from_partition(block_sizes):
        """Build schoolbook decomposition respecting a given block partition."""
        all_terms = []
        num_blocks = len(block_sizes)
        
        # Compute block offsets
        offsets = [0]
        for s in block_sizes:
            offsets.append(offsets[-1] + s)
        
        def add_schoolbook_block(a_block, b_block, c_block_row, c_block_col):
            """
            Add terms for A[a_block, b_block] * B[b_block, c_block_col]
            contributing to C[c_block_row, c_block_col].
            A[a_block, b_block] is block_sizes[a_block] x block_sizes[b_block]
            B[b_block, c_block_col] is block_sizes[b_block] x block_sizes[c_block_col]
            C[c_block_row, c_block_col] is block_sizes[c_block_row] x block_sizes[c_block_col]
            """
            ar = block_sizes[a_block]
            ac = block_sizes[b_block]
            bc = block_sizes[c_block_col]
            a_row_start = offsets[a_block]
            a_col_start = offsets[b_block]
            b_row_start = offsets[b_block]
            b_col_start = offsets[c_block_col]
            c_col_start = offsets[c_block_col]
            c_row_start = offsets[c_block_row]
            
            for ii in range(ar):
                for jj in range(ac):
                    for kk in range(bc):
                        u = (a_row_start + ii) * 5 + (a_col_start + jj)
                        v = (b_row_start + jj) * 5 + (b_col_start + kk)
                        w = (c_col_start + kk) * 5 + (c_row_start + ii)
                        all_terms.append((u, v, w, 1.0))
        
        # For each output block C[i,k], sum over all j: A[i,j] * B[j,k]
        for i in range(num_blocks):
            for k in range(num_blocks):
                for j in range(num_blocks):
                    add_schoolbook_block(i, j, i, k)
        
        return all_terms
    
    def verify_and_build(all_terms):
        """Build U,V,W from terms and verify. Return (rank, U, V, W) or None if invalid."""
        rank = len(all_terms)
        if rank >= 125:
            return None  # No improvement over schoolbook
        
        U = np.zeros((25, rank), dtype=np.float64)
        V = np.zeros((25, rank), dtype=np.float64)
        W = np.zeros((25, rank), dtype=np.float64)
        
        for r, (u, v, w, coeff) in enumerate(all_terms):
            U[u, r] = coeff
            V[v, r] = coeff
            W[w, r] = coeff
        
        # Verify: for each (i,j,k), sum_r U[i,j]_r * V[j,k]_r * W[k,i]_r should be 1 if i,j,k form valid mat-mult, else 0
        # Actually the tensor T[i,j,k,l] = delta_{i,l} * delta_{j,k} for mat mult (i is row of A, j is col of A = row of B, k is col of B, l is row of C)
        # With W indexed as [k*n+i, r], we check: for all (a_row, a_col, b_row, b_col), 
        # sum_r U[a_row*5+a_col, r] * V[b_row*5+b_col, r] * W[b_col*5+a_row, r] = 1 if a_col==b_row else 0
        
        max_err = 0.0
        for a_row in range(5):
            for a_col in range(5):
                for b_row in range(5):
                    for b_col in range(5):
                        expected = 1.0 if a_col == b_row else 0.0
                        val = 0.0
                        for r in range(rank):
                            val += U[a_row*5+a_col, r] * V[b_row*5+b_col, r] * W[b_col*5+a_row, r]
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
                        if max_err > 1e-10:
                            return None
        
        return (rank, U, V, W)
    
    # Try different partitions of 5
    partitions = [
        [1, 4],
        [2, 3],
        [1, 1, 3],
        [1, 2, 2],
        [1, 1, 1, 2],
        [1, 1, 1, 1, 1],
    ]
    
    best_result = None
    best_rank = 126  # worse than schoolbook
    
    for partition in partitions:
        if time.time() > deadline - safety_margin:
            break
        
        all_terms = build_from_partition(partition)
        result = verify_and_build(all_terms)
        
        if result is not None:
            rank, U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
    
    if best_result is None:
        # Fallback to schoolbook
        rank = 125
        U = np.zeros((25, rank), dtype=np.float64)
        V = np.zeros((25, rank), dtype=np.float64)
        W = np.zeros((25, rank), dtype=np.float64)
        r = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U[i*5+j, r] = 1.0
                    V[j*5+k, r] = 1.0
                    W[k*5+i, r] = 1.0
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
