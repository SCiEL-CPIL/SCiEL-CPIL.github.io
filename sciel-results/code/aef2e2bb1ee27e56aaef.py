import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    deadline = start_time + 110  # Safety margin well before 120s limit
    
    # Block structure: 6x6 = 3x3 blocks of 2x2
    # Outer: 3x3 matrix multiplication (Naive rank 27)
    # Inner: 2x2 matrix multiplication (Strassen rank 7)
    # Total Rank: 27 * 7 = 189
    # This is the current best known simple block decomposition for 6x6.
    
    block_size = 2
    num_blocks = 3
    
    total_rank = 189
    
    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    # Strassen coefficients for 2x2 multiplication
    # Returns 7 terms: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (row, col, coeff) for A
    # v_coeffs: list of (row, col, coeff) for B
    # w_coeffs: list of (row, col, coeff) for C
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Naive decomposition for 3x3 multiplication
    # We iterate over all 27 combinations of block indices (ib, jb, kb)
    # such that C[ib, kb] += A[ib, jb] * B[jb, kb]
    
    col = 0
    bs = block_size
    dim = n  # 6
    
    for ib in range(num_blocks):
        for jb in range(num_blocks):
            for kb in range(num_blocks):
                # For each (ib, jb, kb) triple in the outer 3x3 multiplication,
                # we need to multiply the 2x2 block A[ib, jb] by B[jb, kb]
                # and add it to C[ib, kb].
                
                # We use Strassen for this 2x2 multiplication.
                for x_list, y_list, c_list in strassen_2x2:
                    # Expand the Strassen terms into the global 6x6 indices
                    # U part: A[ib, jb] block
                    for (r, s, x_coeff) in x_list:
                        u_idx = (ib * bs + r) * dim + (jb * bs + s)
                        U[u_idx, col] += x_coeff
                    
                    # V part: B[jb, kb] block
                    for (r, s, y_coeff) in y_list:
                        v_idx = (jb * bs + r) * dim + (kb * bs + s)
                        V[v_idx, col] += y_coeff
                    
                    # W part: C[ib, kb] block
                    # Trace convention: W[k*n+i, r]
                    # Here, the result is added to C[ib, kb]
                    # The local indices in the block are (r, s)
                    for (r, s, c_coeff) in c_list:
                        w_idx = (kb * bs + s) * dim + (ib * bs + r)
                        W[w_idx, col] += c_coeff
                    
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
