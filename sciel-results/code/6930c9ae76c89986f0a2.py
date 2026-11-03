import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _build_rank44_decomposition(n, m, p):
    """
    Builds a rank-44 decomposition for 2x3 * 3x8 matrix multiplication.
    Uses 4 blocks of rank-11 decomposition for 2x3 * 3x2.
    """
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Local indices for 2x3x2 subproblem
    # A: 2x3 -> indices 0..5 (i*m+j)
    # B: 3x2 -> indices 0..5 (j*2+k)
    # C: 2x2 -> indices 0..3 (k*2+i) -- Note: Contract W[k*n+i]
    
    # Define the 11 terms for the subproblem
    # Each term is (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # u_indices are into A (2x3)
    # v_indices are into B (3x2)
    # w_indices are into C (2x2)
    
    terms = []
    
    # Strassen terms for A[0:2, 0:2] * B[0:2, :]
    # A indices: a=0, b=1, c=3, d=4
    # B indices: e=0, f=1, g=2, h=3
    # C indices: c00=0, c01=1, c10=2, c11=3
    
    # P1 = (a+d)(e+h)
    terms.append(([0, 4], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]))
    # P2 = (c+d)e
    terms.append(([3, 4], [1, 1], [0], [1], [2, 3], [1, -1]))
    # P3 = a(f-h)
    terms.append(([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]))
    # P4 = d(g-e)
    terms.append(([4], [1], [2, 0], [1, -1], [0, 2], [1, 1]))
    # P5 = (a+b)h
    terms.append(([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]))
    # P6 = (c-a)(e+f)
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    # P7 = (b-d)(g+h)
    terms.append(([1, 4], [1, -1], [2, 3], [1, 1], [0], [1]))
    
    # Schoolbook terms for A[:, 2] * B[2, :]
    # A indices: a2=2, a3=5
    # B indices: b20=4, b21=5
    # C indices: same as above
    
    # A[0,2] * B[2,0] -> C[0,0]
    terms.append(([2], [1], [4], [1], [0], [1]))
    # A[0,2] * B[2,1] -> C[0,1]
    terms.append(([2], [1], [5], [1], [1], [1]))
    # A[1,2] * B[2,0] -> C[1,0]
    terms.append(([5], [1], [4], [1], [2], [1]))
    # A[1,2] * B[2,1] -> C[1,1]
    terms.append(([5], [1], [5], [1], [3], [1]))
    
    # Fill global matrices
    # n=2, m=3, p=8
    # U shape (6, 44)
    # V shape (24, 44)
    # W shape (16, 44)
    
    for block in range(4):
        col_offset = block * 11
        for t_idx, (u_idxs, u_vals, v_idxs, v_vals, w_idxs, w_vals) in enumerate(terms):
            col = col_offset + t_idx
            
            # Map U indices (local A) to global U
            # Local A index l = i*3 + j
            # Global U index is same: i*3 + j
            for u_idx, u_val in zip(u_idxs, u_vals):
                U[u_idx, col] = u_val
                
            # Map V indices (local B) to global V
            # Local B index l = j*2 + k (j in 0..2, k in 0..1)
            # Global B index is j*p + k = j*8 + k
            # For block 'block', global col is 2*block + k
            for v_idx, v_val in zip(v_idxs, v_vals):
                j = v_idx // 2
                k = v_idx % 2
                global_v_idx = j * p + (2 * block + k)
                V[global_v_idx, col] = v_val
                
            # Map W indices (local C) to global W
            # Local C index l = k*2 + i (k in 0..1, i in 0..1)
            # Global W index is (global_k)*n + i
            # global_k = 2*block + k
            for w_idx, w_val in zip(w_idxs, w_vals):
                k = w_idx // 2
                i = w_idx % 2
                global_w_idx = (2 * block + k) * n + i
                W[global_w_idx, col] = w_val
                
    return U, V, W

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    U, V, W = _build_rank44_decomposition(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
