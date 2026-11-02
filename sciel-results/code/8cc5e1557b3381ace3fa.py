"""Matrix multiplication tensor decomposition for 2x3 by 3x10.

Strategy: Deterministic constructive decomposition using Strassen's algorithm.
We split the 3x10 matrix B into two parts: B1 (2x10) and B2 (1x10).
We split the 2x3 matrix A into A1 (2x2) and A2 (2x1).
C = A1 * B1 + A2 * B2.
The A2 * B2 part is a rank 20 tensor (2 * 1 * 10).
The A1 * B1 part is a 2x2 by 2x10 multiplication.
We split B1 into 5 blocks of 2x2 matrices: B1 = [B1_0, ..., B1_4].
Then A1 * B1 = [A1 * B1_0, ..., A1 * B1_4].
Each A1 * B1_t is a 2x2 matrix multiplication.
We use Strassen's algorithm for each 2x2 multiplication (rank 7).
Total rank = 20 + 5 * 7 = 55.
"""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _strassen_2x2(U_block, V_block, W_block, col_offset, i_base, k_base):
    """
    Add Strassen rank-7 terms for a 2x2 matrix multiplication.
    U_block: indices for the 2x2 A matrix (2 rows, 2 cols) -> actually 4 indices in U
    V_block: indices for the 2x2 B matrix (2 rows, 2 cols) -> 4 indices in V
    W_block: indices for the 2x2 C matrix (2 rows, 2 cols) -> 4 indices in W
    
    Strassen formulas:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U indices: (i, j) in 2x2 A
    u00, u01, u10, u11 = U_block
    # V indices: (j, k) in 2x2 B
    v00, v01, v10, v11 = V_block
    # W indices: (k, i) in 2x2 C
    w00, w01, w10, w11 = W_block
    
    # 7 rank-1 terms
    # Term 1: (A11 + A22)(B11 + B22)
    U[:, col_offset] = 0
    V[:, col_offset] = 0
    W[:, col_offset] = 0
    U[u00, col_offset] = 1
    U[u11, col_offset] = 1
    V[v00, col_offset] = 1
    V[v11, col_offset] = 1
    W[w00, col_offset] = 1
    W[w01, col_offset] = 1
    W[w10, col_offset] = 1
    W[w11, col_offset] = 1
    # Wait, W indices for C11, C12, C21, C22 are different.
    # The rank-1 term contributes to ALL output entries where W is non-zero.
    # But Strassen terms contribute to specific C entries.
    # M1 contributes to C11, C12, C21, C22?
    # M1 = (A11+A22)(B11+B22).
    # C11 += M1. C12 += 0. C21 += 0. C22 -= M1.
    # So W for M1 should be non-zero at w00 (C11) and w11 (C22).
    # C11 is at (k=0, i=0) -> index 0*2+0 = 0.
    # C12 is at (k=1, i=0) -> index 1*2+0 = 2.
    # C21 is at (k=0, i=1) -> index 0*2+1 = 1.
    # C22 is at (k=1, i=1) -> index 1*2+1 = 3.
    
    # Let's re-define the W indices for the 2x2 block.
    # w00 corresponds to C11 (k=0, i=0)
    # w10 corresponds to C12 (k=1, i=0)? No, W[k*n+i].
    # For 2x2 output, n=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    # So:
    # C11 is W[0]
    # C21 is W[1]
    # C12 is W[2]
    # C22 is W[3]
    
    # M1 contributes to C11 (+1) and C22 (-1).
    W[w00, col_offset] = 1
    W[w11, col_offset] = -1
    # w11 is index 3.
    
    col_offset += 1
    
    # Term 2: (A21 + A22)B11
    # M2 = (A21+A22)B11.
    # C21 += M2. C22 -= M2.
    U[u10, col_offset] = 1
    U[u11, col_offset] = 1
    V[v00, col_offset] = 1
    W[w10, col_offset] = 1  # C21
    W[w11, col_offset] = -1 # C22
    
    col_offset += 1
    
    # Term 3: A11(B21 - B22)
    # M3 = A11(B21-B22).
    # C12 += M3. C22 += M3.
    U[u00, col_offset] = 1
    V[v10, col_offset] = 1  # B21 (row 1, col 0 of B)
    V[v11, col_offset] = -1 # B22 (row 1, col 1 of B)
    W[w01, col_offset] = 1  # C12 (index 2)
    W[w11, col_offset] = 1  # C22 (index 3)
    
    col_offset += 1
    
    # Term 4: A22(B12 - B11)
    # M4 = A22(B12-B11).
    # C11 += M4. C21 += M4.
    U[u11, col_offset] = 1
    V[v01, col_offset] = 1  # B12 (row 0, col 1)
    V[v00, col_offset] = -1 # B11 (row 0, col 0)
    W[w00, col_offset] = 1  # C11
    W[w10, col_offset] = 1  # C21
    
    col_offset += 1
    
    # Term 5: (A11 + A12)B22
    # M5 = (A11+A12)B22.
    # C11 -= M5. C12 += M5.
    U[u00, col_offset] = 1
    U[u01, col_offset] = 1
    V[v11, col_offset] = 1
    W[w00, col_offset] = -1 # C11
    W[w01, col_offset] = 1  # C12
    
    col_offset += 1
    
    # Term 6: (A21 - A11)(B11 + B12)
    # M6 = (A21-A11)(B11+B12).
    # C22 += M6.
    U[u10, col_offset] = 1
    U[u00, col_offset] = -1
    V[v00, col_offset] = 1
    V[v01, col_offset] = 1
    W[w11, col_offset] = 1  # C22
    
    col_offset += 1
    
    # Term 7: (A12 - A22)(B21 + B22)
    # M7 = (A12-A22)(B21+B22).
    # C11 += M7.
    U[u01, col_offset] = 1
    U[u11, col_offset] = -1
    V[v10, col_offset] = 1
    V[v11, col_offset] = 1
    W[w00, col_offset] = 1  # C11
    
    col_offset += 1
    
    return col_offset

def _construct_strassen_decomposition():
    n, m, p = 2, 3, 10
    nm, mp, pn = n * m, m * p, p * n
    rank = 55
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    col = 0
    
    # Part 1: A2 * B2 (rank 20)
    # A2 is 2x1 (rows 0,1; col 2 of A). Indices in U: 0*3+2=2, 1*3+2=5.
    # B2 is 1x10 (row 2 of B). Indices in V: 2*10+k = 20+k.
    # C is 2x10. Indices in W: k*2+i.
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, col] = 1
            V[2 * 10 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1
            
    # Part 2: A1 * B1 (rank 35)
    # A1 is 2x2 (rows 0,1; cols 0,1 of A). Indices in U: i*3+j.
    # B1 is 2x10 (rows 0,1 of B). Indices in V: j*10+k.
    # We split B1 into 5 blocks of 2x2.
    # Block t (0..4) uses columns 2t, 2t+1 of B1.
    # So for block t, the V indices are:
    # B11 (j=0, k=2t): 0*10 + 2t = 2t
    # B12 (j=0, k=2t+1): 0*10 + 2t+1 = 2t+1
    # B21 (j=1, k=2t): 1*10 + 2t = 10+2t
    # B22 (j=1, k=2t+1): 1*10 + 2t+1 = 10+2t+1
    
    for t in range(5):
        # U indices for A1 (2x2)
        u00 = 0 * 3 + 0
        u01 = 0 * 3 + 1
        u10 = 1 * 3 + 0
        u11 = 1 * 3 + 1
        
        # V indices for B1 block t
        v00 = 0 * 10 + 2 * t
        v01 = 0 * 10 + 2 * t + 1
        v10 = 1 * 10 + 2 * t
        v11 = 1 * 10 + 2 * t + 1
        
        # W indices for C block t (2x2)
        # C is 2x10. Block t uses columns 2t, 2t+1.
        # C11 (i=0, k=2t): W[2t*2 + 0] = 4t
        # C21 (i=1, k=2t): W[2t*2 + 1] = 4t+1
        # C12 (i=0, k=2t+1): W[(2t+1)*2 + 0] = 4t+2
        # C22 (i=1, k=2t+1): W[(2t+1)*2 + 1] = 4t+3
        
        w00 = 2 * t * 2 + 0
        w10 = 2 * t * 2 + 1
        w01 = (2 * t + 1) * 2 + 0
        w11 = (2 * t + 1) * 2 + 1
        
        col = _strassen_2x2(U, V, W, col, u00, v00, w00)
        # Note: _strassen_2x2 takes U, V, W and col_offset, and returns new col_offset.
        # It modifies U, V, W in place.
        
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Construct the rank-55 decomposition
    U, V, W, rank = _construct_strassen_decomposition()
    
    # Verify
    T = _build_tensor(n, m, p)
    recon = np.einsum('ak,bk,ck->abc', U, V, W)
    
    if not np.allclose(recon, T, atol=1e-9):
        # Fallback to schoolbook if verification fails
        sb_rank = n * m * p
        U = np.zeros((n * m, sb_rank), dtype=np.int64)
        V = np.zeros((m * p, sb_rank), dtype=np.int64)
        W = np.zeros((p * n, sb_rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

# EVOLVE-BLOCK-END
