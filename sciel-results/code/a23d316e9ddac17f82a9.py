"""Tensor decomposition for 2x2 * 2x12 matrix multiplication.

Strategy: exploit block structure. The 2x2x12 tensor consists of 6 independent
2x2x2 blocks. Each block uses Strassen's rank-7 decomposition.
Total rank = 6 * 7 = 42.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def _strassen_rank7():
    """Return U, V, W (each 2x7) for the 2x2x2 Strassen decomposition."""
    # U: input A space (2 elements: A00, A10)
    # V: input B space (2 elements: B00, B10)
    # W: output C space (2 elements: C00, C10)
    # Convention: C[i,j] = sum_k A[i,k] * B[k,j]
    # For 2x2x2: A is 2x2, B is 2x2, C is 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen's 7 products for computing C = A*B:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    #
    # But our tensor is 2x2x2 with indices:
    # A has indices (i, j) where i in {0,1}, j in {0,1} -> flattened as i*m+j
    # B has indices (j, k) where j in {0,1}, k in {0,1} -> flattened as j*p+k
    # C has indices (i, k) where i in {0,1}, k in {0,1} -> flattened as k*n+i
    #
    # Wait, the convention is W[k*n+i, r] for C-space.
    # So C-space index = k*n + i where n=2, so index = 2*k + i.
    #
    # For the 2x2x2 block:
    # A-space: i in {0,1}, j in {0,1} -> index = i*m + j = i*2 + j (m=2)
    #   A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    # B-space: j in {0,1}, k in {0,1} -> index = j*p + k = j*2 + k (p=2)
    #   B00 -> 0, B01 -> 1, B10 -> 2, B11 -> 3
    # C-space: k in {0,1}, i in {0,1} -> index = k*n + i = k*2 + i (n=2)
    #   C00 (i=0,k=0) -> 0, C10 (i=1,k=0) -> 1, C01 (i=0,k=1) -> 2, C11 (i=1,k=1) -> 3
    #
    # So for the 2x2x2 tensor:
    # U[0,:] = A00 coefficients, U[1,:] = A01 coefficients, U[2,:] = A10 coefficients, U[3,:] = A11 coefficients
    # V[0,:] = B00 coefficients, V[1,:] = B01 coefficients, V[2,:] = B10 coefficients, V[3,:] = B11 coefficients
    # W[0,:] = C00 coefficients, W[1,:] = C10 coefficients, W[2,:] = C01 coefficients, W[3,:] = C11 coefficients
    #
    # Strassen:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    #
    # U columns (7 columns for 7 products):
    # Product 1: A00 + A11 -> [1, 0, 0, 1]
    # Product 2: A10 + A11 -> [0, 0, 1, 1]
    # Product 3: A00 -> [1, 0, 0, 0]
    # Product 4: A11 -> [0, 0, 0, 1]
    # Product 5: A00 + A01 -> [1, 1, 0, 0]
    # Product 6: A10 - A00 -> [-1, 0, 1, 0]
    # Product 7: A01 - A11 -> [0, 1, 0, -1]
    #
    # V columns:
    # Product 1: B00 + B11 -> [1, 0, 0, 1]
    # Product 2: B00 -> [1, 0, 0, 0]
    # Product 3: B01 - B11 -> [0, 1, 0, -1]
    # Product 4: B10 - B00 -> [-1, 0, 1, 0]
    # Product 5: B11 -> [0, 0, 0, 1]
    # Product 6: B00 + B01 -> [1, 1, 0, 0]
    # Product 7: B10 + B11 -> [0, 0, 1, 1]
    #
    # W columns:
    # C00 = M1 - M4 + M7 -> [1, 0, 0, 0, 0, 0, 1] for M1, [0, 0, 0, 0, 0, 0, 0]... 
    # Wait, W[row, col] where row is C-index, col is product-index.
    # C00: M1 - M4 + M7 -> row 0: [1, 0, 0, -1, 0, 0, 1]
    # C01: M3 + M5 -> row 2: [0, 0, 1, 0, 1, 0, 0]
    # C10: M2 + M4 -> row 1: [0, 1, 0, 1, 0, 0, 0]
    # C11: M1 - M3 + M6 -> row 3: [1, 0, -1, 0, 0, 1, 0]

    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ])
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ])
    W = np.array([
        [1, 0, 0, -1, 0, 0, 1],   # C00 (k=0,i=0)
        [0, 1, 0, 1, 0, 0, 0],    # C10 (k=0,i=1)
        [0, 0, 1, 0, 1, 0, 0],    # C01 (k=1,i=0)
        [1, 0, -1, 0, 0, 1, 0],   # C11 (k=1,i=1)
    ])
    return U, V, W


def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    start = time.time()

    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    # A is n x m = 2x2, B is m x p = 2x12, C is n x p = 2x12
    # A-space size: n*m = 4
    # B-space size: m*p = 24
    # C-space size: p*n = 24 (index = k*n + i)
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_s, V_s, W_s = _strassen_rank7()
    
    # 6 blocks: each block corresponds to a column-block of B (width 2)
    # B is 2x12, so 6 blocks of 2 columns each
    # For block r (0..5):
    #   B-space indices: j*12 + (2*r + k) for j in {0,1}, k in {0,1}
    #   C-space indices: (2*r + k)*n + i for k in {0,1}, i in {0,1}
    
    for r in range(6):
        col_start = r * 7
        col_end = (r + 1) * 7
        # U: same for all blocks (A is the same 2x2 matrix)
        U[:, col_start:col_end] = U_s
        
        # V: place V_s in the r-th block of B-space
        for j in range(m):  # j in {0, 1}
            for k in range(2):  # k in {0, 1}
                b_idx = j * p + (2 * r + k)  # j*12 + (2r + k)
                v_idx = j * 2 + k  # index in V_s
                V[b_idx, col_start:col_end] = V_s[v_idx, :]
        
        # W: place W_s in the r-th block of C-space
        for k in range(2):  # k in {0, 1}
            for i in range(n):  # i in {0, 1}
                c_idx = (2 * r + k) * n + i  # (2r+k)*2 + i
                w_idx = k * n + i  # index in W_s
                W[c_idx, col_start:col_end] = W_s[w_idx, :]
    
    # Verify: time check
    if time.time() > deadline:
        # Return what we have (should be complete already)
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
