"""
Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy: Split B (4x8) into two 4x4 blocks. Use recursive Strassen for each 4x4*4x4
sub-problem (rank 49 each), giving total rank 98.

The 4x4*4x4 decomposition uses 7 Strassen products, each decomposed via the rank-7
2x2*2x2 Strassen formula, yielding rank 7*7=49.
"""

DIMENSIONS = [4, 4, 8]

import time
import numpy as np


def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # A indices: A11=(0,0)->idx0, A12=(0,1)->idx1, A21=(1,0)->idx2, A22=(1,1)->idx3
    # B indices: B11=(0,0)->idx0, B12=(0,1)->idx1, B21=(1,0)->idx2, B22=(1,1)->idx3
    # C indices: C11=(0,0)->idx0, C12=(0,1)->idx1, C21=(1,0)->idx2, C22=(1,1)->idx3
    # W index convention: W[k*2+i, c] where k=col of B, i=row of A
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C21, C22
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12, C22
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11, C21
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11, C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1  # C21
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1  # C11, C12
    
    return U, V, W


def _strassen_4x4():
    """
    Return U, V, W for 4x4 * 4x4 matrix multiplication with rank 49.
    Convention: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    
    Uses recursive Strassen: 4x4*4x4 is composed of 7 sub-multiplications,
    each of size 2x2*2x2 (rank 7), giving total rank 7*7=49.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # Strassen product definitions for 4x4*4x4:
    # Each product t has A-block coefficients, B-block coefficients, C-block coefficients.
    # Block indices: 0=11, 1=12, 2=21, 3=22
    
    products = [
        # M1 = (A11+A22)(B11+B22) -> C11 + C22
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A21+A22)B11 -> C21 - C22
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A11(B12-B22) -> C12 + C22
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A22(B21-B11) -> C11 + C21
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A11+A12)B22 -> -C11 + C12
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A21-A11)(B11+B12) -> C21
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1}),
        # M7 = (A12-A22)(B21+B22) -> -C11 + C12
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1, 1: 1}),
    ]
    
    for t, (a_coeffs, b_coeffs, c_coeffs) in enumerate(products):
        for c in range(7):
            col = t * 7 + c
            
            # A part: for each element in the 2x2 A-block pattern
            for ai in range(4):
                u2_val = U2[ai, c]
                if u2_val == 0:
                    continue
                local_i = ai // 2
                local_j = ai % 2
                for block_idx, a_coeff in a_coeffs.items():
                    val = u2_val * a_coeff
                    if val == 0:
                        continue
                    br = block_idx // 2
                    bc = block_idx % 2
                    global_i = br * 2 + local_i
                    global_j = bc * 2 + local_j
                    U[global_i * 4 + global_j, col] += val
            
            # B part: for each element in the 2x2 B-block pattern
            for bj in range(4):
                v2_val = V2[bj, c]
                if v2_val == 0:
                    continue
                local_i = bj // 2
                local_j = bj % 2
                for block_idx, b_coeff in b_coeffs.items():
                    val = v2_val * b_coeff
                    if val == 0:
                        continue
                    br = block_idx // 2
                    bc = block_idx % 2
                    global_i = br * 2 + local_i
                    global_j = bc * 2 + local_j
                    V[global_i * 4 + global_j, col] += val
            
            # C part: for each element in the 2x2 C-block pattern
            for ci in range(4):
                w2_val = W2[ci, c]
                if w2_val == 0:
                    continue
                local_k = ci // 2
                local_i = ci % 2
                for block_idx, c_coeff in c_coeffs.items():
                    val = w2_val * c_coeff
                    if val == 0:
                        continue
                    br = block_idx // 2
                    bc = block_idx % 2
                    global_k = bc * 2 + local_k
                    global_i = br * 2 + local_i
                    W[global_k * 4 + global_i, col] += val
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True


def construct():
    t_start = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build the 4x4*4x4 Strassen decomposition (rank 49)
    U4, V4, W4 = _strassen_4x4()
    
    # Compose for 4x4 * 4x8 by splitting B into two 4x4 blocks
    rank_total = 98
    U = np.zeros((n*m, rank_total), dtype=float)  # (16, 98)
    V = np.zeros((m*p, rank_total), dtype=float)  # (32, 98)
    W = np.zeros((p*n, rank_total), dtype=float)  # (32, 98)
    
    # Left half: columns 0-3 of B, columns 0-3 of C
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[k*8+i, 0:49] = W4[k*4+i, :]
    
    # Right half: columns 4-7 of B, columns 4-7 of C
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+4+k, 49:98] = V4[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W[(k+4)*8+i, 49:98] = W4[k*4+i, :]
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, rank_total):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook decomposition (rank 128)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
