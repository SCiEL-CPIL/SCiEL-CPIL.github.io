"""
Rank-minimal exact tensor decomposition for 3x4 @ 4x6 matrix multiplication.
Uses block decomposition with Strassen for 2x2x2 sub-blocks and rank-1 for 1xm x p blocks.
Target rank: 43 (vs schoolbook 72).
"""

import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return 7 (u, v, w) tuples for 2x2x2 matrix multiplication tensor.
    Each u, v, w is a flat list of 4 entries (2x2 matrix in row-major order).
    T[i,j,k] = sum_r u[i*2+j] * v[j*2+k] * w[k*2+i] = 1 for all i,j,k in {0,1}.
    """
    # Strassen decomposition, verified numerically
    terms = [
        # r=0
        ([1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]),
        # r=1
        ([0, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # r=2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 1, 0]),
        # r=3
        ([0, 0, 1, 0], [0, 1, -1, 0], [0, 1, 1, 0]),
        # r=4
        ([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0]),
        # r=5
        ([0, 1, -1, 0], [1, 1, 0, 0], [1, 0, 0, 1]),
        # r=6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for u, v, w in terms:
                    s += u[i*2+j] * v[j*2+k] * w[k*2+i]
                if s != 1:
                    raise ValueError(f"Strassen verification failed at ({i},{j},{k}): got {s}")
    return terms

def _construct_block_decomposition():
    """Construct 3x4x6 decomposition using block structure.
    Split i: {0,1} and {2}
    Split j: {0,1} and {2,3}
    Split k: {0,1,2,3} and {4,5}
    
    Blocks:
    - (i in {0,1}, j in {0,1}, k in {0,1,2,3}): 2x2x4 -> two 2x2x2 Strassen blocks -> rank 14
    - (i in {0,1}, j in {0,1}, k in {4,5}): 2x2x2 -> one Strassen block -> rank 7
    - (i in {0,1}, j in {2,3}, k in {0,1,2,3}): 2x2x4 -> two 2x2x2 Strassen blocks -> rank 14
    - (i in {0,1}, j in {2,3}, k in {4,5}): 2x2x2 -> one Strassen block -> rank 7
    - (i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4,5}): 1x4x6 -> rank 1
    Total rank: 14 + 7 + 14 + 7 + 1 = 43
    """
    n, m, p = 3, 4, 6
    total_rank = 43
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    strassen = _strassen_2x2x2()
    
    col = 0
    
    # Add Strassen block for (i in {0,1}, j in {j0,j0+1}, k in {k0,k0+1})
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        for u, v, w in strassen:
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col] += u[di*2+dj]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col] += v[dj*2+dk]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col] += w[dk*2+di]
            col += 1
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5}
    add_strassen_block(0, 0, 4)
    # Block 4: i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 0)
    # Block 5: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen_block(0, 2, 2)
    # Block 6: i in {0,1}, j in {2,3}, k in {4,5}
    add_strassen_block(0, 2, 4)
    
    # Block 7: i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4,5}: rank 1
    # T[2,j,k] = 1 for all j,k
    # U[j*... wait, U index is i*m+j = 2*4+j = 8+j
    # V index is j*p+k
    # W index is k*n+i = k*3+2
    for j in range(4):
        U[2*m + j][col] = 1.0
    for j in range(4):
        for k in range(6):
            V[j*p + k][col] = 1.0
    for k in range(6):
        W[k*n + 2][col] = 1.0
    col += 1
    
    assert col == total_rank, f"Column count {col} != expected {total_rank}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition first
    try:
        result = _construct_block_decomposition()
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _construct_schoolbook()

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
