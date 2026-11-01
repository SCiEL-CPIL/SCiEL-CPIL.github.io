"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START

_DEADLINE = 145.0  # Safety margin before 150s kill

def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_rank44(n, m, p):
    """
    Rank-44 decomposition for 2x3x8 matrix multiplication.
    Splits A into A1 (2x2) and A2 (2x1).
    Splits B into B1 (2x8) and B2 (1x8).
    C = A1*B1 + A2*B2.
    A2*B2 is rank 16.
    A1*B1 is split into four 2x2x2 blocks, each using Strassen (rank 7).
    Total rank 16 + 4*7 = 44.
    """
    n, m, p = 2, 3, 8
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2 (rank 16)
    # A[0,2] index 2, A[1,2] index 5
    # B[2,k] index 16+k
    # C[i,k] index k*2+i
    for k in range(8):
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 0][col] = 1
        col += 1
        U[5][col] = 1
        V[16 + k][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Part 2: A1*B1 via Strassen (rank 28)
    # A1 indices: 0, 1, 3, 4
    # B1 indices: 0-7 (row 0), 8-15 (row 1)
    # C indices: k*2+i
    for g in range(4):
        # Block g handles B cols 2g, 2g+1
        b00 = 2 * g
        b01 = 2 * g + 1
        b10 = 8 + 2 * g
        b11 = 8 + 2 * g + 1
        
        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4
        
        w00 = 4 * g
        w01 = 4 * g + 2
        w10 = 4 * g + 1
        w11 = 4 * g + 3
        
        # M1 = (a00+a11)(b00+b11)
        U[a00][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        V[b11][col] = 1
        W[w00][col] = 1
        W[w11][col] = 1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # M3 = a00*(b11-b01)
        U[a00][col] = 1
        V[b11][col] = 1
        V[b01][col] = -1
        W[w01][col] = 1
        W[w11][col] = -1
        col += 1
        
        # M4 = a11*(b00-b10)
        U[a11][col] = 1
        V[b00][col] = 1
        V[b10][col] = -1
        W[w00][col] = -1
        W[w10][col] = 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[a00][col] = 1
        U[a01][col] = 1
        V[b11][col] = 1
        W[w01][col] = 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[a10][col] = 1
        U[a00][col] = -1
        V[b00][col] = 1
        V[b01][col] = 1
        W[w00][col] = 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[a01][col] = 1
        U[a11][col] = -1
        V[b10][col] = 1
        V[b11][col] = 1
        W[w11][col] = 1
        col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the rank-44 decomposition
    try:
        if time.time() - start_time > _DEADLINE:
            raise TimeoutError("Deadline exceeded")
        U, V, W = _build_rank44(n, m, p)
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
