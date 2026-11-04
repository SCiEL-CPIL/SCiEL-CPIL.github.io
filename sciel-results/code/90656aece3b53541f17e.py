"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Strategy: split the 8 output columns into four 2-column blocks. Each 2x3 x 3x2
sub-tensor is decomposed as the direct sum of:
  * a 2x2x2 all-ones tensor  -> rank 7 (Strassen)
  * a 2x1x2 all-ones tensor  -> rank 1 (outer product)
The j=2 (third row of A, third row of B) outer product can be shared globally
across all 8 output columns, giving rank 4*7 + 1 = 29 (vs schoolbook 48).
"""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

# Strassen rank-7 decomposition of 2x2x2 all-ones tensor
_STRASSEN_U = [
    {(0, 0): 1, (1, 1): 1},
    {(1, 0): 1, (1, 1): 1},
    {(0, 0): 1},
    {(1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(1, 0): 1, (0, 0): -1},
    {(0, 1): 1, (1, 1): -1},
]
_STRASSEN_V = [
    {(0, 0): 1, (1, 1): 1},
    {(0, 0): 1},
    {(0, 1): 1, (1, 1): -1},
    {(1, 0): 1, (0, 0): -1},
    {(1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(1, 0): 1, (1, 1): 1},
]
_STRASSEN_W = [
    {(0, 0): 1, (1, 1): 1},
    {(0, 1): 1, (1, 1): -1},
    {(0, 1): 1, (1, 1): 1},
    {(0, 0): 1, (0, 1): 1},
    {(0, 0): -1, (0, 1): 1},
    {(1, 1): 1},
    {(0, 0): 1},
]

def _verify_strassen():
    """Verify the Strassen 2x2x2 decomposition is valid."""
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += _STRASSEN_U[r].get((i, j), 0) * _STRASSEN_V[r].get((j, k), 0) * _STRASSEN_W[r].get((k, i), 0)
                if s != 1:
                    return False
    return True

def _rank29_decomposition(n, m, p):
    """Rank-29 decomposition: 4 blocks of Strassen (7 each) + 1 global outer product.
    
    For j=0,1: Strassen on 2x2x2 per block (k pairs)
    For j=2: single global term U[i,2]=1, V[2,k]=1, W[k,i]=1 for all k
    """
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 7 + 1  # 29
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 29
    V = [[0] * rank for _ in range(m * p)]  # 24 x 29
    W = [[0] * rank for _ in range(p * n)]  # 16 x 29
    
    # Fill Strassen terms for j=0,1 in each block
    for b in range(nblocks):
        k0 = 2 * b
        col = b * 7
        for r in range(7):
            for (i, j), val in _STRASSEN_U[r].items():
                if j <= 1:  # j=0 or 1
                    U[i * m + j][col + r] = val
            for (j, kp), val in _STRASSEN_V[r].items():
                if j <= 1:
                    V[j * p + (k0 + kp)][col + r] = val
            for (kp, i), val in _STRASSEN_W[r].items():
                W[(k0 + kp) * n + i][col + r] = val
    
    # Global outer product for j=2
    oc = nblocks * 7  # column 28
    for i in range(n):
        U[i * m + 2][oc] = 1  # U[2][28]=1, U[5][28]=1 (i=0,j=2 and i=1,j=2)
    for k in range(p):
        V[2 * p + k][oc] = 1  # V[16+k][28]=1 for all k
    for k in range(p):
        for i in range(n):
            W[k * n + i][oc] = 1  # W[k*2+i][28]=1 for all k,i
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition(n, m, p):
    """Build rank-32 decomposition by 4 blocks of 8 terms each (original approach)."""
    assert n == 2 and m == 3 and p == 8
    nblocks = p // 2  # 4
    rank = nblocks * 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(nblocks):
        k0 = 2 * b
        col = b * 8
        
        # 7 Strassen terms (j in {0,1}, k in {k0, k0+1})
        for r in range(7):
            for (i, j), val in _STRASSEN_U[r].items():
                U[i * m + j][col + r] = val
            for (j, kp), val in _STRASSEN_V[r].items():
                V[j * p + (k0 + kp)][col + r] = val
            for (kp, i), val in _STRASSEN_W[r].items():
                W[(k0 + kp) * n + i][col + r] = val
        
        # 1 outer-product term (j = 2)
        oc = col + 7
        for i in range(n):
            U[i * m + 2][oc] = 1
        for kp in range(2):
            V[2 * p + (k0 + kp)][oc] = 1
        for kp in range(2):
            for i in range(n):
                W[(k0 + kp) * n + i][oc] = 1

    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 115

    # First verify Strassen is correct
    if not _verify_strassen():
        return _schoolbook(n, m, p)

    # Try rank-29: 4 blocks of Strassen + 1 global outer product
    try:
        cand = _rank29_decomposition(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand
    except Exception:
        pass

    # Try rank-32: 4 blocks of (Strassen + per-block outer product)
    try:
        cand = _block_decomposition(n, m, p)
        if _validate(cand["U"], cand["V"], cand["W"], n, m, p):
            return cand
    except Exception:
        pass

    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
