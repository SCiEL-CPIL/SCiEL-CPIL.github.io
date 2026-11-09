"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _add_strassen_2x2x2(U, V, W, i_idx, j_idx, k_idx, rank_offset):
    """
    Adds a 2x2x2 Strassen decomposition.
    i_idx: [i0, i1]
    j_idx: [j0, j1]
    k_idx: [k0, k1]
    """
    n, m, p = DIMENSIONS
    # Map local indices to global U, V, W indices
    # U[i, j] -> i*m + j
    u00 = i_idx[0]*m + j_idx[0]
    u01 = i_idx[0]*m + j_idx[1]
    u10 = i_idx[1]*m + j_idx[0]
    u11 = i_idx[1]*m + j_idx[1]
    
    # V[j, k] -> j*p + k
    v00 = j_idx[0]*p + k_idx[0]
    v01 = j_idx[0]*p + k_idx[1]
    v10 = j_idx[1]*p + k_idx[0]
    v11 = j_idx[1]*p + k_idx[1]
    
    # W[k, i] -> k*n + i
    w00 = k_idx[0]*n + i_idx[0]
    w01 = k_idx[0]*n + i_idx[1]
    w10 = k_idx[1]*n + i_idx[0]
    w11 = k_idx[1]*n + i_idx[1]
    
    # M1 = (a00+a11)(b00+b11) -> c00 + c11
    U[u00][rank_offset] += 1
    U[u11][rank_offset] += 1
    V[v00][rank_offset] += 1
    V[v11][rank_offset] += 1
    W[w00][rank_offset] += 1
    W[w11][rank_offset] += 1
    rank_offset += 1
    
    # M2 = (a10+a11)b00 -> c10 - c11
    U[u10][rank_offset] += 1
    U[u11][rank_offset] += 1
    V[v00][rank_offset] += 1
    W[w10][rank_offset] += 1
    W[w11][rank_offset] -= 1
    rank_offset += 1
    
    # M3 = a00(b01-b11) -> c01 + c11
    U[u00][rank_offset] += 1
    V[v01][rank_offset] += 1
    V[v11][rank_offset] -= 1
    W[w01][rank_offset] += 1
    W[w11][rank_offset] += 1
    rank_offset += 1
    
    # M4 = a11(b10-b00) -> c00 + c10
    U[u11][rank_offset] += 1
    V[v10][rank_offset] += 1
    V[v00][rank_offset] -= 1
    W[w00][rank_offset] += 1
    W[w10][rank_offset] += 1
    rank_offset += 1
    
    # M5 = (a00+a10)b11 -> -c00 + c01
    U[u00][rank_offset] += 1
    U[u10][rank_offset] += 1
    V[v11][rank_offset] += 1
    W[w00][rank_offset] -= 1
    W[w01][rank_offset] += 1
    rank_offset += 1
    
    # M6 = (a01-a00)(b00+b01) -> c11
    U[u01][rank_offset] += 1
    U[u00][rank_offset] -= 1
    V[v00][rank_offset] += 1
    V[v01][rank_offset] += 1
    W[w11][rank_offset] += 1
    rank_offset += 1
    
    # M7 = (a11-a10)(b10+b11) -> c00
    U[u11][rank_offset] += 1
    U[u10][rank_offset] -= 1
    V[v10][rank_offset] += 1
    V[v11][rank_offset] += 1
    W[w00][rank_offset] += 1
    rank_offset += 1
    
    return rank_offset

def _add_schoolbook_block(U, V, W, i_idx, j_idx, k_idx, rank_offset):
    """
    Adds schoolbook terms for given index sets.
    """
    n, m, p = DIMENSIONS
    for i in i_idx:
        for j in j_idx:
            for k in k_idx:
                U[i*m + j][rank_offset] = 1
                V[j*p + k][rank_offset] = 1
                W[k*n + i][rank_offset] = 1
                rank_offset += 1
    return rank_offset

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # Part 1 (j=0,1):
    #   k=0,1: Strassen (7)
    #   k=2,3: Strassen (7)
    #   k=4,5,6: Schoolbook 2x2x3 (12)
    #   Total Part 1: 26
    # Part 2 (j=2):
    #   k=0..6: Schoolbook 2x1x7 (14)
    #   Total Part 2: 14
    # Total Rank: 40
    
    rank = 40
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Part 1: j=0,1
    i_set = [0, 1]
    j_set_01 = [0, 1]
    
    # k=0,1
    rank_offset = _add_strassen_2x2x2(U, V, W, i_set, j_set_01, [0, 1], rank_offset)
    
    # k=2,3
    rank_offset = _add_strassen_2x2x2(U, V, W, i_set, j_set_01, [2, 3], rank_offset)
    
    # k=4,5,6
    rank_offset = _add_schoolbook_block(U, V, W, i_set, j_set_01, [4, 5, 6], rank_offset)
    
    # Part 2: j=2
    j_set_2 = [2]
    k_set_all = [0, 1, 2, 3, 4, 5, 6]
    rank_offset = _add_schoolbook_block(U, V, W, i_set, j_set_2, k_set_all, rank_offset)
    
    # Verify rank_offset matches rank
    assert rank_offset == rank, f"Rank mismatch: {rank_offset} vs {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
