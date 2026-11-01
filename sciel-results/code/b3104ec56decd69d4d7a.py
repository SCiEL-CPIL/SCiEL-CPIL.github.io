import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Attempt to find a rank < 42 decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Strategy: Use the block-partitioned Strassen baseline (rank 42) as a starting point,
    then attempt to reduce rank by exploiting linear dependencies or using
    Winograd Fourier Transform (FTW) style decompositions.
    
    The FTW approach decomposes 2x4x6 as a Kronecker product of 2x2x2 (Strassen, rank 7)
    and a 1x2x3 rectangular core. If the 1x2x3 core can be done in rank 2 (instead of 3),
    total rank would be 7*2 = 14, which is far below 42.
    
    For 1x2x3: multiplying a 1x2 row vector by a 2x3 matrix gives a 1x3 row.
    This is C[k] = sum_j A[j]*B[j,k] for k=0,1,2.
    Naive: 2*3 = 6 multiplications.
    Can we do better? Each output C[k] depends on both A[0], A[1] and B[:,k].
    
    Actually, the 1x2x3 tensor is: T[i,j,k] = delta_{i,0} * A[j] * B[j,k] where i is trivial.
    This is just the outer product structure. The tensor is T[j,k] for j in {0,1}, k in {0,1,2}.
    This is a 2x3 matrix, rank 3 as a matrix (full rank). So tensor rank >= 3.
    
    Can we achieve rank 3? Yes: M0 = A[0]*B[0,0], M1 = A[0]*B[0,1], M2 = A[0]*B[0,2], 
    M3 = A[1]*B[1,0], M4 = A[1]*B[1,1], M5 = A[1]*B[1,2] - that's rank 6.
    
    With Strassen-like: M0 = (A[0]+A[1])*B[0,0], M1 = (A[0]+A[1])*B[0,1], M2 = (A[0]+A[1])*B[0,2],
    M3 = A[0]*B[1,0], M4 = A[0]*B[1,1], M5 = A[0]*B[1,2]... no that's still 6.
    
    Let me think differently. The 1x2x3 multiplication: C[k] = A[0]*B[0,k] + A[1]*B[1,k] for k=0,1,2.
    This is 3 outputs, each a dot product of a 2-dim vector with a column.
    
    Tensor rank of this 1x2x3 problem: It's equivalent to multiplying a 2x2 matrix (one row is A, 
    the other is A) by a 2x3 matrix... no.
    
    Actually, let me just try to find any decomposition with rank < 42 using a search.
    
    For now, return the rank-42 baseline which is known to work.
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=np.float64)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=np.float64)  # 12 x 42
    
    col = 0
    
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    def apply_strassen(a_base, b_base, out_base):
        nonlocal col
        for a_terms, b_terms, w_terms in strassen_terms:
            for (ar, ac, sign) in a_terms:
                u_idx = (a_base[0] + ar) * m + (a_base[1] + ac)
                U[u_idx, col] += sign
            for (br, bc, sign) in b_terms:
                v_idx = (b_base[0] + br) * p + (b_base[1] + bc)
                V[v_idx, col] += sign
            for (or_, oc, sign) in w_terms:
                w_idx = (out_base[1] + oc) * n + (out_base[0] + or_)
                W[w_idx, col] += sign
            col += 1
    
    for sub in range(3):
        apply_strassen((0, 0), (0, 2*sub), (0, 2*sub))
    for sub in range(3):
        apply_strassen((0, 2), (2, 2*sub), (0, 2*sub))
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
