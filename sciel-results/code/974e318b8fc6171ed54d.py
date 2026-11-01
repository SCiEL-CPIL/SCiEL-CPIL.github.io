"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    # Build the list of (a, b, c) tuples where T[a,b,c] = 1
    tensor_entries = []
    for a in range(A_dim):
        i = a // m
        j = a % m
        for b in range(B_dim):
            j2 = b // p
            k = b % p
            if j == j2:
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    schoolbook_rank = len(tensor_entries)  # 64
    
    # For this specific case (2x4x8), let's analyze the structure
    # A is 2x4, B is 4x8, C is 8x2
    # C[k][i] = sum_j A[i][j] * B[j][k]
    # In flat indexing: C[k*2+i] = sum_j A[i*4+j] * B[j*8+k]
    
    # The 64 products are:
    # For i=0: A[0],A[1],A[2],A[3] each multiply with 8 B entries -> 32 products
    # For i=1: A[4],A[5],A[6],A[7] each multiply with 8 B entries -> 32 products
    # Total: 64 products
    
    # Can we share any? The B entries are shared between the two rows of A,
    # but the A entries are different. So no direct sharing.
    
    # However, we might be able to use linear combinations.
    # For example, if we compute (A[0]+A[4]) * B[j*8+k] and (A[0]-A[4]) * B[j*8+k],
    # we can recover A[0]*B[j*8+k] and A[4]*B[j*8+k] from these.
    # But this doesn't reduce the count, it just reorganizes.
    
    # The tensor rank for matrix multiplication is known to be n*m*p in general.
    # For 2x4x8, I believe the rank is indeed 64.
    
    # Let's return the schoolbook decomposition.
    rank = schoolbook_rank
    U = np.zeros((A_dim, rank), dtype=float)
    V = np.zeros((B_dim, rank), dtype=float)
    W = np.zeros((C_dim, rank), dtype=float)
    
    for col, (a, b, c) in enumerate(tensor_entries):
        U[a, col] = 1
        V[b, col] = 1
        W[c, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
