import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 10
    N, M, P = 4, 4, 4
    rank = 23
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard rank-23 decomposition for 4x4x4 matrix multiplication
    # This uses the known minimal decomposition with entries in {0, 1/2, 1}
    # The decomposition is indexed by (i,j,k) in 4x4x4 space
    
    # We embed the 2x3x10 problem into 4x4x4 by zero-padding:
    # A is 2x3 -> placed in 4x4 with rows 0,1 and cols 0,1,2 (col 3 is zero)
    # B is 3x10 -> placed in 4x4 with rows 0,1,2 and cols 0..9... wait, that doesn't work for 4x4
    
    # Actually, the embedding approach: we treat the 2x3x10 tensor as a sub-tensor of 4x4x4
    # by mapping indices:
    # A[i,j] for i in {0,1}, j in {0,1,2} -> A4[i, j] in 4x4 (j=3 is 0)
    # B[j,k] for j in {0,1,2}, k in {0,...,9} -> this doesn't fit in 4x4 directly
    
    # The directive says to embed into 4x4x4 and project. But 10 > 4, so we can't directly embed.
    # Let me re-read the directive...
    
    # "Embed the 2x3x10 multiplication tensor into the 4x4x4 tensor space by zero-padding the dimensions"
    # This doesn't make sense since 10 > 4. Perhaps the idea is different.
    
    # Let me just use the current working approach with Strassen optimization
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :] - straightforward outer product
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A[:, 1:3] * B[1:3, :] using Strassen blocks
    for b in range(5):
        k0 = 2 * b
        
        # M1 = (A00 + A11)(B00 + B11)
        U[1][col] += 1
        U[5][col] += 1
        V[10 + k0][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[4][col] += 1
        U[5][col] += 1
        V[10 + k0][col] += 1
        W[2 * k0 + 1][col] += 1
        W[2 * (k0 + 1) + 1][col] -= 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[1][col] += 1
        V[10 + k0 + 1][col] += 1
        V[20 + k0 + 1][col] -= 1
        W[2 * (k0 + 1)][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[5][col] += 1
        V[20 + k0][col] += 1
        V[10 + k0][col] -= 1
        W[2 * k0][col] += 1
        W[2 * k0 + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[1][col] += 1
        U[2][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] -= 1
        W[2 * (k0 + 1)][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[4][col] += 1
        U[1][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k0 + 1][col] += 1
        W[2 * (k0 + 1) + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[2][col] += 1
        U[5][col] -= 1
        V[20 + k0][col] += 1
        V[20 + k0 + 1][col] += 1
        W[2 * k0][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
