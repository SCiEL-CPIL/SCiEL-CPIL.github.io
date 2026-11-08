import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # We will use a block decomposition strategy.
    # The matrix product C = A * B where A is 2x3 and B is 3x10.
    # We can view A as a row of 2x2 blocks (with the last one being 2x1) and B as a column of 2x2 blocks (with the first one being 1x2).
    # Actually, a simpler way:
    # Let's partition the intermediate dimension m=3 into two parts: m1=2, m2=1.
    # A = [A1 | A2] where A1 is 2x2 and A2 is 2x1.
    # B = [B1; B2] where B1 is 2x10 and B2 is 1x10.
    # C = A1 * B1 + A2 * B2.
    
    # This doesn't directly use Strassen on 2x2x2 unless we split the 10 dimension too.
    # Let's partition p=10 into five 2x2 blocks.
    # B1 is 2x10, so B1 = [B1_1, B1_2, B1_3, B1_4, B1_5] where each Bi is 2x2.
    # C is 2x10, so C = [C_1, C_2, C_3, C_4, C_5] where each Cj is 2x2.
    # Cj = A1 * B1_j + A2 * B2_j.
    # This is 5 independent products of (2x2 * 2x2) + (2x1 * 1x2).
    # The term A2 * B2_j is an outer product of a 2x1 and 1x2, which is rank 2 (or less).
    # Wait, A2 is 2x1, B2_j is 1x2. A2 * B2_j is a 2x2 matrix.
    # Let A2 = [a; b] and B2_j = [c, d]. Then A2 * B2_j = [[ac, ad], [bc, bd]].
    # This requires 4 multiplications (ac, ad, bc, bd).
    # The term A1 * B1_j is a 2x2 * 2x2 product, which can be done in 7 multiplications via Strassen.
    # So for each of the 5 blocks, we need 7 + 4 = 11 multiplications?
    # Total rank = 5 * 11 = 55.
    
    # Let's verify this logic.
    # A1 is 2x2: [[a11, a12], [a21, a22]]
    # A2 is 2x1: [a31, a32] (using indices from A)
    # B1_j is 2x2: [[b11, b12], [b21, b22]] (indices from B)
    # B2_j is 1x2: [b31, b32] (indices from B)
    
    # C_j = A1 * B1_j + A2 * B2_j
    # Let M = A1 * B1_j.
    # M11 = a11*b11 + a12*b21
    # M12 = a11*b12 + a12*b22
    # M21 = a21*b11 + a22*b21
    # M22 = a21*b12 + a22*b22
    
    # Let N = A2 * B2_j
    # N11 = a31*b31
    # N12 = a31*b32
    # N21 = a32*b31
    # N22 = a32*b32
    
    # C11 = M11 + N11
    # C12 = M12 + N12
    # C21 = M21 + N21
    # C22 = M22 + N22
    
    # We can use Strassen for M.
    # Strassen formulas for M = A1 * B1_j:
    # Let A1 = [[a, b], [c, d]] and B1_j = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-d)(e+g)
    
    # M11 = m1 + m4 - m5 + m7
    # M12 = m3 + m5
    # M21 = m2 + m4
    # M22 = m1 - m2 + m3 - m6
    
    # Now we add N.
    # N11 = a31*b31
    # N12 = a31*b32
    # N21 = a32*b31
    # N22 = a32*b32
    
    # So for each block j (0 to 4):
    # We need to compute 7 Strassen products + 4 outer product multiplications.
    # Total 11 products per block.
    # Total 5 blocks * 11 = 55 products.
    
    # Let's construct the U, V, W matrices.
    # Rank = 55.
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (2x3):
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # A1 = [[0, 1], [3, 4]]
    # A2 = [2, 5]
    
    # Indices for B (3x10):
    # B[0,0]=0, B[0,1]=1, ..., B[0,9]=9
    # B[1,0]=10, B[1,1]=11, ..., B[1,9]=19
    # B[2,0]=20, B[2,1]=21, ..., B[2,9]=29
    # For block j (0 to 4):
    # B1_j = [[B[0, 2j], B[0, 2j+1]], [B[1, 2j], B[1, 2j+1]]]
    # B2_j = [B[2, 2j], B[2, 2j+1]]
    
    # Indices for C (2x10):
    # C[0,0]=0, C[0,1]=1, ..., C[0,9]=9
    # C[1,0]=10, C[1,1]=11, ..., C[1,9]=19
    # W[k*n + i, r] corresponds to C[i, k].
    # So W index is k*n + i.
    # For block j, C_j = C[:, 2j:2j+2]
    # C[0, 2j] -> W[2j*2 + 0] = W[4j]
    # C[0, 2j+1] -> W[2j*2 + 1] = W[4j+1]
    # C[1, 2j] -> W[2j*2 + 2] = W[4j+2]
    # C[1, 2j+1] -> W[2j*2 + 3] = W[4j+3]
    
    col = 0
    for j in range(5):
        # B1_j indices
        e_idx = 2 * j      # B[0, 2j]
        f_idx = 2 * j + 1  # B[0, 2j+1]
        g_idx = 10 + 2 * j      # B[1, 2j]
        h_idx = 10 + 2 * j + 1  # B[1, 2j+1]
        
        # B2_j indices
        b31_idx = 20 + 2 * j      # B[2, 2j]
        b32_idx = 20 + 2 * j + 1  # B[2, 2j+1]
        
        # A1 indices
        a_idx = 0 # a11
        b_idx = 1 # a12
        c_idx = 3 # a21
        d_idx = 4 # a22
        
        # A2 indices
        a31_idx = 2 # a31
        a32_idx = 5 # a32
        
        # C indices for this block
        c11_w = 4 * j      # W[4j]
        c12_w = 4 * j + 1  # W[4j+1]
        c21_w = 4 * j + 2  # W[4j+2]
        c22_w = 4 * j + 3  # W[4j+3]
        
        # Strassen products
        # m1 = (a+d)(e+h)
        U[a_idx][col] = 1
        U[d_idx][col] = 1
        V[e_idx][col] = 1
        V[h_idx][col] = 1
        W[c11_w][col] = 1
        W[c22_w][col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[c_idx][col] = 1
        U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[c21_w][col] = 1
        W[c22_w][col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[a_idx][col] = 1
        V[f_idx][col] = 1
        V[h_idx][col] = -1
        W[c12_w][col] = 1
        W[c22_w][col] = -1
        col += 1
        
        # m4 = d(g-e)
        U[d_idx][col] = 1
        V[g_idx][col] = 1
        V[e_idx][col] = -1
        W[c11_w][col] = 1
        W[c21_w][col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[a_idx][col] = 1
        U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[c11_w][col] = -1
        W[c12_w][col] = 1
        col += 1
        
        # m6 = (c-d)(f+h)
        U[c_idx][col] = 1
        U[d_idx][col] = -1
        V[f_idx][col] = 1
        V[h_idx][col] = 1
        W[c22_w][col] = -1
        col += 1
        
        # m7 = (b-d)(e+g)
        U[b_idx][col] = 1
        U[d_idx][col] = -1
        V[e_idx][col] = 1
        V[g_idx][col] = 1
        W[c11_w][col] = 1
        col += 1
        
        # Outer product terms for N = A2 * B2_j
        # N11 = a31*b31
        U[a31_idx][col] = 1
        V[b31_idx][col] = 1
        W[c11_w][col] = 1
        col += 1
        
        # N12 = a31*b32
        U[a31_idx][col] = 1
        V[b32_idx][col] = 1
        W[c12_w][col] = 1
        col += 1
        
        # N21 = a32*b31
        U[a32_idx][col] = 1
        V[b31_idx][col] = 1
        W[c21_w][col] = 1
        col += 1
        
        # N22 = a32*b32
        U[a32_idx][col] = 1
        V[b32_idx][col] = 1
        W[c22_w][col] = 1
        col += 1
        
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
