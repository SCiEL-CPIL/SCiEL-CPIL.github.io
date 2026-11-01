"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    # Target: Rank 22 decomposition for 2x3 * 3x4
    # Split middle dimension m=3 into 1 and 2.
    # Part 1: 2x1 * 1x4 -> Rank 8
    # Part 2: 2x2 * 2x4 -> Split 4 into 2+2 -> 2 * (2x2 * 2x2) -> 2 * Rank 7 = 14
    # Total Rank: 8 + 14 = 22
    
    n, m, p = 2, 3, 4
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: M_{2,1,4}
    # A indices: i in [0,1], j=0. Index in U: i*m + 0 = i*3
    # B indices: j=0, k in [0,3]. Index in V: 0*p + k = k
    # C indices: i in [0,1], k in [0,3]. Index in W: k*n + i = 2i + k
    
    for i in range(2):
        for k in range(4):
            u_idx = i * 3 + 0
            v_idx = 0 * 4 + k
            w_idx = k * 2 + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2: M_{2,2,4}
    # A indices: i in [0,1], j in [1,2]. Index in U: i*3 + j
    # B indices: j in [1,2], k in [0,3]. Index in V: j*4 + k
    # C indices: i in [0,1], k in [0,3]. Index in W: k*2 + i
    
    # Split B into two 2x2 blocks: k in [0,1] and k in [2,3]
    # For each block, use Strassen (Rank 7)
    
    def add_strassen_term(a_coeffs, b_coeffs, c_coeffs, offset):
        # a_coeffs: list of (u_idx, coeff)
        # b_coeffs: list of (v_idx, coeff)
        # c_coeffs: list of (w_idx, coeff)
        # This is not quite right, Strassen involves sums.
        # I will explicitly write out the 7 terms for one 2x2*2x2 block.
        pass

    # Strassen for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We need to map these to U, V, W.
    # U rows correspond to A entries.
    # V rows correspond to B entries.
    # W rows correspond to C entries.
    
    # For a block of B with columns k1, k2 (local 0, 1)
    # B entries: e=B[j1, k1], f=B[j1, k2], g=B[j2, k1], h=B[j2, k2]
    
    def process_strassen_block(j1, j2, k1, k2, start_col):
        # j1, j2 are rows of B (and cols of A)
        # k1, k2 are cols of B (and cols of C)
        
        # A entries:
        # a = A[i1, j1] -> u_idx = i1*3 + j1
        # b = A[i1, j2] -> u_idx = i1*3 + j2
        # c = A[i2, j1] -> u_idx = i2*3 + j1
        # d = A[i2, j2] -> u_idx = i2*3 + j2
        
        # B entries:
        # e = B[j1, k1] -> v_idx = j1*4 + k1
        # f = B[j1, k2] -> v_idx = j1*4 + k2
        # g = B[j2, k1] -> v_idx = j2*4 + k1
        # h = B[j2, k2] -> v_idx = j2*4 + k2
        
        # C entries:
        # C[i1, k1], C[i1, k2], C[i2, k1], C[i2, k2]
        # w_idx = k*2 + i
        
        c = start_col
        
        # M1 = (a+d)(e+h)
        # U: a(1), d(1) -> But U is linear. We need to represent the product.
        # The decomposition is a sum of rank-1 tensors.
        # Term 1: (a+d) * (e+h) * (contribution to C)
        # This is not a single rank-1 term in the original basis.
        # Strassen uses intermediate products.
        # The tensor decomposition of Strassen's algorithm has rank 7.
        # The 7 terms are the products M1..M7.
        # Each Mr is a product of a linear combination of A and a linear combination of B.
        # So for each Mr, we add one column to U, V, W.
        # U column: coefficients for A in the linear combination.
        # V column: coefficients for B in the linear combination.
        # W column: coefficients for C in the result.
        
        # M1 = (a+d)(e+h)
        # U: a=1, d=1
        # V: e=1, h=1
        # W: C11 += 1, C12 += 1, C21 += 1, C22 -= 1 ? 
        # Recall:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1:
        # W: C11=1, C12=0, C21=0, C22=1
        
        # Let's define the 7 terms explicitly.
        
        # Term 1: M1
        u_m1 = {i1*3+j1: 1, i2*3+j2: 1} # a+d
        v_m1 = {j1*4+k1: 1, j2*4+k2: 1} # e+h
        w_m1 = {k1*2+i1: 1, k2*2+i1: 0, k1*2+i2: 0, k2*2+i2: 1} # C11, C22
        
        for idx, val in u_m1.items(): U[idx][c] = val
        for idx, val in v_m1.items(): V[idx][c] = val
        for idx, val in w_m1.items(): W[idx][c] = val
        c += 1
        
        # Term 2: M2 = (c+d)e
        u_m2 = {i2*3+j1: 1, i2*3+j2: 1} # c+d
        v_m2 = {j1*4+k1: 1} # e
        w_m2 = {k1*2+i2: 1} # C21
        
        for idx, val in u_m2.items(): U[idx][c] = val
        for idx, val in v_m2.items(): V[idx][c] = val
        for idx, val in w_m2.items(): W[idx][c] = val
        c += 1
        
        # Term 3: M3 = a(f-h)
        u_m3 = {i1*3+j1: 1} # a
        v_m3 = {j1*4+k2: 1, j2*4+k2: -1} # f-h
        w_m3 = {k2*2+i1: 1, k2*2+i2: 1} # C12, C22
        
        for idx, val in u_m3.items(): U[idx][c] = val
        for idx, val in v_m3.items(): V[idx][c] = val
        for idx, val in w_m3.items(): W[idx][c] = val
        c += 1
        
        # Term 4: M4 = d(g-e)
        u_m4 = {i2*3+j2: 1} # d
        v_m4 = {j2*4+k1: 1, j1*4+k1: -1} # g-e
        w_m4 = {k1*2+i1: 1, k1*2+i2: 1} # C11, C21
        
        for idx, val in u_m4.items(): U[idx][c] = val
        for idx, val in v_m4.items(): V[idx][c] = val
        for idx, val in w_m4.items(): W[idx][c] = val
        c += 1
        
        # Term 5: M5 = (a+b)h
        u_m5 = {i1*3+j1: 1, i1*3+j2: 1} # a+b
        v_m5 = {j2*4+k2: 1} # h
        w_m5 = {k2*2+i1: 1, k1*2+i1: -1} # C12, -C11
        
        for idx, val in u_m5.items(): U[idx][c] = val
        for idx, val in v_m5.items(): V[idx][c] = val
        for idx, val in w_m5.items(): W[idx][c] = val
        c += 1
        
        # Term 6: M6 = (c-a)(e+f)
        u_m6 = {i2*3+j1: 1, i1*3+j1: -1} # c-a
        v_m6 = {j1*4+k1: 1, j1*4+k2: 1} # e+f
        w_m6 = {k2*2+i2: 1} # C22
        
        for idx, val in u_m6.items(): U[idx][c] = val
        for idx, val in v_m6.items(): V[idx][c] = val
        for idx, val in w_m6.items(): W[idx][c] = val
        c += 1
        
        # Term 7: M7 = (b-d)(g+h)
        u_m7 = {i1*3+j2: 1, i2*3+j2: -1} # b-d
        v_m7 = {j2*4+k1: 1, j2*4+k2: 1} # g+h
        w_m7 = {k1*2+i1: 1} # C11
        
        for idx, val in u_m7.items(): U[idx][c] = val
        for idx, val in v_m7.items(): V[idx][c] = val
        for idx, val in w_m7.items(): W[idx][c] = val
        c += 1
        
        return c

    # i1=0, i2=1
    # j1=1, j2=2 (A cols 1,2)
    
    # Block 1: k1=0, k2=1
    col = process_strassen_block(1, 2, 0, 1, col)
    
    # Block 2: k1=2, k2=3
    col = process_strassen_block(1, 2, 2, 3, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
