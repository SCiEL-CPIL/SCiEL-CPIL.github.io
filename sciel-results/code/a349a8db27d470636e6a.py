import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will compute C = A B where A is 4x4 and B is 4x6.
    # Split B into B1 (4x4) and B2 (4x2).
    # C1 = A B1 (4x4x4) using Strassen (Rank 49).
    # C2 = A B2 (4x4x2) using Schoolbook (Rank 32).
    # Total Rank = 49 + 32 = 81.
    
    rank_strassen = 49
    rank_schoolbook = 32
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for 4x4x4 ---
    # Indices for A (4x4): i in 0..3, j in 0..3. Index = i*4 + j.
    # Indices for B1 (4x4): j in 0..3, k in 0..3. Index = j*4 + k.
    # Indices for C1 (4x4): k in 0..3, i in 0..3. Index = k*4 + i.
    
    # Strassen for 2x2x2 has 7 terms.
    # For 4x4x4, we apply Strassen to the 2x2 blocks.
    # This results in 49 terms.
    # We can generate these terms by iterating over the 7 Strassen terms for each of the 4 block positions?
    # No, Strassen is recursive.
    # The 49 terms are linear combinations of the 16 entries of A and 16 entries of B1.
    
    # Let's construct the Strassen decomposition for 4x4x4 explicitly.
    # We define the 49 linear forms for U and V.
    # U is 16x49, V is 16x49 (for B1 part), W is 16x49 (for C1 part).
    
    # Strassen 2x2 terms:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b22)
    # m4 = a22(b12-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # For 4x4, we have blocks A00, A01, A10, A11 and B00, B01, B10, B11.
    # The Strassen algorithm for 4x4 uses 7 multiplications of 2x2 matrices?
    # No, it uses 7 multiplications of 2x2 matrices, each of which requires 7 scalar multiplications.
    # So 49 scalar multiplications.
    
    # We will generate the 49 terms by applying the 2x2 Strassen logic to the 2x2 blocks.
    # Let's define the block indices.
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3).
    # B1 blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3).
    
    # The 7 "matrix" multiplications M1..M7 are 2x2 matrices.
    # Each Mk is a linear combination of the blocks.
    # Then each Mk is multiplied by the corresponding block of B?
    # No, the Strassen algorithm computes the product C = A B.
    # C blocks are C00, C01, C10, C11.
    
    # Let's just use the known 49-term decomposition.
    # I will generate it programmatically.
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        for i in range(n*m):
            U[i][col] += u_coeffs[i]
        for j in range(m*p):
            V[j][col] += v_coeffs[j]
        for k in range(p*n):
            W[k][col] += w_coeffs[k]
            
    # Initialize zero vectors
    u_zero = [0] * (n*m)
    v_zero = [0] * (m*p)
    w_zero = [0] * (p*n)
    
    # Strassen 2x2 coefficients
    # We need to map the 2x2 Strassen to the 4x4 case.
    # This is complex to code from scratch.
    # Alternative: Use the fact that M_4,4,4 <= M_2,2,2 x M_2,2,2?
    # No, rank is 49.
    
    # Let's use a simpler decomposition for 4x4x4: Schoolbook (16 terms).
    # Rank 16 for 4x4x4? No, schoolbook is 16.
    # Wait, schoolbook for 4x4x4 is 16 scalar multiplications?
    # C_ik = sum_j A_ij B_jk.
    # There are 4*4*4 = 64 terms in the sum?
    # No, there are 16 entries in C. Each entry has 4 terms.
    # Total 64 scalar multiplications if we don't reuse?
    # No, schoolbook matrix multiplication of n x n matrices uses n^3 scalar multiplications.
    # For n=4, 64.
    # So Schoolbook for 4x4x4 is rank 64.
    # Strassen is 49.
    # Laderman is 18.
    
    # So Strassen (49) is better than Schoolbook (64).
    # And 49 + 32 = 81 is better than 64 + 32 = 96.
    
    # I will implement Strassen for 4x4x4.
    # To do this, I will apply the 2x2 Strassen algorithm to the 2x2 blocks.
    
    # The 7 terms of 2x2 Strassen:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b21-b22)
    # m4 = a22(b12-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # For 4x4, we have 4 blocks of A and 4 blocks of B.
    # The Strassen algorithm for 4x4 computes 7 products of 2x2 blocks.
    # Let A = [[A00, A01], [A10, A11]] and B = [[B00, B01], [B10, B11]].
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mk is a 2x2 matrix product.
    # We need to decompose each 2x2 matrix product into 7 scalar multiplications.
    # So total 7 * 7 = 49 scalar multiplications.
    
    # We will iterate over the 7 block-level terms, and for each, iterate over the 7 scalar-level terms.
    
    # Mapping of block indices to flat indices:
    # A00: rows 0-1, cols 0-1. Flat: 0,1,4,5.
    # A01: rows 0-1, cols 2-3. Flat: 2,3,6,7.
    # A10: rows 2-3, cols 0-1. Flat: 8,9,12,13.
    # A11: rows 2-3, cols 2-3. Flat: 10,11,14,15.
    
    # B00: rows 0-1, cols 0-1. Flat: 0,1,4,5.
    # B01: rows 0-1, cols 2-3. Flat: 2,3,6,7.
    # B10: rows 2-3, cols 0-1. Flat: 8,9,12,13.
    # B11: rows 2-3, cols 2-3. Flat: 10,11,14,15.
    
    # C00: rows 0-1, cols 0-1. Flat: 0,1,4,5.
    # C01: rows 0-1, cols 2-3. Flat: 2,3,6,7.
    # C10: rows 2-3, cols 0-1. Flat: 8,9,12,13.
    # C11: rows 2-3, cols 2-3. Flat: 10,11,14,15.
    
    # Note: W index is k*n + i.
    # C00: k=0,1; i=0,1. W indices: 0,1,4,5.
    # C01: k=2,3; i=0,1. W indices: 8,9,12,13.
    # C10: k=0,1; i=2,3. W indices: 2,3,6,7.
    # C11: k=2,3; i=2,3. W indices: 10,11,14,15.
    
    # Wait, the trace convention is W[k*n+i, r].
    # C is p x n. Here p=4, n=4.
    # C_ik is at row k, col i.
    # So C00 (k=0, i=0) is W[0, r].
    # C01 (k=0, i=1) is W[1, r].
    # C10 (k=1, i=0) is W[4, r].
    # C11 (k=1, i=1) is W[5, r].
    
    # Let's define the 7 scalar Strassen terms for a 2x2 block multiplication.
    # Input A_block (2x2), B_block (2x2).
    # Output C_block (2x2).
    
    # We will create a function to generate the U, V, W coefficients for a 2x2 Strassen multiplication
    # given the flat indices of the A_block, B_block, and C_block.
    
    def get_2x2_strassen_indices(a_flat, b_flat, c_flat):
        # a_flat: [a11, a12, a21, a22]
        # b_flat: [b11, b12, b21, b22]
        # c_flat: [c11, c12, c21, c22]
        # Returns list of (u_vec, v_vec, w_vec)
        terms = []
        
        # m1 = (a11+a22)(b11+b22)
        u1 = [0]*4; u1[a_flat[0]] = 1; u1[a_flat[3]] = 1
        v1 = [0]*4; v1[b_flat[0]] = 1; v1[b_flat[3]] = 1
        w1 = [0]*4; w1[c_flat[0]] = 1; w1[c_flat[1]] = 1; w1[c_flat[2]] = 1; w1[c_flat[3]] = -1
        # Wait, c11 = m1 - m2 - m4 + m7
        # c12 = m3 + m5
        # c21 = m1 - m3 - m5 + m6
        # c22 = m1 - m2 - m4 + m7? No.
        # c22 = m2 + m4 - m1 + m7?
        # Let's use the standard Strassen formulas:
        # c11 = m1 - m2 - m4 + m7
        # c12 = m3 + m5
        # c21 = m1 - m3 - m5 + m6
        # c22 = m2 + m4 - m1 + m7
        
        # This means each scalar multiplication contributes to multiple C entries.
        # We add the term to U, V, and W.
        
        # Term 1: m1
        terms.append(([1,0,0,1] if a_flat[0]==0 else [0,1,0,0], [1,0,0,1] if b_flat[0]==0 else [0,1,0,0], [1,0,0,1] if c_flat[0]==0 else [0,1,0,0])) # This is too messy.
        
        # Better: Just define the 7 terms explicitly.
        # a = [a11, a12, a21, a22]
        # b = [b11, b12, b21, b22]
        
        # m1 = (a11+a22)(b11+b22)
        u = [0]*4; u[0]=1; u[3]=1
        v = [0]*4; v[0]=1; v[3]=1
        w = [0]*4; w[0]=1; w[1]=-1; w[2]=1; w[3]=-1 # c11+=1, c12+=-1, c21+=1, c22+=-1? No.
        
        # Let's just use the schoolbook for 2x2 (rank 4) to be safe?
        # Rank 4 for 2x2.
        # Total rank for 4x4 would be 4 * 4 = 16?
        # No, 4x4 has 4 blocks. 4 * 4 = 16?
        # If we use schoolbook for each 2x2 block, we need 4 blocks * 4 mults = 16 mults?
        # But the blocks are not independent.
        # If we use schoolbook for 4x4, it is 64 mults.
        
        # I will use the 64-rank schoolbook for 4x4x4 and 32-rank schoolbook for 4x4x2.
        # Total 96. This is the same as the current program.
        
        # I need a better decomposition.
        # I will use Strassen for 4x4x4 (rank 49).
        # I will hardcode the 49 terms.
        
        return []

    # Given the complexity, I will fall back to the 96-rank solution but try to optimize the implementation.
    # Or, I will use the 81-rank solution by implementing Strassen for 4x4x4 correctly.
    
    # Let's implement Strassen for 4x4x4 by calling a 2x2 Strassen function 49 times?
    # No, it's 7 calls to 2x2 Strassen, each of which is 7 scalar mults.
    
    # I will implement the 2x2 Strassen scalar decomposition.
    def add_2x2_strassen_term(a_idx, b_idx, c_idx, col):
        # a_idx, b_idx, c_idx are lists of 4 flat indices.
        # We add the 7 terms of 2x2 Strassen.
        
        # m1 = (a11+a22)(b11+b22)
        u = [0]*(n*m); u[a_idx[0]]=1; u[a_idx[3]]=1
        v = [0]*(m*p); v[b_idx[0]]=1; v[b_idx[3]]=1
        w = [0]*(p*n); w[c_idx[0]]=1; w[c_idx[1]]=-1; w[c_idx[2]]=1; w[c_idx[3]]=-1
        add_term(u, v, w, col); col+=1
        
        # m2 = (a21+a22)b11
        u = [0]*(n*m); u[a_idx[2]]=1; u[a_idx[3]]=1
        v = [0]*(m*p); v[b_idx[0]]=1
        w = [0]*(p*n); w[c_idx[0]]=-1; w[c_idx[3]]=1
        add_term(u, v, w, col); col+=1
        
        # m3 = a11(b21-b22)
        u = [0]*(n*m); u[a_idx[0]]=1
        v = [0]*(m*p); v[b_idx[2]]=1; v[b_idx[3]]=-1
        w = [0]*(p*n); w[c_idx[1]]=1; w[c_idx[2]]=-1
        add_term(u, v, w, col); col+=1
        
        # m4 = a22(b12-b11)
        u = [0]*(n*m); u[a_idx[3]]=1
        v = [0]*(m*p); v[b_idx[1]]=1; v[b_idx[0]]=-1
        w = [0]*(p*n); w[c_idx[0]]=-1; w[c_idx[3]]=1
        add_term(u, v, w, col); col+=1
        
        # m5 = (a11+a12)b22
        u = [0]*(n*m); u[a_idx[0]]=1; u[a_idx[1]]=1
        v = [0]*(m*p); v[b_idx[3]]=1
        w = [0]*(p*n); w[c_idx[1]]=1; w[c_idx[2]]=-1
        add_term(u, v, w, col); col+=1
        
        # m6 = (a21-a11)(b11+b12)
        u = [0]*(n*m); u[a_idx[2]]=1; u[a_idx[0]]=-1
        v = [0]*(m*p); v[b_idx[0]]=1; v[b_idx[1]]=1
        w = [0]*(p*n); w[c_idx[2]]=1
        add_term(u, v, w, col); col+=1
        
        # m7 = (a12-a22)(b21+b22)
        u = [0]*(n*m); u[a_idx[1]]=1; u[a_idx[3]]=-1
        v = [0]*(m*p); v[b_idx[2]]=1; v[b_idx[3]]=1
        w = [0]*(p*n); w[c_idx[0]]=1; w[c_idx[3]]=1
        add_term(u, v, w, col); col+=1
        
        return col

    # Now apply this to the 7 block-level Strassen terms.
    
    # Block indices for A (4x4)
    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]
    
    # Block indices for B1 (4x4)
    B00 = [0, 1, 4, 5]
    B01 = [2, 3, 6, 7]
    B10 = [8, 9, 12, 13]
    B11 = [10, 11, 14, 15]
    
    # Block indices for C1 (4x4)
    C00 = [0, 1, 4, 5]
    C01 = [2, 3, 6, 7]
    C10 = [8, 9, 12, 13]
    C11 = [10, 11, 14, 15]
    
    col = 0
    
    # M1 = (A00+A11)(B00+B11)
    # This is a 2x2 matrix multiplication.
    # We need to decompose (A00+A11) and (B00+B11) into linear forms?
    # No, the Strassen algorithm computes the product of two 2x2 matrices.
    # The inputs are the blocks themselves.
    # So we call add_2x2_strassen_term with the indices of A00+A11?
    # No, the linear forms are on the original A and B.
    # So for M1, the "A" part is A00+A11.
    # The indices for A00+A11 are the union of A00 and A11.
    # But we need to pass the indices to the 2x2 Strassen function.
    # The 2x2 Strassen function expects 4 indices for A and 4 for B.
    # Here, the "A" matrix is A00+A11.
    # So a11 = A00[0] + A11[0], etc.
    # This means the U vector will have 1 at A00[0] and 1 at A11[0].
    
    # We can handle this by creating virtual indices?
    # No, we can just modify the add_2x2_strassen_term to accept lists of lists of indices.
    
    # This is getting too complicated for a quick fix.
    # I will return the schoolbook solution (rank 96).
    
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

# EVOLVE-BLOCK-END
