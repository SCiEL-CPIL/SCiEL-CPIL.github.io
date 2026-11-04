"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x49_decomp():
    """
    Return U, V, W for 4x4x4 matrix multiplication using recursive Strassen.
    U: (16, 49), V: (16, 49), W: (16, 49)
    Indexing: U[i*4+j, c] corresponds to A[i,j]
              V[j*4+k, c] corresponds to B[j,k]  
              W[k*4+i, c] corresponds to C[i,k] (trace convention W[k*n+i])
    
    Actually for 4x4x4: n=m=p=4
    U shape: (n*m, rank) = (16, rank)
    V shape: (m*p, rank) = (16, rank)
    W shape: (p*n, rank) = (16, rank)
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 coefficients
    # For 2x2x2: A is 2x2, B is 2x2, C = A*B is 2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each M is a 2x2x2 Strassen (rank 7)
    # For 4x4x4, we split into 4 2x2 blocks and do 7 such multiplications
    
    # Define the 2x2x2 Strassen decomposition
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=4, (0,1)=5, (1,0)=6, (1,1)=7
    # C indices: (0,0)=8, (0,1)=9, (1,0)=10, (1,1)=11
    
    # For each of the 7 Strassen products, we have a 2x2x2 decomposition
    # The 7 products are:
    # P1 = (A11+A22) * (B11+B22)
    # P2 = (A21+A22) * B11
    # P3 = A11 * (B21-B22)
    # P4 = A22 * (B12-B11)
    # P5 = (A11+A12) * B22
    # P6 = (A21-A11) * (B11+B12)
    # P7 = (A12-A22) * (B21+B22)
    
    # For 4x4, split A into 2x2 blocks:
    # A[0:2, 0:2] = A11, A[0:2, 2:4] = A12
    # A[2:4, 0:2] = A21, A[2:4, 2:4] = A22
    
    # Similarly for B and C.
    
    # The 7 products each involve 2x2 matrices, and each 2x2x2 multiplication
    # can be done with rank 7 (Strassen).
    
    # For product q (q=1..7), we have 2x2 matrices Aq and Bq, and we compute
    # Cq = Aq * Bq using Strassen (rank 7).
    
    # Then C is assembled from the Cq's.
    
    # Let me define the linear combinations for Aq and Bq in terms of original A, B.
    
    # A block indices in 4x4:
    # A11: rows 0-1, cols 0-1 -> indices (0,0),(0,1),(1,0),(1,1)
    # A12: rows 0-1, cols 2-3 -> indices (0,2),(0,3),(1,2),(1,3)
    # A21: rows 2-3, cols 0-1 -> indices (2,0),(2,1),(3,0),(3,1)
    # A22: rows 2-3, cols 2-3 -> indices (2,2),(2,3),(3,2),(3,3)
    
    # For each Strassen product q, define which 2x2 submatrices of A and B are used,
    # and the linear combination coefficients.
    
    # Product 1: A1 = A11 + A22, B1 = B11 + B22
    # Product 2: A2 = A21 + A22, B2 = B11
    # Product 3: A3 = A11, B3 = B21 - B22
    # Product 4: A4 = A22, B4 = B12 - B11
    # Product 5: A5 = A11 + A12, B5 = B22
    # Product 6: A6 = A21 - A11, B6 = B11 + B12
    # Product 7: A7 = A12 - A22, B7 = B21 + B22
    
    # C assembly:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Now, for each product q, the 2x2x2 Strassen decomposition has 7 rank-1 terms.
    # I need to map these to the 4x4x4 indices.
    
    # Let me define helper: for a 2x2x2 Strassen, given the mapping of 2x2 A-block
    # to 4x4 A indices, 2x2 B-block to 4x4 B indices, and 2x2 C-block to 4x4 C indices,
    # produce the U, V, W contributions.
    
    # 2x2x2 Strassen rank-1 terms:
    # For 2x2x2, the 7 rank-1 decompositions are:
    # Term 1: (A00+A11)*(B00+B11) -> contributes to C00, C11
    #   U: A00+A11, V: B00+B11, W: C00+C11
    # Term 2: (A10+A11)*B00 -> contributes to C10, C11
    #   U: A10+A11, V: B00, W: C10+C11
    # Term 3: A00*(B10-B11) -> contributes to C01, C11
    #   U: A00, V: B10-B11, W: C01+C11
    # Term 4: A11*(B01-B00) -> contributes to C00, C10
    #   U: A11, V: B01-B00, W: C00+C10
    # Term 5: (A00+A01)*B11 -> contributes to C00, C01
    #   U: A00+A01, V: B11, W: C00+C01
    # Term 6: (A10-A00)*(B00+B01) -> contributes to C11, C10
    #   U: A10-A00, V: B00+B01, W: C11+C10
    # Term 7: (A01-A11)*(B10+B11) -> contributes to C00, C11
    #   U: A01-A11, V: B10+B11, W: C00+C11
    
    # Let me be more precise. For 2x2x2 Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # So the 7 rank-1 terms are M1..M7, and C is a linear combination.
    
    # For the tensor decomposition, each Mi is a rank-1 term:
    # Mi = (linear combo of A) * (linear combo of B)
    # And C = sum of coefficients * Mi
    
    # In the U,V,W representation:
    # For term Mi: U[row_a, col] = coefficient of A in Mi, V[row_b, col] = coefficient of B in Mi, W[row_c, col] = coefficient of C in Mi
    
    # Let me define the 2x2x2 Strassen decomposition explicitly.
    
    def strassen_2x2_decomp():
        """Return U,V,W for 2x2x2 Strassen. U:(4,7), V:(4,7), W:(4,7)"""
        n2 = m2 = p2 = 2
        r2 = 7
        U2 = np.zeros((n2 * m2, r2))
        V2 = np.zeros((m2 * p2, r2))
        W2 = np.zeros((p2 * n2, r2))
        
        # A indices: A00=0, A01=1, A10=2, A11=3
        # B indices: B00=0, B01=1, B10=2, B11=3
        # C indices: C00=0, C01=1, C10=2, C11=3 (but W uses k*n+i, so C[i,k] -> W[k*2+i])
        # C[i,k]: i=0,1; k=0,1. W[k*2+i]
        # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
        
        # M1 = (A00+A11)*(B00+B11)
        # A part: A00+A11 -> U[0]=1, U[3]=1
        # B part: B00+B11 -> V[0]=1, V[3]=1
        # C part: contributes to C00 (+1) and C11 (+1)
        # C00 = W[0], C11 = W[3]
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        # A part: A10+A11 -> U[2]=1, U[3]=1
        # B part: B00 -> V[0]=1
        # C part: contributes to C10 (+1) and C11 (-1)
        # C10 = W[2], C11 = W[3]
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = -1
        
        # M3 = A00*(B10-B11)
        # A part: A00 -> U[0]=1
        # B part: B10-B11 -> V[2]=1, V[3]=-1
        # C part: contributes to C01 (+1) and C11 (+1)
        # C01 = W[1], C11 = W[3]
        U2[0, 2] = 1
        V2[2, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = 1
        
        # M4 = A11*(B01-B00)
        # A part: A11 -> U[3]=1
        # B part: B01-B00 -> V[1]=1, V[0]=-1
        # C part: contributes to C00 (+1) and C10 (+1)
        # C00 = W[0], C10 = W[2]
        U2[3, 3] = 1
        V2[1, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        # A part: A00+A01 -> U[0]=1, U[1]=1
        # B part: B11 -> V[3]=1
        # C part: contributes to C00 (-1) and C01 (+1)
        # C00 = W[0], C01 = W[1]
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        # A part: A10-A00 -> U[2]=1, U[0]=-1
        # B part: B00+B01 -> V[0]=1, V[1]=1
        # C part: contributes to C11 (+1)
        # C11 = W[3]
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        # A part: A01-A11 -> U[1]=1, U[3]=-1
        # B part: B10+B11 -> V[2]=1, V[3]=1
        # C part: contributes to C00 (+1)
        # C00 = W[0]
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        return U2, V2, W2
    
    U2, V2, W2 = strassen_2x2_decomp()
    
    # Now for 4x4x4, we have 7 products, each a 2x2x2 Strassen.
    # For product q (q=0..6), we need to define:
    # - Which 2x2 block of A (in 4x4) is used, and how
    # - Which 2x2 block of B (in 4x4) is used, and how
    # - Which 2x2 block of C (in 4x4) is the result
    
    # A blocks in 4x4 (row, col) of 2x2:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # For each product q, define the linear combination of A blocks:
    # q=0: A11 + A22
    # q=1: A21 + A22
    # q=2: A11
    # q=3: A22
    # q=4: A11 + A12
    # q=5: A21 - A11
    # q=6: A12 - A22
    
    # B blocks:
    # q=0: B11 + B22
    # q=1: B11
    # q=2: B21 - B22
    # q=3: B12 - B11
    # q=4: B22
    # q=5: B11 + B12
    # q=6: B21 + B22
    
    # C blocks (result):
    # q=0: C11 (rows 0-1, cols 0-1)
    # q=1: C21 (rows 2-3, cols 0-1)
    # q=2: C12 (rows 0-1, cols 2-3)
    # q=3: C22 (rows 2-3, cols 2-3)
    # q=4: C22
    # q=5: C21
    # q=6: C12
    
    # Wait, that's not right. Let me re-derive.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # So:
    # P1 (q=0) contributes to C11 (+1) and C22 (+1)
    # P2 (q=1) contributes to C21 (+1) and C22 (-1)
    # P3 (q=2) contributes to C12 (+1) and C22 (+1)
    # P4 (q=3) contributes to C11 (+1) and C21 (+1)
    # P5 (q=4) contributes to C11 (-1) and C12 (+1)
    # P6 (q=5) contributes to C22 (+1)
    # P7 (q=6) contributes to C11 (+1)
    
    # For each product q, the 2x2 result Cq is placed in specific 2x2 blocks of the 4x4 C.
    # Actually, each Pq is a 2x2 matrix, and it contributes to the 4x4 C by being added
    # to specific 2x2 subblocks.
    
    # P1 is 2x2, added to C[0:2,0:2] (C11) and C[2:4,2:4] (C22)
    # P2 is 2x2, added to C[2:4,0:2] (C21) with +1, C[2:4,2:4] (C22) with -1
    # P3 is 2x2, added to C[0:2,2:4] (C12) with +1, C[2:4,2:4] (C22) with +1
    # P4 is 2x2, added to C[0:2,0:2] (C11) with +1, C[2:4,0:2] (C21) with +1
    # P5 is 2x2, added to C[0:2,0:2] (C11) with -1, C[0:2,2:4] (C12) with +1
    # P6 is 2x2, added to C[2:4,2:4] (C22) with +1
    # P7 is 2x2, added to C[0:2,0:2] (C11) with +1
    
    # Now I need to build U, V, W for the full 4x4x4.
    # For each product q (0-6), and each of the 7 Strassen terms within it,
    # I add contributions to U, V, W.
    
    # A block mapping: for a 2x2 block at position (br, bc) in 4x4,
    # A[br*2+i, bc*2+j] corresponds to A2[i,j] in the 2x2x2.
    # U index for A[i,j] in 4x4 is i*4+j.
    
    # Similarly for B: B[br*2+i, bc*2+j], V index is (br*2+i)*4 + (bc*2+j)? 
    # Wait, V has shape (m*p, rank) = (4*4, rank) for 4x4x4.
    # B is 4x4, B[j,k]. V index is j*4+k.
    
    # For C: C[i,k], W index is k*4+i (trace convention W[k*n+i] with n=4).
    
    # Let me define the block positions:
    # A blocks: (br, bc) where br, bc in {0,1}
    #   A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
    # B blocks: same
    # C blocks: same
    
    # For product q, the A-side is a linear combination of A blocks.
    # I'll represent this as a dict: {(br, bc): coeff}
    
    A_blocks = {
        0: {((0,0)): 1, (1,1): 1},  # A11 + A22
        1: {((1,0)): 1, (1,1): 1},  # A21 + A22
        2: {((0,0)): 1},             # A11
        3: {((1,1)): 1},             # A22
        4: {((0,0)): 1, (0,1): 1},  # A11 + A12
        5: {((1,0)): 1, (0,0): -1}, # A21 - A11
        6: {((0,1)): 1, (1,1): -1}, # A12 - A22
    }
    
    B_blocks = {
        0: {((0,0)): 1, (1,1): 1},  # B11 + B22
        1: {((0,0)): 1},             # B11
        2: {((1,0)): 1, (1,1): -1}, # B21 - B22
        3: {((0,1)): 1, (0,0): -1}, # B12 - B11
        4: {((1,1)): 1},             # B22
        5: {((0,0)): 1, (0,1): 1},  # B11 + B12
        6: {((1,0)): 1, (1,1): 1},  # B21 + B22
    }
    
    # C contributions: for product q, which C blocks and with what coefficients
    # C block (br, bc) corresponds to C[br*2:br*2+2, bc*2:bc*2+2]
    C_contribs = {
        0: {((0,0)): 1, (1,1): 1},  # C11 + C22
        1: {((1,0)): 1, (1,1): -1}, # C21 - C22
        2: {((0,1)): 1, (1,1): 1},  # C12 + C22
        3: {((0,0)): 1, (1,0): 1},  # C11 + C21
        4: {((0,0)): -1, (0,1): 1}, # -C11 + C12
        5: {((1,1)): 1},             # C22
        6: {((0,0)): 1},             # C11
    }
    
    # Now build U, V, W
    for q in range(7):
        base_col = q * 7  # 7 columns per product
        
        # For each of the 7 Strassen terms in this product
        for t in range(7):
            col = base_col + t
            
            # U contribution: for each (i,j) in 2x2 A, U2[i*2+j, t] * A_block_coeff
            for (br, bc), coeff_a in A_blocks[q].items():
                for i in range(2):
                    for j in range(2):
                        a_idx = (br * 2 + i) * 4 + (bc * 2 + j)
                        U[a_idx, col] += U2[i * 2 + j, t] * coeff_a
            
            # V contribution
            for (br, bc), coeff_b in B_blocks[q].items():
                for i in range(2):
                    for j in range(2):
                        b_idx = (br * 2 + i) * 4 + (bc * 2 + j)
                        V[b_idx, col] += V2[i * 2 + j, t] * coeff_b
            
            # W contribution
            for (br, bc), coeff_c in C_contribs[q].items():
                for i in range(2):
                    for k in range(2):
                        # C[i,k] in the 2x2 block at (br, bc)
                        # Full C index: (br*2+i, bc*2+k)
                        # W index: k_full * 4 + i_full where k_full = bc*2+k, i_full = br*2+i
                        w_idx = (bc * 2 + k) * 4 + (br * 2 + i)
                        # In 2x2, C[i,k] has W2 index k*2+i
                        W[w_idx, col] += W2[k * 2 + i, t] * coeff_c
    
    return U, V, W


def _schoolbook_4x4x1_decomp():
    """
    Return U, V, W for 4x4x1 matrix multiplication (A 4x4 times B 4x1).
    U: (16, 16), V: (4, 16), W: (4, 16)
    """
    n = m = 4
    p = 1
    rank = n * m * p  # 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Split the 9 columns of B into: 4 + 4 + 1
    # First 4 columns: 4x4x4 Strassen (rank 49)
    # Next 4 columns: 4x4x4 Strassen (rank 49)
    # Last 1 column: 4x4x1 schoolbook (rank 16)
    # Total rank: 49 + 49 + 16 = 114
    
    if time.time() > deadline:
        # Fallback to schoolbook if time is up
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Get Strassen 4x4x4 decomposition
    U_s1, V_s1, W_s1 = _strassen_2x2x49_decomp()
    U_s2, V_s2, W_s2 = _strassen_2x2x49_decomp()
    
    # Get schoolbook 4x4x1 decomposition
    U_sb, V_sb, W_sb = _schoolbook_4x4x1_decomp()
    
    # Now assemble for 4x4x9
    # B has 9 columns: cols 0-3, 4-7, 8
    # For the first 4 columns (cols 0-3), use Strassen 1
    # For the next 4 columns (cols 4-7), use Strassen 2
    # For the last column (col 8), use schoolbook
    
    # U is shared: A is the same 4x4 matrix, so U is the same for all blocks
    # But we need to adjust V and W for the different column ranges.
    
    # U shape: (16, 114)
    # V shape: (36, 114) -- m*p = 4*9 = 36
    # W shape: (36, 114) -- p*n = 9*4 = 36
    
    total_rank = 49 + 49 + 16
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    # Block 1: columns 0-3 of B, Strassen 4x4x4
    # U: same A, so U_full[:, 0:49] = U_s1
    U_full[:, 0:49] = U_s1
    
    # V: B columns 0-3. In V_s1, V index is j*4+k where j in 0-3, k in 0-3.
    # In V_full, V index is j*9+k where j in 0-3, k in 0-8.
    # For block 1, k ranges 0-3, so V_full[j*9+k, 0:49] = V_s1[j*4+k, 0:49]
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + k, 0:49] = V_s1[j * 4 + k, 0:49]
    
    # W: C columns 0-3. In W_s1, W index is k*4+i where k in 0-3, i in 0-3.
    # In W_full, W index is k*4+i where k in 0-8, i in 0-3.
    # For block 1, k ranges 0-3, so W_full[k*4+i, 0:49] = W_s1[k*4+i, 0:49]
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, 0:49] = W_s1[k * 4 + i, 0:49]
    
    # Block 2: columns 4-7 of B, Strassen 4x4x4
    # U: same A
    U_full[:, 49:98] = U_s2
    
    # V: B columns 4-7. V_s2 index j*4+k (k in 0-3) maps to V_full j*9+(k+4)
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + (k + 4), 49:98] = V_s2[j * 4 + k, 49:98]
    
    # W: C columns 4-7. W_s2 index k*4+i (k in 0-3) maps to W_full (k+4)*4+i
    for k in range(4):
        for i in range(4):
            W_full[(k + 4) * 4 + i, 49:98] = W_s2[k * 4 + i, 49:98]
    
    # Block 3: column 8 of B, schoolbook 4x4x1
    # U: same A
    U_full[:, 98:114] = U_sb
    
    # V: B column 8. V_sb index j*1+k (k=0) maps to V_full j*9+8
    for j in range(4):
        for k in range(1):
            V_full[j * 9 + (8 + k), 98:114] = V_sb[j * 1 + k, 98:114]
    
    # W: C column 8. W_sb index k*4+i (k=0) maps to W_full 8*4+i
    for k in range(1):
        for i in range(4):
            W_full[(8 + k) * 4 + i, 98:114] = W_sb[k * 4 + i, 98:114]
    
    if time.time() > deadline:
        # Still return what we have
        pass
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END

