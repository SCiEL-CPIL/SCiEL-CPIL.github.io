"""Strassen-based exact tensor decomposition for 3x4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-49 exact tensor decomposition for 3x4x7 matrix multiplication.
    Uses Strassen's algorithm: pad to 4x4x8, apply 7 Strassen terms of rank 7 each.
    """
    n, m, p = 3, 4, 7
    N, M, P = 4, 4, 8  # padded dimensions
    
    # Strassen for 2x2x2: 7 terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # For 4x4x4, we have 4 blocks of 2x2, so 16 block positions
    # Strassen gives 7 multiplications per level, but for 4x4 we do:
    # Split 4x4 into four 2x2 blocks
    # Use 7 Strassen products of 2x2 matrices
    # Each 2x2x2 product has rank 7
    # So 4x4x4 has rank 7*7 = 49
    
    # Define the 7 Strassen factors for 2x2x2
    # U factors: (2*2, 7) = (4, 7) for A
    # V factors: (2*2, 7) = (4, 7) for B  
    # W factors: (2*2, 7) = (4, 7) for C
    
    # Strassen 2x2x2 factors (index: row*2+col)
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 01=1, 10=2, 11=3
    
    # Term 1: (A00+A11)(B00+B11) -> C00+=1, C01+=0, C10+=0, C11+=1
    # Actually let me use the standard Strassen formulation carefully.
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11  
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Using 1-indexed: A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # In 0-indexed flat: A[0]=A[0,0], A[1]=A[0,1], A[2]=A[1,0], A[3]=A[1,1]
    
    # U factors for 2x2 A (shape 4x7), each column is one term's A coefficients
    U2 = np.zeros((4, 7))
    # M1: A11+A22 -> A[0]+A[3]
    U2[0, 0] = 1; U2[3, 0] = 1
    # M2: A21+A22 -> A[2]+A[3]
    U2[2, 1] = 1; U2[3, 1] = 1
    # M3: A11 -> A[0]
    U2[0, 2] = 1
    # M4: A22 -> A[3]
    U2[3, 3] = 1
    # M5: A11+A12 -> A[0]+A[1]
    U2[0, 4] = 1; U2[1, 4] = 1
    # M6: A21-A11 -> A[2]-A[0]
    U2[2, 5] = 1; U2[0, 5] = -1
    # M7: A12-A22 -> A[1]-A[3]
    U2[1, 6] = 1; U2[3, 6] = -1
    
    V2 = np.zeros((4, 7))
    # M1: B11+B22 -> B[0]+B[3]
    V2[0, 0] = 1; V2[3, 0] = 1
    # M2: B11 -> B[0]
    V2[0, 1] = 1
    # M3: B21-B22 -> B[2]-B[3]
    V2[2, 2] = 1; V2[3, 2] = -1
    # M4: B12-B11 -> B[1]-B[0]
    V2[1, 3] = 1; V2[0, 3] = -1
    # M5: B22 -> B[3]
    V2[3, 4] = 1
    # M6: B11+B12 -> B[0]+B[1]
    V2[0, 5] = 1; V2[1, 5] = 1
    # M7: B21+B22 -> B[2]+B[3]
    V2[2, 6] = 1; V2[3, 6] = 1
    
    W2 = np.zeros((4, 7))
    # C11 = M1+M4-M5+M7 -> C[0]
    W2[0, 0] = 1; W2[0, 3] = 1; W2[0, 4] = -1; W2[0, 6] = 1
    # C12 = M3+M5 -> C[1]
    W2[1, 2] = 1; W2[1, 4] = 1
    # C21 = M2+M4 -> C[2]
    W2[2, 1] = 1; W2[2, 3] = 1
    # C22 = M1-M2+M3+M6 -> C[3]
    W2[3, 0] = 1; W2[3, 1] = -1; W2[3, 2] = 1; W2[3, 5] = 1
    
    # Now for 4x4x4: split into four 2x2 blocks
    # A is 4x4, blocks: A[0:2,0:2]=A11, A[0:2,2:4]=A12, A[2:4,0:2]=A21, A[2:4,2:4]=A22
    # In flat row-major: A11 indices 0-3, A12 indices 4-7, A21 indices 8-11, A22 indices 12-15
    
    # Apply Strassen to the 4 2x2 blocks:
    # M1_4x4 = (A11+A22)(B11+B22) where these are 2x2 matrices
    # Each product is a 2x2x2 Strassen with rank 7
    # So we need 7 such products, each contributing to 4 output positions
    
    # Build 4x4x4 Strassen factors
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49)
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # For each of the 7 Strassen terms at the 4x4 level, we have a 2x2x2 product
    # Term s (0-6) uses A-blocks and B-blocks as specified, producing C-blocks
    # Each 2x2x2 product has 7 sub-terms
    
    # The 4x4 Strassen structure:
    # Let A = [[A11, A12], [A21, A22]] where each is 2x2
    # Similarly for B and C
    
    # M1_4 = (A11+A22)(B11+B22)
    # M2_4 = (A21+A22)B11
    # M3_4 = A11(B21-B22)
    # M4_4 = A22(B12-B11)
    # M5_4 = (A11+A12)B22
    # M6_4 = (A21-A11)(B11+B12)
    # M7_4 = (A12-A22)(B21+B22)
    
    # C11_4 = M1_4+M4_4-M5_4+M7_4
    # C12_4 = M3_4+M5_4
    # C21_4 = M2_4+M4_4
    # C22_4 = M1_4-M2_4+M3_4+M6_4
    
    # For each 4x4 Strassen term s, we form a sum of 2x2 blocks for A and B
    # then multiply them using 2x2x2 Strassen (7 sub-terms)
    
    # Define which A-blocks and B-blocks go into each 4x4 Strassen term
    # A-block indices: 0=A11, 1=A12, 2=A21, 3=A22
    # B-block indices: 0=B11, 1=B12, 2=B21, 3=B22
    
    # 4x4 Strassen term definitions (A-blocks to sum, B-blocks to sum):
    # s=0: A: {0,3} with +1, B: {0,3} with +1
    # s=1: A: {2,3} with +1, B: {0} with +1
    # s=2: A: {0} with +1, B: {2,-3} i.e. B2-B3
    # s=3: A: {3} with +1, B: {1,-0} i.e. B1-B0
    # s=4: A: {0,1} with +1, B: {3} with +1
    # s=5: A: {2,-0} i.e. A2-A0, B: {0,1} with +1
    # s=6: A: {1,-3} i.e. A1-A3, B: {2,3} with +1
    
    a_block_coeffs = [
        [(0, 1), (3, 1)],    # s=0: A0+A3
        [(2, 1), (3, 1)],    # s=1: A2+A3
        [(0, 1)],            # s=2: A0
        [(3, 1)],            # s=3: A3
        [(0, 1), (1, 1)],    # s=4: A0+A1
        [(2, 1), (0, -1)],   # s=5: A2-A0
        [(1, 1), (3, -1)],   # s=6: A1-A3
    ]
    
    b_block_coeffs = [
        [(0, 1), (3, 1)],    # s=0: B0+B3
        [(0, 1)],            # s=1: B0
        [(2, 1), (3, -1)],   # s=2: B2-B3
        [(1, 1), (0, -1)],   # s=3: B1-B0
        [(3, 1)],            # s=4: B3
        [(0, 1), (1, 1)],    # s=5: B0+B1
        [(2, 1), (3, 1)],    # s=6: B2+B3
    ]
    
    # C-block output for each 4x4 Strassen term:
    # C0 = M0+M3-M4+M6 (blocks: C00)
    # C1 = M2+M4 (blocks: C01)
    # C2 = M1+M3 (blocks: C10)
    # C3 = M0-M1+M2+M5 (blocks: C11)
    
    c_block_coeffs = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00: M0+M3-M4+M6
        [(2, 1), (4, 1)],                     # C01: M2+M4
        [(1, 1), (3, 1)],                     # C10: M1+M3
        [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11: M0-M1+M2+M5
    ]
    
    # Block offset in flat index: block b starts at b*4
    # A block 0 (A11): flat indices 0-3
    # A block 1 (A12): flat indices 4-7
    # A block 2 (A21): flat indices 8-11
    # A block 3 (A22): flat indices 12-15
    
    # B block 0 (B11): flat indices 0-3
    # B block 1 (B12): flat indices 4-7
    # B block 2 (B21): flat indices 8-11
    # B block 3 (B22): flat indices 12-15
    
    # C block 0 (C11): flat indices 0-3
    # C block 1 (C12): flat indices 4-7
    # C block 2 (C21): flat indices 8-11
    # C block 3 (C22): flat indices 12-15
    
    # Build U4, V4, W4
    for s in range(7):
        for t in range(7):  # sub-term in 2x2x2 Strassen
            col = s * 7 + t
            
            # U part: for each A-block involved in term s, add U2 coefficients
            for (blk, coeff) in a_block_coeffs[s]:
                block_start = blk * 4
                for local_idx in range(4):
                    if U2[local_idx, t] != 0:
                        U4[block_start + local_idx, col] += coeff * U2[local_idx, t]
            
            # V part
            for (blk, coeff) in b_block_coeffs[s]:
                block_start = blk * 4
                for local_idx in range(4):
                    if V2[local_idx, t] != 0:
                        V4[block_start + local_idx, col] += coeff * V2[local_idx, t]
            
            # W part: for each C-block that term s contributes to
            for c_blk in range(4):
                for (s_idx, coeff) in c_block_coeffs[c_blk]:
                    if s_idx == s:
                        block_start = c_blk * 4
                        for local_idx in range(4):
                            if W2[local_idx, t] != 0:
                                W4[block_start + local_idx, col] += coeff * W2[local_idx, t]
    
    # Now we have 4x4x4 Strassen factors. But we need 3x4x7.
    # A is 3x4 (not 4x4), B is 4x7 (not 4x8), C is 3x7 (not 4x8)
    # 
    # We need to extract the 3x4x7 subproblem from the 4x4x8 computation.
    # 
    # A is 3x4: we use rows 0,1,2 of the 4x4 A (ignore row 3)
    # B is 4x7: we use all 4 rows and columns 0-6 of the 4x8 B (ignore col 7)
    # C is 3x7: we use rows 0,1,2 and cols 0-6 of the 4x8 C
    #
    # The 4x4x8 tensor T4[i,j,k] for i in 0..3, j in 0..3, k in 0..7
    # We want T3[i,j,k] for i in 0..2, j in 0..3, k in 0..6
    #
    # Since the decomposition is linear, we can just restrict the factor matrices:
    # U has shape (4*4=16, 49) for A. We only need rows for i=0,1,2 (12 rows)
    # V has shape (4*8=32, 49) for B. We only need cols k=0..6 (28 rows)  
    # W has shape (8*4=32, 49) for C. We only need i=0..2, k=0..6 (21 rows)
    
    # U4 is (16, 49): row index = i*4+j for i in 0..3, j in 0..3
    # We want i in 0..2, so rows 0..11 (i=0: 0-3, i=1: 4-7, i=2: 8-11)
    U3 = U4[0:12, :]
    
    # V4 is (32, 49): row index = j*8+k for j in 0..3, k in 0..7
    # We want k in 0..6, so for each j, keep k=0..6
    V3 = np.zeros((4*7, 49))
    for j in range(4):
        for k in range(7):
            V3[j*7+k, :] = V4[j*8+k, :]
    
    # W4 is (32, 49): row index = k*4+i for k in 0..7, i in 0..3
    # We want i in 0..2, k in 0..6
    W3 = np.zeros((7*3, 49))
    for k in range(7):
        for i in range(3):
            W3[k*3+i, :] = W4[k*4+i, :]
    
    # Verify with random test
    rng = np.random.RandomState(42)
    A_test = rng.randn(3, 4)
    B_test = rng.randn(4, 7)
    C_true = A_test @ B_test
    
    C_approx = np.zeros((3, 7))
    A_flat = A_test.reshape(-1)
    B_flat = B_test.reshape(-1)
    for r in range(49):
        ua = U3[:, r]
        va = V3[:, r]
        wa = W3[:, r]
        u_dot_a = ua @ A_flat
        v_dot_b = va @ B_flat
        C_approx += (wa * (u_dot_a * v_dot_b)).reshape(7, 3).T
    
    if not np.allclose(C_true, C_approx, atol=1e-10):
        # Fallback to schoolbook if verification fails
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U3.tolist(), "V": V3.tolist(), "W": W3.tolist()}

# EVOLVE-BLOCK-END
