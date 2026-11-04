import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 5
    
    # Build 4x4x4 decomposition using Strassen tiling
    # Strassen 2x2x2: 7 terms
    # 4x4x4: tile into 2x2 blocks, each block mult uses 7 terms, total 49
    
    # Strassen 2x2x2 factors:
    # For A (2x2) * B (2x2) = C (2x2):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 4x4x4, we tile A, B, C into 2x2 blocks of 2x2 submatrices
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # Similarly B, C
    # Then C00 = A00*B00 + A01*B10
    #         = (using Strassen on the 2x2 block level)
    # Each block product Ai*Bj is a 2x2*2x2 = 2x2 multiplication, rank 7
    # But we have 4 block multiplications for each C-block, and 4 C-blocks
    # Total block multiplications: 16, but Strassen reduces to 7 at the block level
    # Wait, I need to be more careful.
    
    # Actually, the 4x4x4 multiplication tensor can be decomposed by viewing
    # it as a 2x2 block matrix multiplication where each block is 2x2.
    # The 2x2 block multiplication has rank 7 (Strassen).
    # Each block multiplication is itself a 2x2x2 tensor with rank 7.
    # So total rank = 7 * 7 = 49.
    
    # Let me construct this explicitly.
    # Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For 4x4x4: n=m=p=4, so U[4i+j, c], V[4j+k, c], W[4k+i, c]
    
    # Block indices: let i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # where i1,j1,k0 in {0,1} are block indices, i0,j0,k0 in {0,1} are within-block
    
    # U[4i+j, c] = U[(2*i1+i0)*4 + (2*j1+j0), c] = U[8*i1 + 4*i0 + 2*j1 + j0, c]
    # V[4j+k, c] = V[8*j1 + 4*j0 + 2*k1 + k0, c]  
    # W[4k+i, c] = W[8*k1 + 4*k0 + 2*i1 + i0, c]
    
    # The 49 terms correspond to (block_strassen_idx, within_block_strassen_idx)
    # where block_strassen_idx in {0,...,6} and within_block_strassen_idx in {0,...,6}
    
    # Strassen 2x2x2 factors for block level:
    # Let's define the 7 block-level terms M1..M7 as linear combinations of blocks
    # For block multiplication:
    # S1 = A00 + A11, T1 = B00 + B11
    # S2 = A10 + A11, T2 = B00
    # S3 = A00, T3 = B01 - B11
    # S4 = A11, T4 = B10 - B00
    # S5 = A00 + A01, T5 = B11
    # S6 = A10 - A00, T6 = B00 + B01
    # S7 = A01 - A11, T7 = B10 + B11
    
    # Then C00 = M1 + M4 - M5 + M7 where Mq = Sq * Tq (block mult)
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mq is a 2x2 block product, which itself has rank 7 (Strassen within-block)
    
    # So for each q in 0..6, we have 7 within-block terms
    # Total: 49 terms
    
    # Let me build U, V, W for 4x4x4 first, then extend to 4x4x5
    
    rank_444 = 49
    U444 = np.zeros((16, rank_444))
    V444 = np.zeros((16, rank_444))
    W444 = np.zeros((16, rank_444))
    
    # Strassen within-block factors for 2x2x2:
    # For A(2x2)*B(2x2)=C(2x2), with A[i,j], B[j,k], C[k,i] (note: C is transposed in our convention)
    # Actually, in our convention: U[i*m+j, c] for A's (i,j), V[j*p+k, c] for B's (j,k), W[k*n+i, c] for C's (k,i)
    # For 2x2x2: n=m=p=2, so U[2i+j,c], V[2j+k,c], W[2k+i,c]
    
    # Strassen 2x2x2: 7 terms
    # Term 0: (a00+a11)*(b00+b11) -> contributes to c00, c11
    # Let me use the standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # For each term t, we have U values (for a's), V values (for b's), W values (for c's)
    # U[2i+j, t], V[2j+k, t], W[2k+i, t]
    
    strassen_U = np.zeros((4, 7))  # 2x2 matrix for A, flattened as 2i+j
    strassen_V = np.zeros((4, 7))  # 2x2 matrix for B, flattened as 2j+k
    strassen_W = np.zeros((4, 7))  # 2x2 matrix for C, flattened as 2k+i
    
    # Term 0: M1 = (a00+a11)(b00+b11)
    strassen_U[0, 0] = 1  # a00
    strassen_U[3, 0] = 1  # a11 (2*1+1=3)
    strassen_V[0, 0] = 1  # b00
    strassen_V[3, 0] = 1  # b11 (2*1+1=3)
    # W: c00 += 1, c11 += 1
    strassen_W[0, 0] = 1  # c00 (2*0+0=0)
    strassen_W[3, 0] = 1  # c11 (2*1+1=3)
    
    # Term 1: M2 = (a10+a11)b00
    strassen_U[2, 1] = 1  # a10 (2*1+0=2)
    strassen_U[3, 1] = 1  # a11
    strassen_V[0, 1] = 1  # b00
    # W: c10 += 1
    strassen_W[2, 1] = 1  # c10 (2*1+0=2)
    
    # Term 2: M3 = a00(b01-b11)
    strassen_U[0, 2] = 1  # a00
    strassen_V[1, 2] = 1  # b01 (2*0+1=1)
    strassen_V[3, 2] = -1 # b11
    # W: c01 += 1
    strassen_W[1, 2] = 1  # c01 (2*0+1=1)
    
    # Term 3: M4 = a11(b10-b00)
    strassen_U[3, 3] = 1  # a11
    strassen_V[2, 3] = 1  # b10 (2*1+0=2)
    strassen_V[0, 3] = -1 # b00
    # W: c00 += 1, c10 += 1
    strassen_W[0, 3] = 1  # c00
    strassen_W[2, 3] = 1  # c10
    
    # Term 4: M5 = (a00+a01)b11
    strassen_U[0, 4] = 1  # a00
    strassen_U[1, 4] = 1  # a01 (2*0+1=1)
    strassen_V[3, 4] = 1  # b11
    # W: c00 -= 1, c01 += 1
    strassen_W[0, 4] = -1 # c00
    strassen_W[1, 4] = 1  # c01
    
    # Term 5: M6 = (a10-a00)(b00+b01)
    strassen_U[2, 5] = 1  # a10
    strassen_U[0, 5] = -1 # a00
    strassen_V[0, 5] = 1  # b00
    strassen_V[1, 5] = 1  # b01
    # W: c11 += 1
    strassen_W[3, 5] = 1  # c11
    
    # Term 6: M7 = (a01-a11)(b10+b11)
    strassen_U[1, 6] = 1  # a01
    strassen_U[3, 6] = -1 # a11
    strassen_V[2, 6] = 1  # b10
    strassen_V[3, 6] = 1  # b11
    # W: c00 += 1, c11 += 1
    strassen_W[0, 6] = 1  # c00
    strassen_W[3, 6] = 1  # c11
    
    # Verify Strassen 2x2x2
    def verify_2x2x2(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = np.dot(U[2*i+j, :], V[2*j+k, :] * W[2*k+i, :])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    assert verify_2x2x2(strassen_U, strassen_V, strassen_W), "Strassen 2x2x2 verification failed"
    
    # Now build 4x4x4 using block Strassen
    # Block level: 7 terms (q=0..6)
    # Within block: 7 terms (t=0..6)
    # Total: 49 terms, indexed by (q, t) -> col = q*7 + t
    
    # Block Strassen factors for the 2x2 block multiplication:
    # A_blocks: A00, A01, A10, A11 (each 2x2)
    # B_blocks: B00, B01, B10, B11
    # C_blocks: C00, C01, C10, C11
    
    # The 7 block-level terms:
    # q=0: (A00+A11)*(B00+B11)
    # q=1: (A10+A11)*B00
    # q=2: A00*(B01-B11)
    # q=3: A11*(B10-B00)
    # q=4: (A00+A01)*B11
    # q=5: (A10-A00)*(B00+B01)
    # q=6: (A01-A11)*(B10+B11)
    
    # C00 = q0 + q3 - q4 + q6
    # C01 = q2 + q4
    # C10 = q1 + q3
    # C11 = q0 - q1 + q2 + q5
    
    # For each block-level term q, and within-block term t:
    # U[4i+j, q*7+t] = block_U[q, block_idx] * strassen_U[2*i0+j0, t]
    # where i = 2*i1+i0, j = 2*j1+j0, block_idx = 2*i1+j1
    # Similarly for V, W
    
    # Block U factors (for A's blocks):
    # q=0: A00:1, A11:1
    # q=1: A10:1, A11:1
    # q=2: A00:1
    # q=3: A11:1
    # q=4: A00:1, A01:1
    # q=5: A10:1, A00:-1
    # q=6: A01:1, A11:-1
    
    block_U = np.zeros((4, 7))  # 4 blocks, 7 terms
    block_U[0, 0] = 1  # A00, q0
    block_U[3, 0] = 1  # A11, q0
    block_U[2, 1] = 1  # A10, q1
    block_U[3, 1] = 1  # A11, q1
    block_U[0, 2] = 1  # A00, q2
    block_U[3, 3] = 1  # A11, q3
    block_U[0, 4] = 1  # A00, q4
    block_U[1, 4] = 1  # A01, q4
    block_U[2, 5] = 1  # A10, q5
    block_U[0, 5] = -1 # A00, q5
    block_U[1, 6] = 1  # A01, q6
    block_U[3, 6] = -1 # A11, q6
    
    # Block V factors (for B's blocks):
    # q=0: B00:1, B11:1
    # q=1: B00:1
    # q=2: B01:1, B11:-1
    # q=3: B10:1, B00:-1
    # q=4: B11:1
    # q=5: B00:1, B01:1
    # q=6: B10:1, B11:1
    
    block_V = np.zeros((4, 7))
    block_V[0, 0] = 1  # B00, q0
    block_V[3, 0] = 1  # B11, q0
    block_V[0, 1] = 1  # B00, q1
    block_V[1, 2] = 1  # B01, q2
    block_V[3, 2] = -1 # B11, q2
    block_V[2, 3] = 1  # B10, q3
    block_V[0, 3] = -1 # B00, q3
    block_V[3, 4] = 1  # B11, q4
    block_V[0, 5] = 1  # B00, q5
    block_V[1, 5] = 1  # B01, q5
    block_V[2, 6] = 1  # B10, q6
    block_V[3, 6] = 1  # B11, q6
    
    # Block W factors (for C's blocks):
    # C00 = q0 + q3 - q4 + q6
    # C01 = q2 + q4
    # C10 = q1 + q3
    # C11 = q0 - q1 + q2 + q5
    
    block_W = np.zeros((4, 7))
    # C00 is block 0
    block_W[0, 0] = 1  # C00, q0
    block_W[0, 3] = 1  # C00, q3
    block_W[0, 4] = -1 # C00, q4
    block_W[0, 6] = 1  # C00, q6
    # C01 is block 1
    block_W[1, 2] = 1  # C01, q2
    block_W[1, 4] = 1  # C01, q4
    # C10 is block 2
    block_W[2, 1] = 1  # C10, q1
    block_W[2, 3] = 1  # C10, q3
    # C11 is block 3
    block_W[3, 0] = 1  # C11, q0
    block_W[3, 1] = -1 # C11, q1
    block_W[3, 2] = 1  # C11, q2
    block_W[3, 5] = 1  # C11, q5
    
    # Now fill U444, V444, W444
    for q in range(7):
        for t in range(7):
            col = q * 7 + t
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            i = 2 * i1 + i0
                            j = 2 * j1 + j0
                            block_idx = 2 * i1 + j1
                            U444[4 * i + j, col] = block_U[block_idx, q] * strassen_U[2 * i0 + j0, t]
                            V444[4 * j + (2 * i1 + i0), col] = 0  # placeholder, fill below
                            # Actually V depends on j and k, not i
                            # Let me redo this more carefully
    
    # Let me redo the filling properly
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    for q in range(7):
        for t in range(7):
            col = q * 7 + t
            # U: for each i,j in 4x4
            for i in range(4):
                for j in range(4):
                    i1, i0 = divmod(i, 2)
                    j1, j0 = divmod(j, 2)
                    block_idx = 2 * i1 + j1
                    U444[4 * i + j, col] = block_U[block_idx, q] * strassen_U[2 * i0 + j0, t]
            
            # V: for each j,k in 4x4
            for j in range(4):
                for k in range(4):
                    j1, j0 = divmod(j, 2)
                    k1, k0 = divmod(k, 2)
                    block_idx = 2 * j1 + k1
                    V444[4 * j + k, col] = block_V[block_idx, q] * strassen_V[2 * j0 + k0, t]
            
            # W: for each k,i in 4x4
            for k in range(4):
                for i in range(4):
                    k1, k0 = divmod(k, 2)
                    i1, i0 = divmod(i, 2)
                    block_idx = 2 * k1 + i1
                    W444[4 * k + i, col] = block_W[block_idx, q] * strassen_W[2 * k0 + i0, t]
    
    # Verify 4x4x4
    def verify_444(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = np.dot(U[4 * i + j, :], V[4 * j + k, :] * W[4 * k + i, :])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    assert verify_444(U444, V444, W444), "4x4x4 verification failed"
    
    # Now extend to 4x4x5
    # The 5th column of B: B[j,4] for j=0..3
    # We need C[i,4] = sum_j A[i,j] * B[j,4]
    # This is a 4x4x1 multiplication, which has rank 16 (schoolbook)
    # But we can potentially share with the existing decomposition
    
    # For the 4x4x1 part:
    # U_ext[i*4+j, c] * V_ext[j*5+4, c] * W_ext[4*4+i, c] = 1 for the (i,4) entry
    # Wait, W[k*n+i] with k=4, n=4: W[4*4+i, c] = W[16+i, c]
    # But W is supposed to be (p*n, rank) = (5*4, rank) = (20, rank)
    # And V is (m*p, rank) = (4*5, rank) = (20, rank)
    
    # So for the extra column, we need 16 more terms (schoolbook for 4x4x1)
    # Total rank: 49 + 16 = 65
    
    rank = 65
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))
    W = np.zeros((20, rank))
    
    # Copy the 4x4x4 part (columns 0..48)
    U[:, :49] = U444
    V[:, :49] = V444  # V444 is 16x49, but V is 20x49, so rows 0..15 get values, 16..19 stay 0
    W[:, :49] = W444  # Similarly, W444 is 16x49, rows 0..15 get values
    
    # Wait, V444 has shape (16, 49) but V needs shape (20, 49). The first 16 rows correspond to j*4+k for k=0..3.
    # Rows 16..19 correspond to j*4+4 for j=0..3, which is the 5th column. These should be 0 in the 4x4x4 part.
    # Similarly W444 has shape (16, 49) but W needs (20, 49). Rows 0..15 are k*4+i for k=0..3.
    # Rows 16..19 are k*4+i for k=4, i.e., the 5th row of C. These should be 0 in the 4x4x4 part.
    
    # Now add the 4x4x1 part (columns 49..64)
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U[4 * i + j, col] = 1
            V[4 * j + 4, col] = 1  # j*5+4 = 4j+4
            W[4 * 4 + i, col] = 1  # 4*4+i = 16+i
    
    # Verify full 4x4x5
    def verify_445(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    s = np.dot(U[4 * i + j, :], V[4 * j + k, :] * W[4 * k + i, :])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    assert verify_445(U, V, W), "4x4x5 verification failed"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

