"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the standard rank-144 schoolbook decomposition for 4x4 * 4x9."""
    n, m, p = DIMENSIONS  # 4, 4, 9
    rank = n * m * p  # 144

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] -> i*m+j ; B[j,k] -> j*p+k ; C[i,k] -> k*n+i (official trace convention)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """Return a rank-7 Strassen decomposition for 2x2 * 2x2.
    U: 4x7, V: 4x7, W: 4x7 with W[k*n+i] convention (n=2).
    Indices: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> k*2+i
    """
    n = 2
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3-M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: A00,A11 * B00,B11 -> C00(+), C11(+)
    U[0,0] = 1; U[3,0] = 1  # A00, A11
    V[0,0] = 1; V[3,0] = 1  # B00, B11
    W[0,0] = 1; W[1,0] = 1  # C00, C11
    
    # M2: A10,A11 * B00 -> C10(+), C11(-)
    U[2,1] = 1; U[3,1] = 1  # A10, A11
    V[0,1] = 1              # B00
    W[1,1] = 1; W[0,1] = -1  # C10, C11
    
    # M3: A00 * B01,B11 -> C01(+), C11(+)
    U[0,2] = 1              # A00
    V[1,2] = 1; V[3,2] = -1  # B01, B11
    W[0,2] = 1; W[1,2] = 1  # C01, C11
    
    # M4: A11 * B10,B00 -> C00(+), C10(+)
    U[3,3] = 1              # A11
    V[2,3] = 1; V[0,3] = -1  # B10, B00
    W[0,3] = 1; W[1,3] = 1  # C00, C10
    
    # M5: A00,A10 * B11 -> C00(-), C01(+)
    U[0,4] = 1; U[2,4] = 1  # A00, A10
    V[3,4] = 1              # B11
    W[0,4] = -1; W[0,2] = 1  # C00, C01 (note: C01 index is 0*2+1=1? No, C[i,k] -> k*n+i. C01 -> 1*2+0=2? Wait)
    
    # Let me re-check W indexing: W[k*n+i] where C = A*B, C[i][k]
    # C[0][0] -> 0*2+0 = 0
    # C[0][1] -> 1*2+0 = 2
    # C[1][0] -> 0*2+1 = 1
    # C[1][1] -> 1*2+1 = 3
    
    # So:
    # M1 -> C00(idx 0), C11(idx 3)
    # M2 -> C10(idx 1), C11(idx 3)
    # M3 -> C01(idx 2), C11(idx 3)
    # M4 -> C00(idx 0), C10(idx 1)
    # M5 -> C00(idx 0), C01(idx 2)
    # M6 -> C11(idx 3)
    # M7 -> C00(idx 0), C01(idx 2)
    
    # Let me redo:
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    
    # M3: A00(B01-B11) -> C01 += M3, C11 += M3
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    # M4: A11(B10-B00) -> C00 += M4, C10 += M4
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # M5: (A00+A10)B11 -> C00 -= M5, C01 += M5
    U[0,4] = 1; U[2,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # M6: (A10-A00)(B00+B10) -> C11 -= M6
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[2,5] = 1
    W[3,5] = -1
    
    # M7: (A01-A11)(B01+B11) -> C00 -= M7, C01 -= M7  -- wait, let me check
    # C00 = M1+M4-M5-M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3-M6
    # So M7 appears in C00 with -1. But standard Strassen has M7 in C00 and C01?
    # Let me re-derive. Standard Strassen:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    # So M7 only in C00.
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[3,6] = 1
    W[0,6] = -1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify the tensor identity exactly."""
    U2 = (np.array(U) * 2).astype(int)
    V2 = (np.array(V) * 2).astype(int)
    W2 = (np.array(W) * 2).astype(int)
    R = U2.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U2[i * m + j, r] * V2[j * p + k, r] * W2[k * n + i, r]
                if s != 8:
                    return False
    return True


def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 115.0  # safety margin

    # --- Step 1: always have a valid fallback ---------------------------------
    best = _schoolbook()
    best_rank = n * m * p  # 144

    # --- Try block decomposition: split 4x4 into 2x2 blocks ---
    # A is 4x4, B is 4x9, C is 4x9
    # We can think of A as 2x2 blocks of 2x2, B as 2x5 blocks of 2x2 (with padding)
    # Actually, B is 4x9. We can split B into 2 blocks of 2x9 (vertically by j)
    # and 9/2... 9 is odd. Let's split 4 into 2+2 for j, and 9 into 4+4+1 for k?
    # Or just use the 2x2 Strassen for the 4x4 part of B (columns 0-3) and do
    # columns 4-8 schoolbook.
    # 
    # Strategy: Split j into j1=0,1 and j2=2,3. Split k into k1=0..3 and k2=4..8.
    # For (j1, k1): 2x2 * 2x4 -> use Strassen-like? No, Strassen is 2x2*2x2.
    # We can split k1 into two 2x2 blocks: k1a=0,1 and k1b=2,3.
    # So: A (4x4) = [[A11,A12],[A21,A22]] each 2x2
    # B (4x9) columns 0-3: [[B11a,B11b],[B21a,B21b]] where B11a is 2x2 (rows 0-1, cols 0-1)
    # B (4x9) columns 4-8: [[B12],[B22]] each 2x5
    #
    # This is getting complex. Let me try a different approach.
    # 
    # Use Strassen for the 4x4 multiplication part, treating B's first 4 columns
    # as a 4x4 matrix, and handle the remaining 5 columns schoolbook.
    #
    # C[0:4, 0:4] = A * B[:, 0:4] (4x4 result)
    # C[0:4, 4:9] = A * B[:, 4:9] (4x5 result)
    #
    # For the 4x4 part, use Strassen recursively: 4x4 = 2x2 blocks of 2x2.
    # This requires 7 multiplications of 2x2*2x2 = 7*7 = 49 multiplications... no.
    # Strassen for 4x4: split into 2x2 blocks, use 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication uses 7 scalar multiplications (Strassen).
    # Total: 7 * 7 = 49 scalar multiplications for the 4x4 part.
    # Plus 4*5 = 20 for the 4x5 part (schoolbook).
    # Total rank: 49 + 20 = 69.
    #
    # But wait, Strassen for 4x4 doesn't directly give 49 rank for the tensor.
    # The rank of M_4 is 49 (known), but constructing it explicitly is hard.
    #
    # Let me try a simpler approach: use 2x2 Strassen blocks.
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split B columns 0-3 into 2x2 blocks: B4 = [[B00, B01], [B10, B11]]
    # Split C columns 0-3 into 2x2 blocks: C4 = [[C00, C01], [C10, C11]]
    #
    # C4 = A * B4 where each block is 2x2.
    # Using Strassen: 7 multiplications of 2x2*2x2.
    # Each 2x2*2x2 can be done with rank 7 (Strassen) or rank 8 (schoolbook).
    # If we use Strassen for each, total rank for 4x4 part = 7 * 7 = 49.
    # But the intermediate 2x2 multiplications need to be composed.
    #
    # Actually, the rank of the 4x4 multiplication tensor when using Strassen
    # at the 2x2 level is: 7 (Strassen for 4x4) * 1? No.
    # Strassen for 4x4: 7 multiplications of 2x2 matrices. Each 2x2 matrix
    # multiplication is a rank-7 tensor. So the total rank is 7*7 = 49?
    # Not exactly, because the 7 multiplications produce 2x2 results that are
    # combined. The tensor rank is not simply multiplicative.
    #
    # Let me just try a direct construction using 2x2 Strassen blocks.
    #
    # For the 4x4 * 4x4 part (columns 0-3 of B):
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # B4 = [[B00, B01], [B10, B11]] (2x2 blocks of 2x2)
    # C4 = A * B4 = [[C00, C01], [C10, C11]]
    #
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3-M6
    #
    # Each Mi is a 2x2 matrix product. If we use schoolbook for each 2x2*2x2,
    # that's 8 multiplications each, total 7*8 = 56.
    # If we use Strassen for each 2x2*2x2, that's 7 each, total 7*7 = 49.
    #
    # Let me implement the 49-rank version using Strassen at both levels.
    #
    # The tensor for 4x4*4x4 with rank 49:
    # Each of the 7 Strassen multiplications is a 2x2*2x2 tensor of rank 7.
    # The total tensor is the sum of 7 such tensors, each contributing to
    # specific 2x2 blocks of the output.
    #
    # This is complex to implement correctly. Let me try a different approach.
    #
    # --- Alternative: use rank 64 (4x4*4x4 schoolbook) + 20 (4x5 schoolbook) = 84 ---
    # That's better than 144. Let me verify: 4*4*4 + 4*5 = 64 + 20 = 84.
    #
    # Actually, schoolbook for 4x4*4x9 is 4*4*9 = 144.
    # If I split B into 4x4 + 4x5, the 4x4 part is 4*4*4 = 64, the 4x5 part is 4*5*4 = 80? No.
    # C[i,k] = sum_j A[i,j]*B[j,k]. For k=0..3: sum_{j=0..3} A[i,j]*B[j,k]. That's 4*4*4=64 terms.
    # For k=4..8: sum_{j=0..3} A[i,j]*B[j,k]. That's 4*5*4=80 terms.
    # Total: 64+80 = 144. Same as schoolbook. No gain.
    #
    # The gain from Strassen comes from reducing the number of multiplications,
    # not from splitting the problem.
    #
    # Let me try to implement a rank-69 decomposition using 2x2 Strassen blocks.
    #
    # Actually, let me just try to find a valid lower-rank decomposition using
    # a more systematic search with the deadline.

    # --- Step 2: try a structured decomposition ---
    # Use 2x2 Strassen for the 4x4*4x4 sub-problem (first 4 columns of B),
    # and schoolbook for the 4x5 sub-problem (last 5 columns of B).
    
    # For the 4x4*4x4 part, use Strassen recursively:
    # Level 1: 4x4 = 2x2 blocks, 7 multiplications of 2x2*2x2
    # Level 2: each 2x2*2x2 = 7 multiplications of scalars (Strassen)
    # Total: 49 multiplications for the 4x4*4x4 part
    
    # For the 4x5 part: 4*5 = 20 multiplications (schoolbook)
    # Total rank: 49 + 20 = 69
    
    U = np.zeros((n * m, 69))
    V = np.zeros((m * p, 69))
    W = np.zeros((p * n, 69))
    
    col = 0
    # 4x4*4x4 part using 2x2 Strassen blocks
    # A indices: A[i,j] for i,j in 0..3. In 2x2 blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], etc.
    # B indices: B[j,k] for j in 0..3, k in 0..3 (first 4 columns)
    # C indices: C[i,k] for i in 0..3, k in 0..3 (first 4 columns)
    
    # Strassen 2x2 block decomposition for 4x4*4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Each Mi is a 2x2*2x2 product. Use Strassen for each (rank 7 each).
    
    def add_strassen_block(U, V, W, col, a_block, b_block, c_block, sign_c):
        """Add a 2x2*2x2 Strassen decomposition.
        a_block, b_block, c_block are 2x2 index mappings.
        a_block: (i1,j1,i2,j2) for the 2x2 A block
        b_block: (j1,k1,j2,k2) for the 2x2 B block
        c_block: (i1,k1,i2,k2) for the 2x2 C block
        sign_c: multiplier for the C contribution
        Returns new col offset.
        """
        (ai1, aj1, ai2, aj2) = a_block
        (bj1, bk1, bj2, bk2) = b_block
        (ci1, ck1, ci2, ck2) = c_block
        
        # 7 Strassen multiplications for 2x2*2x2
        # Each has rank 7 (scalar Strassen)
        # M1 = (A00+A11)(B00+B11)
        # A00 -> a_block[0,0] = (ai1,aj1), A11 -> a_block[1,1] = (ai2,aj2)
        # B00 -> b_block[0,0] = (bj1,bk1), B11 -> b_block[1,1] = (bj2,bk2)
        # C00 -> c_block[0,0] = (ci1,ck1), C11 -> c_block[1,1] = (ci2,ck2)
        
        # Scalar Strassen for each 2x2*2x2:
        # S1 = (A00+A11)(B00+B11) -> C00 += S1, C11 += S1
        # S2 = (A10+A11)B00 -> C10 += S2, C11 -= S2
        # S3 = A00(B01-B11) -> C01 += S3, C11 += S3
        # S4 = A11(B10-B00) -> C00 += S4, C10 += S4
        # S5 = (A00+A10)B11 -> C00 -= S5, C01 += S5
        # S6 = (A10-A00)(B00+B10) -> C11 -= S6
        # S7 = (A01-A11)(B01+B11) -> C00 -= S7
        
        # For each scalar Strassen, we need rank 7? No, scalar Strassen is rank 7
        # for 2x2*2x2. But here we're doing 2x2*2x2 where each "scalar" is actually
        # a product. So each S is a single multiplication.
        
        # Actually, for 2x2*2x2, Strassen uses 7 scalar multiplications.
        # So each S is one scalar multiplication.
        
        # S1: (A00+A11)*(B00+B11)
        U[ai1*m+aj1, col] = sign_c
        U[ai2*m+aj2, col] = sign_c
        V[bj1*p+bk1, col] = 1
        V[bj2*p+bk2, col] = 1
        W[ck1*n+ci1, col] = 1
        W[ck2*n+ci2, col] = 1
        col += 1
        
        # S2: (A10+A11)*B00
        # A10 -> a_block[1,0] = (ai2,aj1), A11 -> a_block[1,1] = (ai2,aj2)
        U[ai2*m+aj1, col] = sign_c
        U[ai2*m+aj2, col] = sign_c
        V[bj1*p+bk1, col] = 1
        W[ck1*n+ci2, col] = 1  # C10
        W[ck2*n+ci2, col] = -1  # C11
        col += 1
        
        # S3: A00*(B01-B11)
        # A00 -> (ai1,aj1), B01 -> b_block[0,1] = (bj1,bk2), B11 -> (bj2,bk2)
        U[ai1*m+aj1, col] = sign_c
        V[bj1*p+bk2, col] = 1
        V[bj2*p+bk2, col] = -1
        W[ck1*n+ci2, col] = 1  # C01
        W[ck2*n+ci2, col] = 1  # C11
        col += 1
        
        # S4: A11*(B10-B00)
        # A11 -> (ai2,aj2), B10 -> b_block[1,0] = (bj2,bk1), B00 -> (bj1,bk1)
        U[ai2*m+aj2, col] = sign_c
        V[bj2*p+bk1, col] = 1
        V[bj1*p+bk1, col] = -1
        W[ck1*n+ci1, col] = 1  # C00
        W[ck1*n+ci2, col] = 1  # C10
        col += 1
        
        # S5: (A00+A10)*B11
        # A00 -> (ai1,aj1), A10 -> (ai2,aj1), B11 -> (bj2,bk2)
        U[ai1*m+aj1, col] = sign_c
        U[ai2*m+aj1, col] = sign_c
        V[bj2*p+bk2, col] = 1
        W[ck1*n+ci1, col] = -1  # C00
        W[ck1*n+ci2, col] = 1   # C01
        col += 1
        
        # S6: (A10-A00)*(B00+B10)
        # A10 -> (ai2,aj1), A00 -> (ai1,aj1), B00 -> (bj1,bk1), B10 -> (bj2,bk1)
        U[ai2*m+aj1, col] = sign_c
        U[ai1*m+aj1, col] = -sign_c
        V[bj1*p+bk1, col] = 1
        V[bj2*p+bk1, col] = 1
        W[ck2*n+ci2, col] = -1  # C11
        col += 1
        
        # S7: (A01-A11)*(B01+B11)
        # A01 -> a_block[0,1] = (ai1,aj2), A11 -> (ai2,aj2), B01 -> (bj1,bk2), B11 -> (bj2,bk2)
        U[ai1*m+aj2, col] = sign_c
        U[ai2*m+aj2, col] = -sign_c
        V[bj1*p+bk2, col] = 1
        V[bj2*p+bk2, col] = 1
        W[ck1*n+ci1, col] = -1  # C00
        col += 1
        
        return col
    
    # 4x4 blocks:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4], C10 = C[2:4, 0:2], C11 = C[2:4, 2:4]
    
    # Block index mappings (i,j) for 2x2 blocks:
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
    # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)
    
    # M1 = (A00+A11)(B00+B11)
    col = add_strassen_block(U, V, W, col, (0,0,2,2), (0,0,2,2), (0,0,2,2), 1)
    # M2 = (A10+A11)B00
    col = add_strassen_block(U, V, W, col, (2,0,2,2), (0,0,0,0), (2,0,2,2), 1)
    # M3 = A00(B01-B11)
    col = add_strassen_block(U, V, W, col, (0,0,0,0), (0,2,2,2), (0,2,2,2), 1)
    # M4 = A11(B10-B00)
    col = add_strassen_block(U, V, W, col, (2,2,2,2), (2,0,0,0), (0,0,2,0), 1)
    # M5 = (A00+A10)B11
    col = add_strassen_block(U, V, W, col, (0,0,2,0), (2,2,2,2), (0,0,0,2), 1)
    # M6 = (A10-A00)(B00+B10)
    col = add_strassen_block(U, V, W, col, (2,0,0,0), (0,0,2,0), (2,2,2,2), 1)
    # M7 = (A01-A11)(B01+B11)
    col = add_strassen_block(U, V, W, col, (0,2,2,2), (0,2,2,2), (0,0,0,2), 1)
    
    # Now add schoolbook for columns 4-8 of B
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify
    if col <= 69:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        if _verify(U, V, W, n, m, p):
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = col
            return best
    
    # Fallback to schoolbook
    return best
# EVOLVE-BLOCK-END

