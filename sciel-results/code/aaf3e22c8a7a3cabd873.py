import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    # Strassen's 7-term algorithm for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 4x4x4 using Kronecker product of two Strassen instances:
    # Rank = 7*7 = 49
    # Then for the extra 3 columns, use schoolbook: 4*4*3 = 48
    # Total: 49 + 48 = 97
    
    # But we can do better by using a mixed approach.
    # Let's build the 4x4x4 part with rank 49, and 4x4x3 with rank 48.
    
    def strassen_factors_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition, each of shape (4, 7)."""
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Index: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> k*2+i (per task: W[k*n+i])
        # For 2x2: W[k*2+i]
        
        # M1 = (A00+A11)(B00+B11)
        # A00 -> U[0,0]=1, A11 -> U[3,0]=1
        # B00 -> V[0,0]=1, B11 -> V[3,0]=1
        # C00 gets +M1, C11 gets +M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # C[0,0] = k=0,i=0 -> W[0*2+0]=W[0]
        W[3, 0] = 1  # C[1,1] = k=1,i=1 -> W[1*2+1]=W[3]
        
        # M2 = (A10+A11)B00
        # A10 -> U[2,1]=1, A11 -> U[3,1]=1
        # B00 -> V[0,1]=1
        # C10 gets +M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C[1,0] = k=0,i=1 -> W[0*2+1]=W[1]... wait
        
        # Let me re-check indexing. W[k*n+i] where n=2.
        # C[i,k] is stored at W[k*2+i]
        # C[1,0]: i=1, k=0 -> W[0*2+1] = W[1]
        # C[1,1]: i=1, k=1 -> W[1*2+1] = W[3]
        # C[0,0]: i=0, k=0 -> W[0*2+0] = W[0]
        # C[0,1]: i=0, k=1 -> W[1*2+0] = W[2]
        
        # Let me redo this more carefully.
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00(0), A11(3)
        # V: B00(0), B11(3)
        # W: C00(0), C11(3)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        # U: A10(2), A11(3)
        # V: B00(0)
        # W: C10(1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = A00(B01-B11)
        # U: A00(0)
        # V: B01(1), B11(3) with -1
        # W: C01(2)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4 = A11(B10-B00)
        # U: A11(3)
        # V: B10(2), B00(0) with -1
        # W: C10(1)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1
        
        # M5 = (A00+A10)B11
        # U: A00(0), A10(2)
        # V: B11(3)
        # W: C00(0) with -1, C10(1) with +1
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A01-A10)(B00+B10)
        # U: A01(1), A10(2) with -1
        # V: B00(0), B10(2)
        # W: C11(3)
        U[1, 5] = 1; U[2, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B01+B11)
        # U: A01(1), A11(3) with -1
        # V: B01(1), B11(3)
        # W: C00(0)
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Verify Strassen 2x2x2
    U2, V2, W2 = strassen_factors_2x2x2()
    
    # Now build 4x4x4 using Kronecker product
    # The 4x4 matrix can be seen as 2x2 blocks of 2x2 submatrices
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = A*B = [[C00, C01], [C10, C11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Using Strassen on the 2x2 block structure:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each Mi is a 2x2 matrix product, which can be done with Strassen (rank 7)
    # So total rank for 4x4x4 is 7*7 = 49
    
    # For the Kronecker construction:
    # The indices for 4x4:
    # A[i,j] where i,j in 0..3. Block index: (i//2, i%2) and (j//2, j%2)
    # U4[i*4+j, r] where r encodes (block_row, block_col, strassen_term)
    
    # Let's build it properly.
    # For 4x4x4, we have:
    # U shape (16, 49), V shape (16, 49), W shape (16, 49)
    
    # The Kronecker product approach:
    # U4 = U2 ⊗ U2 (in some sense)
    # But we need to be careful about the indexing.
    
    # Actually, let's use a different approach. For 4x4x4:
    # Split into 4 quadrants. Use Strassen at the block level (7 terms),
    # and for each term, use Strassen at the sub-block level (7 terms).
    # Total: 7*7 = 49 terms.
    
    # Let me construct this explicitly.
    
    def build_4x4x4_strassen():
        """Build rank-49 decomposition for 4x4x4 using Strassen⊗Strassen."""
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # Strassen 2x2x2 factors
        U2, V2, W2 = strassen_factors_2x2x2()
        
        # For the block-level Strassen (7 terms), each term involves products of 2x2 blocks.
        # The block-level indices:
        # Block (bi, bj) of A: i in {0,1} for bi, j in {0,1} for bj
        # Within block: local indices (li, lj) in {0,1}
        # Global index: i = bi*2 + li, j = bj*2 + lj
        # A[i,j] -> U[i*4+j, col]
        
        # B block (bj, bk): j = bj*2 + lj, k = bk*2 + lk
        # B[j,k] -> V[j*4+k, col]
        
        # C block (bi, bk): i = bi*2 + li, k = bk*2 + lk
        # C[i,k] -> W[k*4+i, col]
        
        # For each block-level Strassen term t (0..6), and sub-block Strassen term s (0..6):
        # col = t*7 + s
        
        for t in range(7):
            for s in range(7):
                col = t * 7 + s
                
                # Block-level: which blocks are involved in term t?
                # U2[bi*2 + bj, t] gives the coefficient for A block (bi, bj)
                # V2[bj*2 + bk, t] gives the coefficient for B block (bj, bk)
                # W2[bk*2 + bi, t] gives the coefficient for C block (bi, bk)
                
                for bi in range(2):
                    for bj in range(2):
                        u_block_coeff = U2[bi * 2 + bj, t]
                        if u_block_coeff == 0:
                            continue
                        for li in range(2):
                            for lj in range(2):
                                # Sub-block Strassen: U2[li*2+lj, s]
                                u_sub_coeff = U2[li * 2 + lj, s]
                                if u_sub_coeff == 0:
                                    continue
                                i = bi * 2 + li
                                j = bj * 2 + lj
                                U[i * 4 + j, col] = u_block_coeff * u_sub_coeff
                
                for bj in range(2):
                    for bk in range(2):
                        v_block_coeff = V2[bj * 2 + bk, t]
                        if v_block_coeff == 0:
                            continue
                        for lj in range(2):
                            for lk in range(2):
                                v_sub_coeff = V2[lj * 2 + lk, s]
                                if v_sub_coeff == 0:
                                    continue
                                j = bj * 2 + lj
                                k = bk * 2 + lk
                                V[j * 4 + k, col] = v_block_coeff * v_sub_coeff
                
                for bk in range(2):
                    for bi in range(2):
                        w_block_coeff = W2[bk * 2 + bi, t]
                        if w_block_coeff == 0:
                            continue
                        for lk in range(2):
                            for li in range(2):
                                w_sub_coeff = W2[lk * 2 + li, s]
                                if w_sub_coeff == 0:
                                    continue
                                k = bk * 2 + lk
                                i = bi * 2 + li
                                W[k * 4 + i, col] = w_block_coeff * w_sub_coeff
        
        return U, V, W
    
    U4, V4, W4 = build_4x4x4_strassen()
    
    # Verify 4x4x4
    def verify_4x4x4(U, V, W, n=4, m=4, p=4):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # For 4x4x7, we need to handle the extra 3 columns.
    # Approach: Use the 4x4x4 part (rank 49) for the first 4 columns of B,
    # and schoolbook for the last 3 columns (rank 4*4*3 = 48).
    # Total rank: 49 + 48 = 97
    
    # But we can do better. Let's use a mixed approach:
    # For the 4x4x7 problem, split B into B1 (4x4) and B2 (4x3).
    # C1 = A*B1 (4x4) using rank 49
    # C2 = A*B2 (4x3) using schoolbook rank 48
    # Total: 97
    
    # Build the full 4x4x7 decomposition
    rank_total = 49 + 48
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((28, rank_total))  # m*p = 4*7 = 28
    W_full = np.zeros((28, rank_total))  # p*n = 7*4 = 28
    
    # Copy the 4x4x4 part (columns 0-3 of B)
    U_full[:, 0:49] = U4
    V_full[:, 0:49] = V4  # V4 is (16, 49) for 4x4, but we need (28, 49)
    # V4 has 16 rows (for 4x4 B matrix). We need 28 rows (for 4x7 B matrix).
    # The first 16 rows correspond to B columns 0-3, rows 16-27 are zero.
    V_full[0:16, 0:49] = V4
    V_full[16:28, 0:49] = 0
    
    # W4 is (16, 49) for 4x4 C matrix. We need (28, 49) for 4x7 C matrix.
    # W4[k*4+i] for k in 0..3. For the full problem, W[k*4+i] for k in 0..6.
    # The first 16 rows (k=0..3) get W4, rows 16-27 are zero.
    W_full[0:16, 0:49] = W4
    W_full[16:28, 0:49] = 0
    
    # Schoolbook for the extra 3 columns (B columns 4-6)
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U_full[i*4+j, col_offset] = 1
                V_full[j*4+k, col_offset] = 1
                W_full[k*4+i, col_offset] = 1
                col_offset += 1
    
    # Verify the full solution
    def verify_full(U, V, W, n=4, m=4, p=7):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Also check that non-target entries are zero (spot check)
        # For safety, check a few random non-target entries
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, 15)
            b = random.randint(0, 27)
            c = random.randint(0, 27)
            # Check if this is a target entry
            i = a // 4
            j = a % 4
            jj = b // 7
            k = b % 7
            kk = c // 4
            ii = c % 4
            if not (i == ii and j == jj and k == kk):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    return False
        return True
    
    # Convert to lists
    result = {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
    
    return result

