import numpy as np
from fractions import Fraction
import time
import sys

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 140  # safety margin
    
    # Deterministic construction: Strassen (rank 7) for 2x2x2 + trivial (rank 3) for 2x2x3
    # Per 2x2x5 block: rank 10. Two blocks (A1*B1 + A2*B2): total rank 20.
    #
    # Strassen 2x2x2 decomposition (7 terms):
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B22)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #   C11 = M1+M4-M5+M7
    #   C12 = M3+M5
    #   C21 = M2+M4
    #   C22 = M1-M2+M3+M6
    #
    # Trivial 2x2x3 (3 columns): rank 3
    #   For each column l: C[i,l] = sum_j A[i,j]*B[j,l]
    #   Use 2 terms per column? No, trivial is 1 term per (i,j,l) but we can group.
    #   Actually for 2x2x3, trivial rank = 2*2*3 = 12? No.
    #   Wait: for a single 2x2x3 tensor, the schoolbook rank is 2*2*3=12? No!
    #   The tensor T[i,j,l] for i in {0,1}, j in {0,1}, l in {0,1,2} has 2*2*3=12 entries.
    #   Schoolbook: for each (i,j,l), one term. So rank 12.
    #   But we can do better: for fixed j, sum over i and l.
    #   Actually, the minimal rank of 2x2x3 is 7 (Strassen for 2x2x2 gives 7 for 2x2x2, 
    #   and 2x2x3 needs at least 7... actually it's known that R(2,2,3)=7).
    #   But let's just use schoolbook for the 3 columns: rank = 2*2*3 = 12? No wait.
    #   
    #   Let me reconsider. For 2x2x5:
    #   Option 1: Strassen(2x2x2, rank 7) + schoolbook(2x2x3, rank 12) = 19
    #   Option 2: schoolbook(2x2x5, rank 20) = 20
    #   So option 1 gives rank 19 per block, total 38.
    #   
    #   But the cell finding says rank 36 works. Let me think again.
    #   
    #   Actually, for 2x2x3, we can use a smarter decomposition.
    #   The 2x2x3 tensor: T[i,j,l] = delta_{i,j} for each l... no.
    #   T[i,j,l] = 1 if the (i,j) entry of A and (j,l) entry of B contribute to C[i,l].
    #   For 2x2x3: C[i,l] = A[i,0]*B[0,l] + A[i,1]*B[1,l] for i in {0,1}, l in {0,1,2}
    #   That's 6 outputs, each is a dot product of length 2.
    #   Schoolbook: 2*2*3 = 12 terms.
    #   Can we do better? Yes! For each l, C[:,l] = A * B[:,l] (2x2 matrix times 2x1 vector).
    #   Strassen for 2x2x1? That's just 4 terms (schoolbook) or we can use Strassen on 2x2x2
    #   by padding. Actually, 2x2x1 has rank 4 (it's just 4 independent terms).
    #   So 2x2x3 = three independent 2x2x1 tensors, each rank 4, total rank 12.
    #   Hmm, that doesn't help.
    #   
    #   Wait, I think the key is: we don't need to decompose 2x2x5 as a standalone tensor.
    #   We can share structure between the two blocks.
    #   
    #   Let me try a different approach: use Strassen on the full 2x4x5 by treating it as
    #   a 2x2x5 where the "2" in the middle is split into two 2x2 blocks.
    #   
    #   Actually, let me just use the known result: R(2,4,5) <= 36 via Strassen + trivial.
    #   The construction: 
    #   - Split A (2x4) into A1 (2x2) and A2 (2x2)
    #   - Split B (4x5) into B1 (2x5) and B2 (2x5) [rows 0-1 and 2-3]
    #   - C = A1*B1 + A2*B2
    #   - For A1*B1 (2x2x5): use Strassen on cols 0-1 (rank 7) + schoolbook on cols 2-4 (rank 6)
    #     Wait, schoolbook for 2x2x3 is 2*2*3=12? No, it's 4*3=12.
    #     Hmm, that gives 7+12=19 per block, total 38.
    #   
    #   Let me just use schoolbook for everything: rank 40. Then try to improve.
    #   
    #   Actually, I recall that R(2,4,5) = 36 is achievable. Let me construct it directly.
    #   
    #   Alternative: use Strassen for 2x2x2 (rank 7) twice (for two 2x2x2 sub-blocks covering
    #   cols 0-1 and cols 2-3 of B) plus schoolbook for the last column (2x2x1, rank 4).
    #   Per 2x2x5 block: 7 + 7 + 4 = 18. Total: 36. Yes!
    #   
    #   So: split B's 5 columns into {0,1}, {2,3}, {4}.
    #   For each 2x2x2 sub-block: Strassen (rank 7)
    #   For the 2x2x1 sub-block: schoolbook (rank 4)
    #   Per 2x2x5: 7+7+4 = 18
    #   Two 2x2x5 blocks (A1*B1 + A2*B2): 18+18 = 36

    # Build the rank-36 decomposition
    # Total rank: 36
    # Structure:
    #   Block 1 (A1 * B1): 18 terms
    #     - Strassen on cols 0-1: 7 terms (indices 0-6)
    #     - Strassen on cols 2-3: 7 terms (indices 7-13)
    #     - Schoolbook on col 4: 4 terms (indices 14-17)
    #   Block 2 (A2 * B2): 18 terms
    #     - Strassen on cols 0-1: 7 terms (indices 18-24)
    #     - Strassen on cols 2-3: 7 terms (indices 25-31)
    #     - Schoolbook on col 4: 4 terms (indices 32-35)
    
    rank_total = 36
    U = np.zeros((n*m, rank_total))  # (8, 36)
    V = np.zeros((m*p, rank_total))  # (20, 36)
    W = np.zeros((p*n, rank_total))  # (10, 36)
    
    def strassen_block(U_base, V_base, W_base, a_row_offset, b_row_offset, col_offset, term_offset):
        """
        Add Strassen 2x2x2 terms for A[a_row_offset:a_row_offset+2, :] * B[b_row_offset:b_row_offset+2, col_offset:col_offset+2]
        U_base: row offset in U for the 2x2 A block
        V_base: row offset in V for the 2x2 B block  
        W_base: row offset in W for the 2x2 C block
        a_row_offset: which rows of A (0 or 2 for the two 2x2 blocks)
        b_row_offset: which rows of B (0 or 2)
        col_offset: which columns of B (0, 2, or 4)
        term_offset: starting index in the rank dimension
        """
        # Strassen terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # U entries: U[i*4 + j, k] where i is row of A (0,1), j is col of A within 2x2 block
        # For block a_row_offset, the A rows are a_row_offset and a_row_offset+1
        # U index: i*4 + (j + a_row_offset) for i in {0,1}, j in {0,1}
        # Wait, U has shape (8, rank). U[i*m + j, k] = U[i*4 + j, k]
        # For A1 (j in 0,1): U[0*4+0], U[0*4+1], U[1*4+0], U[1*4+1] = U[0], U[1], U[4], U[5]
        # For A2 (j in 2,3): U[0*4+2], U[0*4+3], U[1*4+2], U[1*4+3] = U[2], U[3], U[6], U[7]
        
        # V entries: V[j*p + l, k] = V[j*5 + l, k]
        # For B1 (j in 0,1): V[0*5+col_offset], V[0*5+col_offset+1], V[1*5+col_offset], V[1*5+col_offset+1]
        # For B2 (j in 2,3): V[2*5+col_offset], V[2*5+col_offset+1], V[3*5+col_offset], V[3*5+col_offset+1]
        
        # W entries: W[l*n + i, k] = W[l*2 + i, k]
        # For C block: i in {0,1}, l in {col_offset, col_offset+1}
        # W[col_offset*2 + 0], W[col_offset*2 + 1], W[(col_offset+1)*2 + 0], W[(col_offset+1)*2 + 1]
        
        # A11 = A[0, a_row_offset], A12 = A[0, a_row_offset+1]
        # A21 = A[1, a_row_offset], A22 = A[1, a_row_offset+1]
        # B11 = B[b_row_offset, col_offset], B12 = B[b_row_offset, col_offset+1]
        # B21 = B[b_row_offset+1, col_offset], B22 = B[b_row_offset+1, col_offset+1]
        
        # U indices:
        # A11: U[0*4 + a_row_offset] = U[a_row_offset]
        # A12: U[0*4 + a_row_offset + 1] = U[a_row_offset + 1]
        # A21: U[1*4 + a_row_offset] = U[4 + a_row_offset]
        # A22: U[1*4 + a_row_offset + 1] = U[4 + a_row_offset + 1]
        
        # V indices:
        # B11: V[b_row_offset*5 + col_offset]
        # B12: V[b_row_offset*5 + col_offset + 1]
        # B21: V[(b_row_offset+1)*5 + col_offset]
        # B22: V[(b_row_offset+1)*5 + col_offset + 1]
        
        # W indices:
        # C11: W[col_offset*2 + 0]
        # C12: W[col_offset*2 + 1]
        # C21: W[(col_offset+1)*2 + 0]
        # C22: W[(col_offset+1)*2 + 1]
        
        # M1 = (A11+A22)(B11+B22)
        k = term_offset
        U[a_row_offset, k] += 1
        U[4 + a_row_offset + 1, k] += 1
        V[b_row_offset*5 + col_offset, k] += 1
        V[(b_row_offset+1)*5 + col_offset + 1, k] += 1
        W[col_offset*2 + 0, k] += 1
        W[(col_offset+1)*2 + 1, k] += 1
        
        # M2 = (A21+A22)B11
        k = term_offset + 1
        U[4 + a_row_offset, k] += 1
        U[4 + a_row_offset + 1, k] += 1
        V[b_row_offset*5 + col_offset, k] += 1
        W[(col_offset+1)*2 + 0, k] += 1
        W[(col_offset+1)*2 + 1, k] -= 1
        
        # M3 = A11(B21-B22)
        k = term_offset + 2
        U[a_row_offset, k] += 1
        V[(b_row_offset+1)*5 + col_offset, k] += 1
        V[(b_row_offset+1)*5 + col_offset + 1, k] -= 1
        W[col_offset*2 + 1, k] += 1
        W[(col_offset+1)*2 + 1, k] += 1
        
        # M4 = A22(B12-B22)
        k = term_offset + 3
        U[4 + a_row_offset + 1, k] += 1
        V[b_row_offset*5 + col_offset + 1, k] += 1
        V[(b_row_offset+1)*5 + col_offset + 1, k] -= 1
        W[col_offset*2 + 0, k] += 1
        W[(col_offset+1)*2 + 0, k] += 1
        
        # M5 = (A11+A12)B22
        k = term_offset + 4
        U[a_row_offset, k] += 1
        U[a_row_offset + 1, k] += 1
        V[(b_row_offset+1)*5 + col_offset + 1, k] += 1
        W[col_offset*2 + 0, k] -= 1
        W[col_offset*2 + 1, k] += 1
        
        # M6 = (A21-A11)(B11+B12)
        k = term_offset + 5
        U[4 + a_row_offset, k] += 1
        U[a_row_offset, k] -= 1
        V[b_row_offset*5 + col_offset, k] += 1
        V[b_row_offset*5 + col_offset + 1, k] += 1
        W[(col_offset+1)*2 + 1, k] += 1
        
        # M7 = (A12-A22)(B21+B22)
        k = term_offset + 6
        U[a_row_offset + 1, k] += 1
        U[4 + a_row_offset + 1, k] -= 1
        V[(b_row_offset+1)*5 + col_offset, k] += 1
        V[(b_row_offset+1)*5 + col_offset + 1, k] += 1
        W[col_offset*2 + 0, k] += 1
        W[(col_offset+1)*2 + 1, k] += 1
        
        # Verify: C11 = M1+M4-M5+M7
        # C11 contributions: W[col_offset*2+0] gets: M1(+1), M4(+1), M5(-1), M7(+1) = 1+1-1+1 = 2? That's wrong.
        # Let me recheck Strassen:
        # C11 = M1 + M4 - M5 + M7
        # So W[col_offset*2+0] should get: M1: +1, M4: +1, M5: -1, M7: +1
        # But that sums to 2, not 1. I think I have the Strassen formulas wrong.
        # 
        # Correct Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # 
        # Let me verify C11: 
        # M1 = A11B11 + A11B22 + A22B11 + A22B22
        # M4 = A22B12 - A22B22
        # M5 = A11B22 + A12B22
        # M7 = A12B21 + A12B22 - A22B21 - A22B22
        # M1+M4-M5+M7 = A11B11 + A11B22 + A22B11 + A22B22 + A22B12 - A22B22 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
        # = A11B11 + A22B11 + A22B12 - A22B21 - A22B22 + A12B21
        # That's not A11B11 + A12B21. Something is wrong.
        # 
        # I think I'm confusing myself. Let me use the standard Strassen more carefully.
        # Standard Strassen for C = AB where A, B are 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # 
        # Let me expand C11 = M1 + M4 - M5 + M7:
        # M1 = A11B11 + A11B22 + A22B11 + A22B22
        # M4 = A22B12 - A22B22
        # -M5 = -A11B22 - A12B22
        # M7 = A12B21 + A12B22 - A22B21 - A22B22
        # Sum: A11B11 + A11B22 + A22B11 + A22B22 + A22B12 - A22B22 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
        # = A11B11 + A22B11 + A22B12 - A22B21 - A22B22 + A12B21
        # This should equal A11B11 + A12B21.
        # The extra terms: A22B11 + A22B12 - A22B21 - A22B22 don't cancel. 
        # I think the issue is that Strassen works for square matrices and the indices are different.
        # 
        # Actually, I think I'm making an error. In Strassen, A and B are 2x2, and C = AB.
        # C11 = A11B11 + A12B21
        # Let me recheck M7: M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
        # And the sum should work out. Let me just trust the standard formulas and implement them,
        # then verify numerically.
        
        pass  # This function is too complex to get right in one pass. Let me use a simpler approach.
    
    # Given the complexity, let me just use schoolbook (rank 40) as a safe fallback,
    # and then try to verify if a rank-36 construction is feasible.
    # 
    # Actually, let me just do schoolbook. It's guaranteed correct.
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    s = 0
                    for k in range(r):
                        s += U[i*m+j, k] * V[j*p+l, k] * W[l*n+i, k]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W):
        raise RuntimeError("Schoolbook verification failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
