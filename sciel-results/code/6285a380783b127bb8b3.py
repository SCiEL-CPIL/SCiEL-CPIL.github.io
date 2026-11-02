import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Keep safety margin
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Use Strassen-based decomposition for rank reduction.
    # Strategy: Split A (2x4) into two 2x2 blocks: A = [A1, A2]
    # Split B (4x6) into blocks: B = [[B11, B12], [B21, B22]] where each is 2x3
    # C = A1*B11 + A2*B21 (cols 0-2) and A1*B12 + A2*B22 (cols 3-5)
    #
    # For each 2x2 * 2x3 multiplication, we can use a rank-9 decomposition.
    # The 2x2x3 tensor has known rank 9.
    #
    # We'll build a rank-18 decomposition: 9 terms for left half, 9 for right half.
    #
    # Actually, let's use a more direct approach with Strassen on 2x2 sub-blocks.
    # Pad the problem to use Strassen's 7-term formula.
    #
    # Alternative: use the known optimal decomposition for M<2,4,6>.
    # Let's try to construct a rank-18 decomposition directly.
    
    # Rank-9 decomposition for 2x2 * 2x3 matrix multiplication.
    # The 2x2x3 tensor T[a,b,c] where a in {0,1} (row of A), b in {0,1,2,3} (col of B, split as 2x2), c in {0,1,2} (col of C)
    # Actually for 2x2 * 2x3: A is 2x2, B is 2x3, C is 2x3.
    # T[a, b, c] = 1 if a=i, b=j*3+k... no.
    # A[i,j] * B[j,k] -> C[i,k] where i in 0..1, j in 0..1, k in 0..2
    # Index: a = i*2+j, b = j*3+k, c = k*2+i
    
    # Let me just use a direct construction.
    # For 2x2 * 2x3, the rank is 9. Here's one such decomposition:
    # We need 9 rank-1 terms.
    
    def build_2x2x3_decomp():
        """Build a rank-9 decomposition for 2x2 * 2x3 matrix multiplication."""
        # A is 2x2, indexed by (i,j) -> a = i*2+j
        # B is 2x3, indexed by (j,k) -> b = j*3+k  
        # C is 2x3, indexed by (i,k) -> c = k*2+i
        # T[a,b,c] = 1 if a=i*2+j, b=j*3+k, c=k*2+i
        
        U_dim_sub = 4  # 2*2
        V_dim_sub = 6  # 2*3
        W_dim_sub = 6  # 3*2
        
        T_sub = np.zeros((U_dim_sub, V_dim_sub, W_dim_sub), dtype=np.int32)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a = i * 2 + j
                    b = j * 3 + k
                    c = k * 2 + i
                    T_sub[a, b, c] = 1
        
        # Rank-9 decomposition for 2x2x3
        # Using known structure:
        # Terms 1-4: schoolbook for first 2 cols of B (2x2 part)
        # Terms 5-9: handle the 3rd column
        
        # Actually, let's use a systematic approach.
        # The 2x2x3 tensor can be decomposed as:
        # For the 2x2 * 2x2 part (first 2 cols), use Strassen (rank 7)
        # Plus 2 more terms for the 3rd column
        # Total: 7 + 2 = 9
        
        # Strassen for 2x2*2x2:
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B10-B11)
        # m4 = A11(B01-B10)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B10)
        # m7 = (A01-A11)(B01+B11)
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # For 2x2 * 2x3, B has columns b0, b1, b2 (each 2x1)
        # C has columns c0, c1, c2 (each 2x1)
        # c0 = A * b0, c1 = A * b1, c2 = A * b2
        
        # Use Strassen for c0 and c1 (treating [b0,b1] as 2x2 B), giving 7 terms
        # For c2 = A * b2, use 2 terms (schoolbook for 2x1)
        # Total: 9 terms
        
        rank_sub = 9
        U_sub = np.zeros((U_dim_sub, rank_sub), dtype=np.int32)
        V_sub = np.zeros((V_dim_sub, rank_sub), dtype=np.int32)
        W_sub = np.zeros((W_dim_sub, rank_sub), dtype=np.int32)
        
        # Strassen terms for first 2 columns of B (indices 0,1 in B)
        # B indices: b = j*3+k, so for k=0: b=j, for k=1: b=j+2, for k=2: b=j+4
        # B00 = b[0], B01 = b[2], B10 = b[1], B11 = b[3]
        # C indices: c = k*2+i, so for k=0: c=i, for k=1: c=i+2, for k=2: c=i+4
        # C00 = c[0], C01 = c[2], C10 = c[1], C11 = c[3]
        
        # Strassen m1 = (A00+A11)(B00+B11)
        U_sub[0, 0] = 1  # A00
        U_sub[3, 0] = 1  # A11
        V_sub[0, 0] = 1  # B00
        V_sub[3, 0] = 1  # B11
        W_sub[0, 0] = 1  # C00
        W_sub[3, 0] = 1  # C11
        W_sub[1, 0] = -1  # -C10... wait, let me recheck Strassen
        
        # Standard Strassen:
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B10-B11)
        # m4 = A11(B01-B10)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B10)
        # m7 = (A01-A11)(B01+B11)
        
        # Let me redo this properly.
        U_sub = np.zeros((U_dim_sub, rank_sub), dtype=np.int32)
        V_sub = np.zeros((V_dim_sub, rank_sub), dtype=np.int32)
        W_sub = np.zeros((W_dim_sub, rank_sub), dtype=np.int32)
        
        # Term 0: m1 = (A00+A11)(B00+B11)
        U_sub[0, 0] = 1; U_sub[3, 0] = 1
        V_sub[0, 0] = 1; V_sub[3, 0] = 1
        W_sub[0, 0] = 1; W_sub[3, 0] = 1  # C00 += m1, C11 += m1
        W_sub[1, 0] = -1  # C10 -= m1? No...
        
        # I'm getting confused. Let me just use schoolbook for simplicity and accept rank 12 (4*3) for 2x2x3.
        # Then total rank = 2 * 12 = 24 for the full problem.
        # That's still better than 48.
        
        # Schoolbook for 2x2 * 2x3: 2*2*3 = 12 terms
        U_sub = np.zeros((U_dim_sub, 12), dtype=np.int32)
        V_sub = np.zeros((V_dim_sub, 12), dtype=np.int32)
        W_sub = np.zeros((W_dim_sub, 12), dtype=np.int32)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_sub[i*2+j, col] = 1
                    V_sub[j*3+k, col] = 1
                    W_sub[k*2+i, col] = 1
                    col += 1
        
        return U_sub, V_sub, W_sub
    
    # Build the full 2x4 * 4x6 decomposition
    # A (2x4) = [A_left (2x2), A_right (2x2)]
    # B (4x6) = [[B_tl (2x3), B_tr (2x3)],
    #            [B_bl (2x3), B_br (2x3)]]
    # C (2x6) = [A_left * B_tl + A_right * B_bl, A_left * B_tr + A_right * B_br]
    
    # Each 2x2 * 2x3 uses 12 terms (schoolbook)
    # Total: 4 * 12 = 48... that's the same as schoolbook!
    
    # We need to share terms. The key insight is that A_left and A_right are shared.
    # 
    # Let me try a different blocking.
    # 
    # Alternative: Use Strassen's algorithm on the full problem by padding.
    # Pad A from 2x4 to 4x4, pad B from 4x6 to 4x8, compute 4x4*4x8, extract.
    # 4x4*4x8 can be done with 2 Strassen 4x4*4x4 blocks, each of rank 7^2=49... no.
    # 
    # Let me just go with a direct rank-42 construction using known Strassen sub-decompositions.
    
    # Simplest improvement: use Strassen on 2x2 sub-blocks where possible.
    # 
    # For the 2x4 * 4x6 problem, split into 2 sub-problems:
    # Left 3 cols: A(2x4) * B_left(4x3) -> C_left(2x3)
    # Right 3 cols: A(2x4) * B_right(4x3) -> C_right(2x3)
    #
    # Each 2x4 * 4x3 can be further split:
    # A = [A1(2x2), A2(2x2)], B_left = [B1(2x3); B2(2x3)]
    # C_left = A1*B1 + A2*B2
    #
    # Each 2x2*2x3 has rank 9 (known optimal).
    # So C_left needs rank 9+9=18, C_right needs rank 9+9=18.
    # Total: 36.
    
    # Let me construct rank-9 decompositions for 2x2*2x3.
    
    def build_2x2x3_rank9():
        """Rank-9 decomposition for 2x2 * 2x3."""
        U_dim_s = 4
        V_dim_s = 6
        W_dim_s = 6
        rank_s = 9
        
        U_s = np.zeros((U_dim_s, rank_s), dtype=np.int32)
        V_s = np.zeros((V_dim_s, rank_s), dtype=np.int32)
        W_s = np.zeros((W_dim_s, rank_s), dtype=np.int32)
        
        # Use Strassen for the 2x2*2x2 part (first 2 cols of B), plus 2 terms for 3rd col.
        # Strassen indices for A: A00=0, A01=1, A10=2, A11=3
        # B (2x2 part): B00=0, B01=2, B10=1, B11=3 (b=j*3+k, k=0,1)
        # B 3rd col: B02=4, B12=5 (k=2)
        # C: C00=0, C01=2, C10=1, C11=3 (c=k*2+i, k=0,1), C02=4, C12=5 (k=2)
        
        # Strassen m1-m7 for 2x2*2x2:
        # m1 = (A00+A11)(B00+B11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1  # C00, C11 get +m1
        
        # m2 = (A10+A11)B00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[1, 1] = 1; W_s[3, 1] = -1  # C10 += m2, C11 -= m2
        
        # m3 = A00(B10-B11)
        U_s[0, 2] = 1
        V_s[1, 2] = 1; V_s[3, 2] = -1
        W_s[2, 2] = 1; W_s[3, 2] = 1  # C01 += m3, C11 += m3
        
        # m4 = A11(B01-B10)
        U_s[3, 3] = 1
        V_s[2, 3] = 1; V_s[1, 3] = -1
        W_s[0, 3] = 1; W_s[1, 3] = 1  # C00 += m4, C10 += m4
        
        # m5 = (A00+A01)B11
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[2, 4] = 1  # C00 -= m5, C01 += m5
        
        # m6 = (A10-A00)(B00+B10)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[3, 5] = 1  # C11 += m6
        
        # m7 = (A01-A11)(B01+B11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1  # C00 += m7
        
        # Verify Strassen part: C00 = m1+m4-m5+m7, C01=m3+m5, C10=m2+m4, C11=m1-m2+m3+m6
        # Check C00: W_s[0,:] = [1,0,0,1,-1,0,1] -> m1+m4-m5+m7 ✓
        # Check C01: W_s[2,:] = [0,0,1,0,1,0,0] -> m3+m5 ✓
        # Check C10: W_s[1,:] = [0,1,0,1,0,0,0] -> m2+m4 ✓
        # Check C11: W_s[3,:] = [1,-1,1,0,0,1,0] -> m1-m2+m3+m6 ✓
        
        # Now handle 3rd column of B (indices 4,5 in B): C02 = A00*B02 + A01*B12, C12 = A10*B02 + A11*B12
        # Term 7: A00*B02
        U_s[0, 7] = 1
        V_s[4, 7] = 1
        W_s[4, 7] = 1  # C02
        
        # Term 8: A01*B12
        U_s[1, 8] = 1
        V_s[5, 8] = 1
        W_s[4, 8] = 1  # C02
        
        # Wait, C12 also needs terms. Let me add them.
        # Actually I need 4 more terms for C12, but I only have 9 total.
        # Let me reconsider.
        
        # For the 3rd column: C02 = A00*B02 + A01*B12, C12 = A10*B02 + A11*B12
        # That's 4 terms. 7 (Strassen) + 4 = 11, not 9.
        
        # The rank of 2x2x3 is actually 9, so there must be a smarter decomposition.
        # Let me just use 12 terms (schoolbook) for simplicity.
        
        U_s = np.zeros((U_dim_s, 12), dtype=np.int32)
        V_s = np.zeros((V_dim_s, 12), dtype=np.int32)
        W_s = np.zeros((W_dim_s, 12), dtype=np.int32)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U_s[i*2+j, col] = 1
                    V_s[j*3+k, col] = 1
                    W_s[k*2+i, col] = 1
                    col += 1
        
        return U_s, V_s, W_s, 12
    
    # Use schoolbook 2x2*2x3 with 12 terms each.
    # Total: 4 * 12 = 48. Same as before. Not helpful.
    
    # Let me just return the schoolbook decomposition for now, which is rank 48.
    # The structural change should be to use a smarter algorithm.
    
    # For now, return schoolbook to ensure correctness.
    best_rank = n * m * p
    U_school = np.zeros((U_dim, best_rank), dtype=np.int32)
    V_school = np.zeros((V_dim, best_rank), dtype=np.int32)
    W_school = np.zeros((W_dim, best_rank), dtype=np.int32)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
