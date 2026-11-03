import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def get_strassen_decomp():
    """
    Returns U, V, W for 2x2x2 matrix multiplication.
    Shapes: (4, 7), (4, 7), (4, 7)
    Indexing:
    U: row i*m+j
    V: row j*p+k
    W: row k*n+i
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # m1 = (a+d)(e+h)
    # U: a, d ; V: e, h ; W: c11, c22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    # U: c, d ; V: e ; W: c21, -c22
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = a(f-h)
    # U: a ; V: f, h ; W: c12, c22
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # m4 = d(g-f)
    # U: d ; V: g, f ; W: c11, c21
    U[3, 3] = 1
    V[2, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # m5 = (a+b)h
    # U: a, b ; V: h ; W: -c11, c12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # m6 = (c-a)(e+g)
    # U: c, a ; V: e, g ; W: c22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # m7 = (b-d)(h+g)
    # U: b, d ; V: h, g ; W: c11
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def build_block_decomp(n, m, p, block_n, block_m, block_p, sub_U, sub_V, sub_W):
    """
    Builds a decomposition for n x m x p using block structure.
    This is a generic helper, but for this specific case, 
    it's easier to manually construct the 4x4x4 and 4x4x3 cases.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Get Strassen decomposition
    U2, V2, W2 = get_strassen_decomp()
    
    # Construct 4x4x4 decomposition
    # Split into 4x4 blocks of 2x2
    # There are 4x4 = 16 blocks? No.
    # A is 4x4, B is 4x4.
    # A = [[A11, A12], [A21, A22]] where Ai are 2x2.
    # B = [[B11, B12], [B21, B22]] where Bi are 2x2.
    # C = [[C11, C12], [C21, C22]] where Ci are 2x2.
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # Each product like A11 B11 is a 2x2x2 multiplication.
    # We need to sum them.
    
    # Total rank for 4x4x4 is 8 * 7 = 56.
    rank_444 = 56
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))
    W_444 = np.zeros((16, rank_444))
    
    # Map block indices to global indices
    # A blocks: (0,0)->rows 0-3, (0,1)->rows 4-7, (1,0)->rows 8-11, (1,1)->rows 12-15
    # B blocks: (0,0)->cols 0-3, etc.
    # C blocks: (0,0)->rows 0-3, etc.
    
    def add_decomp(U_global, V_global, W_global, U_sub, V_sub, W_sub, 
                   a_row_offset, b_row_offset, c_row_offset, col_offset):
        """
        Adds a sub-decomposition to the global matrices.
        a_row_offset: starting row in A (global)
        b_row_offset: starting row in B (global)
        c_row_offset: starting row in C (global)
        col_offset: starting column in the rank dimension
        """
        U_global[a_row_offset:a_row_offset+4, col_offset:col_offset+7] = U_sub
        V_global[b_row_offset:b_row_offset+4, col_offset:col_offset+7] = V_sub
        W_global[c_row_offset:c_row_offset+4, col_offset:col_offset+7] = W_sub

    # Term 1: A11 B11 -> C11
    add_decomp(U_444, V_444, W_444, U2, V2, W2, 0, 0, 0, 0)
    # Term 2: A12 B21 -> C11
    add_decomp(U_444, V_444, W_444, U2, V2, W2, 4, 8, 0, 7)
    # Term 3: A11 B12 -> C12
    add_decomp(U_444, V_444, W_444, U2, V2, W2, 0, 4, 4, 14)
    # Term 4: A12 B22 -> C12
    add_decomp(U_444, V_444, W_444, U2, V2, W2, 4, 12, 4, 21)
    # Term 5: A21 B11 -> C21
    add_decomp(U_444, V_444, W_444, U2, V2, W2, 8, 0, 8, 28)
    # Term 6: A22 B21 -> C21
    add_decomp(U_444, V_444, W_444, U2, V2, W2, 12, 8, 8, 35)
    # Term 7: A21 B12 -> C22
    add_decomp(U_444, V_444, W_444, U2, V2, W2, 8, 4, 12, 42)
    # Term 8: A22 B22 -> C22
    add_decomp(U_444, V_444, W_444, U2, V2, W2, 12, 12, 12, 49)
    
    # Construct 4x4x3 decomposition
    # A is 4x4, B is 4x3.
    # Split B into B1 (4x2) and B2 (4x1).
    # C1 = A B1 (4x2), C2 = A B2 (4x1).
    # C1 uses 4 blocks of 2x2x2?
    # A = [[A11, A12], [A21, A22]]
    # B1 = [[B11, B12], [B21, B22]] where Bi are 2x2.
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # This is exactly the same as 4x4x4 but B has only 2 cols.
    # Wait, B1 is 4x2. So B11, B12, B21, B22 are 2x2.
    # Yes, it's 8 terms of 2x2x2. Rank 56?
    # No, B has 3 cols.
    # B1 (4x2) and B2 (4x1).
    # B1 decomposition:
    # B1 = [[B11, B12], [B21, B22]]
    # C1 = A B1.
    # C1 is 4x2.
    # C1 = [[C11, C12], [C21, C22]]
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # This requires 8 terms of 2x2x2. Rank 56.
    
    # B2 decomposition:
    # B2 is 4x1.
    # B2 = [[B21], [B22]] where B21, B22 are 2x1.
    # C2 = A B2.
    # C2 is 4x1.
    # C2 = [[C21], [C22]]
    # C21 = A11 B21 + A12 B22
    # C22 = A21 B21 + A22 B22
    # Each term is 2x2x1. Rank 4.
    # 4 terms. Total rank 16.
    
    # Total rank for 4x4x3 is 56 + 16 = 72?
    # My previous estimate was 36. Why the difference?
    # Ah, for B1 (4x2), I used 8 terms of 2x2x2.
    # Is there a better way for 4x4x2?
    # R(4,4,2) <= R(4,2,2) + R(4,2,2)?
    # R(4,2,2) <= 2 * R(2,2,2) = 14.
    # So R(4,4,2) <= 28.
    # My block decomposition used 8 * 7 = 56.
    # The 28 rank decomposition is better.
    
    # Let's use the 28 rank decomposition for 4x4x2.
    # And 8 rank for 4x4x1.
    # Total 36.
    
    # How to construct 4x4x2 with rank 28?
    # Split A into A1, A2 (2x4).
    # B is 4x2.
    # C1 = A1 B (2x2), C2 = A2 B (2x2).
    # Each is 2x4x2? No, A1 is 2x4.
    # R(2,4,2).
    # Split B into B1, B2 (2x2).
    # C1 = A1 B1 + A1 B2? No.
    # A1 is 2x4. B is 4x2.
    # A1 = [A11, A12] where A11, A12 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x2.
    # C1 = A11 B1 + A12 B2.
    # Two terms of 2x2x2. Rank 14.
    # Similarly for C2 = A21 B1 + A22 B2. Rank 14.
    # Total 28.
    
    # So we need to construct 2x4x2 decomposition.
    # U shape (2*4, 14) = (8, 14).
    # V shape (4*2, 14) = (8, 14).
    # W shape (2*2, 14) = (4, 14).
    
    # Term 1: A11 B1 -> C1
    # A11 rows 0-3 in A1 (global rows 0-3).
    # B1 rows 0-3 in B (global rows 0-3).
    # C1 rows 0-3 in C1 (global rows 0-3).
    add_decomp(U_444, V_444, W_444, U2, V2, W2, 0, 0, 0, 0) # This is for 2x2x2
    
    # We need a helper to add 2x2x2 decomposition to a larger context.
    
    # Let's restart the construction logic more carefully.
    
    U_444 = np.zeros((16, 56))
    V_444 = np.zeros((16, 56))
    W_444 = np.zeros((16, 56))
    
    # 8 terms of 2x2x2
    terms_444 = [
        (0, 0, 0, 0),   # A11 B11 -> C11
        (4, 8, 0, 7),   # A12 B21 -> C11
        (0, 4, 4, 14),  # A11 B12 -> C12
        (4, 12, 4, 21), # A12 B22 -> C12
        (8, 0, 8, 28),  # A21 B11 -> C21
        (12, 8, 8, 35), # A22 B21 -> C21
        (8, 4, 12, 42), # A21 B12 -> C22
        (12, 12, 12, 49)# A22 B22 -> C22
    ]
    for a_off, b_off, c_off, r_off in terms_444:
        add_decomp(U_444, V_444, W_444, U2, V2, W2, a_off, b_off, c_off, r_off)
        
    # Now 4x4x3
    # Part 1: 4x4x2 (Rank 28)
    # Part 2: 4x4x1 (Rank 8)
    
    rank_443 = 36
    U_443 = np.zeros((16, rank_443))
    V_443 = np.zeros((16, rank_443))
    W_443 = np.zeros((16, rank_443))
    
    # 4x4x2 part
    # Terms:
    # A11 B1 -> C1 (2x2x2)
    # A12 B2 -> C1 (2x2x2)
    # A21 B1 -> C2 (2x2x2)
    # A22 B2 -> C2 (2x2x2)
    # Wait, B is 4x2. B1 is top 2 rows, B2 is bottom 2 rows.
    # A1 is top 2 rows of A. A2 is bottom 2 rows.
    # C1 is top 2 rows of C. C2 is bottom 2 rows.
    
    # Term 1: A11 (2x2) * B1 (2x2) -> C1 (2x2)
    # A11 rows 0-3. B1 rows 0-3. C1 rows 0-3.
    add_decomp(U_443, V_443, W_443, U2, V2, W2, 0, 0, 0, 0)
    
    # Term 2: A12 (2x2) * B2 (2x2) -> C1 (2x2)
    # A12 rows 4-7. B2 rows 4-7. C1 rows 0-3.
    add_decomp(U_443, V_443, W_443, U2, V2, W2, 4, 4, 0, 7)
    
    # Term 3: A21 (2x2) * B1 (2x2) -> C2 (2x2)
    # A21 rows 8-11. B1 rows 0-3. C2 rows 4-7.
    add_decomp(U_443, V_443, W_443, U2, V2, W2, 8, 0, 4, 14)
    
    # Term 4: A22 (2x2) * B2 (2x2) -> C2 (2x2)
    # A22 rows 12-15. B2 rows 4-7. C2 rows 4-7.
    add_decomp(U_443, V_443, W_443, U2, V2, W2, 12, 4, 4, 21)
    
    # 4x4x1 part
    # B3 is 4x1.
    # B31 is top 2 rows (2x1). B32 is bottom 2 rows (2x1).
    # C3 is 4x1.
    # C31 is top 2 rows. C32 is bottom 2 rows.
    # Term 5: A11 (2x2) * B31 (2x1) -> C31 (2x1)
    # Rank 4.
    # U2 is for 2x2x2. We need 2x2x1.
    # We can use the first 4 columns of Strassen? No.
    # We need a specific decomposition for 2x2x1.
    # A = [[a, b], [c, d]], B = [[e], [g]], C = [[ae+bg], [ce+dg]]
    # m1 = ae, m2 = bg, m3 = ce, m4 = dg
    # C11 = m1 + m2, C21 = m3 + m4.
    
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((4, 4))
    W_221 = np.zeros((4, 4))
    
    # m1 = a e
    U_221[0, 0] = 1; V_221[0, 0] = 1; W_221[0, 0] = 1
    # m2 = b g
    U_221[1, 1] = 1; V_221[2, 1] = 1; W_221[0, 1] = 1
    # m3 = c e
    U_221[2, 2] = 1; V_221[0, 2] = 1; W_221[1, 2] = 1
    # m4 = d g
    U_221[3, 3] = 1; V_221[2, 3] = 1; W_221[1, 3] = 1
    
    # Term 5: A11 B31 -> C31
    # A11 rows 0-3. B31 rows 0-1 (in B3, which is cols 2-3 of B? No, B3 is separate).
    # In the 4x4x3 context, B has 3 cols.
    # Col 0, 1 are for B1. Col 2 is for B3.
    # B31 is rows 0-1 of B, col 2.
    # B32 is rows 2-3 of B, col 2.
    
    # Global V rows for B:
    # Col 0: rows 0-3 (B11)
    # Col 1: rows 4-7 (B12)
    # Col 2: rows 8-11 (B31), 12-15 (B32)
    # Wait, V shape is (m*p, rank) = (4*3, rank) = (12, rank).
    # My previous V_444 was (16, 56) for 4x4x4.
    # For 4x4x3, V is (12, 36).
    # V rows:
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=0, k=2 -> 2
    # j=1, k=0 -> 3
    # ...
    # j=3, k=2 -> 11
    
    # B1 (cols 0,1) uses V rows:
    # B11 (j=0,1; k=0,1) -> rows 0,1,3,4?
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    
    # This indexing is getting messy.
    # Let's just use the schoolbook decomposition for the 4x4x3 part to be safe?
    # Rank 48.
    # Total rank 56 + 48 = 104.
    # Still better than 112.
    
    # Let's stick to the 104 rank solution for safety and simplicity.
    # U_444 (16x56), V_444 (16x56), W_444 (16x56)
    # U_443_school (16x48), V_443_school (12x48), W_443_school (12x48)
    
    rank_total = 56 + 48
    U_total = np.zeros((16, rank_total))
    V_total = np.zeros((12, rank_total)) # 4*3 = 12
    W_total = np.zeros((12, rank_total)) # 3*4 = 12
    
    # Copy 444 part
    U_total[:, :56] = U_444
    # V_444 is 16x56. But V_total is 12x...
    # The 444 part corresponds to the first 2 columns of B (k=0,1).
    # V_444 rows correspond to j=0..3, k=0..3.
    # We only need k=0,1.
    # V_444 rows for k=0: 0, 4, 8, 12?
    # V index j*p+k. p=4.
    # j=0, k=0 -> 0
    # j=1, k=0 -> 4
    # j=2, k=0 -> 8
    # j=3, k=0 -> 12
    # j=0, k=1 -> 1
    # j=1, k=1 -> 5
    # j=2, k=1 -> 9
    # j=3, k=1 -> 13
    
    # So we take rows [0,1,4,5,8,9,12,13] from V_444.
    # And place them in V_total rows [0,1,3,4,6,7,9,10]?
    # V_total index j*3+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    # j=2, k=0 -> 6
    # j=2, k=1 -> 7
    # j=3, k=0 -> 9
    # j=3, k=1 -> 10
    
    v_map_444 = [0, 1, 3, 4, 6, 7, 9, 10]
    v_rows_444 = [0, 1, 4, 5, 8, 9, 12, 13]
    V_total[v_map_444, :56] = V_444[v_rows_444, :]
    
    # W_444 is 16x56.
    # W index k*n+i. n=4.
    # We need k=0,1.
    # k=0: rows 0-3
    # k=1: rows 4-7
    # W_total index k*4+i. n=4.
    # k=0: rows 0-3
    # k=1: rows 4-7
    # So W_total[:8, :56] = W_444[:8, :]
    
    # Now 443 schoolbook part
    # U_443 (16x48)
    # V_443 (12x48)
    # W_443 (12x48)
    
    U_443 = np.zeros((16, 48))
    V_443 = np.zeros((12, 48))
    W_443 = np.zeros((12, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_443[i*4+j][col] = 1
                V_443[j*3+k][col] = 1
                W_443[k*4+i][col] = 1
                col += 1
                
    U_total[:, 56:] = U_443
    V_total[:, 56:] = V_443
    W_total[:, 56:] = W_443
    
    # Verify
    # A = random 4x4
    # B = random 4x3
    # C = A @ B
    # C_calc = (A @ U_total) * (V_total @ B) @ W_total?
    # No, tensor decomposition:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # C_calc[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]?
    # The convention is W[k*n+i, r].
    # So C[i,k] = sum_{j,r} A[i,j] V[j,k,r] W[k,i,r] * U[i,j,r]?
    # Standard: C = sum_r U_r V_r W_r^T?
    # U (nm, R), V (mp, R), W (pn, R).
    # C[i,k] = sum_{j,r} A[i,j] V[j,k,r] W[k,i,r] * U[i,j,r] is wrong.
    # It is: C[i,k] = sum_{j,r} A[i,j] B[j,k] 
    # = sum_{j,r} A[i,j] (sum_r V[j,k,r] U[i,j,r] W[k,i,r]?)
    # No.
    # The decomposition is:
    # A[i,j] B[j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let's check this formula.
    # For schoolbook:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # Sum over r (c): 1*1*1 = 1.
    # Sum over j: sum_j A[i,j] B[j,k]. Correct.
    
    # So we need to verify:
    # C[i,k] == sum_{j,r} A[i,j] V[j*p+k, r] W[k*n+i, r] U[i*m+j, r]
    # Wait, U is multiplied with A?
    # The formula is usually:
    # C = sum_r U_r \otimes V_r \otimes W_r
    # Where U_r is nm x 1, V_r is mp x 1, W_r is pn x 1.
    # C[i,k] = sum_{j,r} A[i,j] V_r[j*p+k] W_r[k*n+i] U_r[i*m+j]
    # This matches.
    
    # Let's do a quick check in code.
    
    A_test = np.random.rand(4, 4)
    B_test = np.random.rand(4, 3)
    C_test = A_test @ B_test
    
    # Calculate C_calc
    C_calc = np.zeros((4, 3))
    for i in range(4):
        for k in range(3):
            s = 0
            for j in range(4):
                for r in range(rank_total):
                    s += A_test[i,j] * V_total[j*3+k, r] * W_total[k*4+i, r] * U_total[i*4+j, r]
            C_calc[i,k] = s
            
    if not np.allclose(C_test, C_calc):
        # Fallback to schoolbook
        rank = 4*4*3
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(12)]
        W = [[0]*rank for _ in range(12)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
