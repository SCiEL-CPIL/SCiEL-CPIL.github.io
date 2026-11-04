"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c].reshape(n, m)
        v_c = V_arr[:, c].reshape(m, p)
        w_c = W_arr[:, c].reshape(p, n)
        # T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
    
    T_expected = np.ones((n, m, p))
    return np.allclose(T, T_expected, atol=1e-8)

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank

def _strassen_2x2_block(n, m, p):
    """
    Try to use Strassen-like decomposition for 2x2 sub-blocks where possible.
    For a 3x4 * 4x6 matrix multiplication, we can partition into blocks.
    
    Strategy: Use Strassen for 2x2 blocks where dimensions allow, 
    and schoolbook for remaining parts.
    
    Partition A (3x4) into:
      A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x2, A21 is 1x2, A22 is 1x2
    
    Actually, let's think differently. We can split the 4 intermediate dimension:
    A (3x4) = [A_left (3x2) | A_right (3x2)]
    B (4x6) = [B_top (2x6); B_bottom (2x6)]
    
    C = A_left @ B_top + A_right @ B_bottom
    
    Each product is (3x2) @ (2x6) = 3x6, which has rank 3*2*6=36 schoolbook.
    Total: 72. No gain.
    
    Alternative: Use Strassen on a 2x2 sub-problem.
    Consider the top-left 2x2 block of A and top 2 rows of B:
    A_tl (2x2) @ B_tl (2x6) - this is 2x2 @ 2x6
    
    For 2x2 @ 2x2, Strassen uses 7 multiplications.
    For 2x2 @ 2x6, we can split B into three 2x2 blocks:
    B_tl = [B1 B2 B3] each 2x2
    Result = A_tl @ B1 + A_tl @ B2 + A_tl @ B3
    
    Each A_tl @ Bx is 2x2 @ 2x2, which Strassen does in 7 mults.
    So 3 * 7 = 21 mults for the 2x2 @ 2x6 part.
    Schoolbook would be 2*2*6 = 24.
    
    Let me try a more systematic approach.
    """
    # For now, return schoolbook as fallback
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p, rank):
        raise ValueError("Schoolbook verification failed")
    
    # Try to improve using Strassen-like techniques
    # For 3x4 * 4x6, one approach:
    # Split the middle dimension (4) into 2+2
    # A = [A1 | A2] where A1, A2 are 3x2
    # B = [B1; B2] where B1, B2 are 2x6
    # C = A1@B1 + A2@B2
    #
    # Each 3x2 @ 2x6 has tensor rank at most 3*2*6 = 36
    # But we can do better on sub-blocks.
    #
    # Let's try: for each 3x2 @ 2x6 product, split the 3 rows into 2+1
    # and 6 cols into 2+2+2
    #
    # Actually, let's try a direct Strassen embedding.
    # 
    # The key insight: for multiplying (n x m) by (m x p), if we can find
    # a sub-structure that allows Strassen, we save.
    #
    # For 3x4 * 4x6:
    # Let's partition A into rows: A = [A_0; A_1; A_2] where each A_i is 1x4
    # B into cols: B = [B_0 B_1 B_2 B_3 B_4 B_5] where each B_j is 4x1
    # C[i,j] = sum_k A[i,k] * B[k,j]
    #
    # This is just the standard formulation. Let's try to use the fact that
    # we can group terms.
    
    # Try: partition the 4 intermediate indices into pairs
    # k in {0,1,2,3} -> groups {0,1}, {2,3}
    # A = [A_g0 | A_g1] where A_g0 is 3x2, A_g1 is 3x2
    # B = [B_g0; B_g1] where B_g0 is 2x6, B_g1 is 2x6
    # C = A_g0 @ B_g0 + A_g1 @ B_g1
    
    # For each 3x2 @ 2x6, use a decomposition.
    # 3x2 @ 2x6: let's split 3 rows into 2+1, 6 cols into 3+3
    # A_g = [A_top (2x2); A_bot (1x2)]
    # B_g = [B_left (2x3) | B_right (2x3)]
    # Result = A_top@B_left + A_top@B_right + A_bot@B_left + A_bot@B_right
    # = 2x2@2x3 + 2x2@2x3 + 1x2@2x3 + 1x2@2x3
    #
    # 2x2@2x3: split 3 cols into 1+2
    # = 2x2@2x1 + 2x2@2x2
    # 2x2@2x1: schoolbook rank 4, or use... it's just 4 mults
    # 2x2@2x2: Strassen rank 7
    # So 2x2@2x3 has rank 4+7 = 11
    #
    # 1x2@2x3: schoolbook rank 6
    #
    # Total for 3x2@2x6: 11 + 11 + 6 + 6 = 34
    # Two such: 68
    # Schoolbook was 72. We save 4.
    
    # Let me implement this carefully.
    
    def _matmul_2x2_strassen(A, B):
        """
        Strassen decomposition for 2x2 @ 2x2.
        A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-e)
        M5 = (a+b)h
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        Result = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        """
        # Returns list of (coeff_a, coeff_b, coeff_c, coeff_d, coeff_e, coeff_f, coeff_g, coeff_h)
        # for each of the 7 multiplications
        # and the output combination
        mults = [
            # (A coeffs, B coeffs)
            ((1,0,0,1), (1,1,0,0)),   # M1: (a+d)(e+h)
            ((0,0,1,1), (1,0,0,0)),   # M2: (c+d)e
            ((1,0,0,0), (0,1,0,-1)),  # M3: a(f-h)
            ((0,0,0,1), (0,0,1,-1)),  # M4: d(g-e)
            ((1,1,0,0), (0,0,0,1)),   # M5: (a+b)h
            ((-1,0,1,0), (1,1,0,0)),  # M6: (c-a)(e+f)
            ((0,1,0,-1), (0,0,1,1)),  # M7: (b-d)(g+h)
        ]
        # Output: C = [[c11, c12], [c21, c22]]
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        output_coeffs = [
            [1, 0, 0, 1, -1, 0, 1],   # c11
            [0, 0, 1, 0, 1, 0, 0],    # c12
            [0, 1, 0, 1, 0, 0, 0],    # c21
            [1, -1, 1, 0, 0, 1, 0],   # c22
        ]
        return mults, output_coeffs
    
    def _build_strassen_3x2_2x6():
        """
        Build decomposition for 3x2 @ 2x6 using the partitioning above.
        A is 3x2, B is 2x6.
        Split A rows: top 2, bottom 1
        Split B cols: 3 groups of 2
        
        A = [A_top (2x2); A_bot (1x2)]
        B = [B_0 (2x2) B_1 (2x2) B_2 (2x2)]
        
        Result = A_top@B_0 + A_top@B_1 + A_top@B_2 + A_bot@B_0 + A_bot@B_1 + A_bot@B_2
        
        For A_top@B_i (2x2@2x2): use Strassen, rank 7 each
        For A_bot@B_i (1x2@2x2): schoolbook, rank 4 each
        
        Total: 3*7 + 3*4 = 21 + 12 = 33
        """
        # U: 3*2 = 6 entries, V: 2*6 = 12 entries, W: 6*3 = 18 entries
        # We'll build lists of (U_vals, V_vals, W_vals) for each rank-1 component
        
        components = []
        
        # A_top is rows 0,1 of A (3x2), so indices in U: 0,1
        # A_bot is row 2 of A, index in U: 2
        # B has 2 rows, 6 cols. V indices: row 0: 0-5, row 1: 6-11
        # W: 6 cols x 3 rows = 18 entries. W index for (col j, row i): j*3 + i
        
        # Strassen for 2x2 @ 2x2
        mults, output_coeffs = _matmul_2x2_strassen()
        
        for col_group in range(3):
            # B_col_group is columns 2*col_group to 2*col_group+1
            b_col_offset = 2 * col_group
            
            # A_top @ B_col_group: 2x2 @ 2x2 via Strassen
            for mu, (a_coeff, b_coeff) in enumerate(mults):
                # a_coeff: (a, b, c, d) for A_top
                # b_coeff: (e, f, g, h) for B_col_group
                u_vals = [0.0] * 6
                v_vals = [0.0] * 12
                w_vals = [0.0] * 18
                
                # A_top is 2x2: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
                # U indices for A_top: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
                u_vals[0] = a_coeff[0]  # a
                u_vals[1] = a_coeff[1]  # b
                u_vals[2] = a_coeff[2]  # c
                u_vals[3] = a_coeff[3]  # d
                
                # B_col_group is 2x2: e=B[0,2cg], f=B[0,2cg+1], g=B[1,2cg], h=B[1,2cg+1]
                # V indices: B[0,j] -> j, B[1,j] -> 6+j
                v_vals[b_col_offset] = b_coeff[0]       # e = B[0, 2cg]
                v_vals[b_col_offset + 1] = b_coeff[1]   # f = B[0, 2cg+1]
                v_vals[6 + b_col_offset] = b_coeff[2]   # g = B[1, 2cg]
                v_vals[6 + b_col_offset + 1] = b_coeff[3] # h = B[1, 2cg+1]
                
                # W: for each output position (i, j) in the 2x2 result
                # The 2x2 result goes to A_top@B_col_group
                # Output positions: (0, 2cg), (0, 2cg+1), (1, 2cg), (1, 2cg+1)
                # W index for (col j, row i): j*3 + i
                for out_idx in range(4):
                    out_i, out_j = divmod(out_idx, 2)
                    coeff = output_coeffs[out_idx][mu]
                    if coeff != 0:
                        w_col = 2 * col_group + out_j
                        w_row = out_i  # 0 or 1
                        w_vals[w_col * 3 + w_row] = coeff
                
                components.append((u_vals, v_vals, w_vals))
            
            # A_bot @ B_col_group: 1x2 @ 2x2, schoolbook rank 4
            # A_bot: A[2,0], A[2,1] -> U indices 4, 5
            for k in range(2):  # intermediate index
                for j in range(2):  # output col in the 2x2 block
                    u_vals = [0.0] * 6
                    v_vals = [0.0] * 12
                    w_vals = [0.0] * 18
                    
                    u_vals[4 + k] = 1.0  # A[2, k]
                    v_vals[k * 6 + b_col_offset + j] = 1.0  # B[k, 2cg+j]
                    w_vals[(2 * col_group + j) * 3 + 2] = 1.0  # W[j, row 2]
                    
                    components.append((u_vals, v_vals, w_vals))
        
        # Assemble into U, V, W matrices
        rank = len(components)
        U_mat = [[0.0] * rank for _ in range(6)]
        V_mat = [[0.0] * rank for _ in range(12)]
        W_mat = [[0.0] * rank for _ in range(18)]
        
        for c_idx, (u_vals, v_vals, w_vals) in enumerate(components):
            for r in range(6):
                U_mat[r][c_idx] = u_vals[r]
            for r in range(12):
                V_mat[r][c_idx] = v_vals[r]
            for r in range(18):
                W_mat[r][c_idx] = w_vals[r]
        
        return U_mat, V_mat, W_mat, rank
    
    # Build the full 3x4 * 4x6 decomposition
    # Split middle dim 4 into 2+2
    # A (3x4) = [A_left (3x2) | A_right (3x2)]
    # B (4x6) = [B_top (2x6); B_bot (2x6)]
    # C = A_left @ B_top + A_right @ B_bot
    
    # For each 3x2 @ 2x6, use the Strassen-based decomposition above
    U_left, V_top, W_left, rank1 = _build_strassen_3x2_2x6()
    U_right, V_bot, W_right, rank2 = _build_strassen_3x2_2x6()
    
    # Combine
    total_rank = rank1 + rank2
    U_full = [[0.0] * total_rank for _ in range(12)]  # 3*4
    V_full = [[0.0] * total_rank for _ in range(24)]  # 4*6
    W_full = [[0.0] * total_rank for _ in range(18)]  # 6*3
    
    # A_left is rows 0-2, cols 0-1 of A (3x4)
    # U_left has 6 rows (3*2), map to U_full rows 0-5 (col 0-1 of A)
    for r in range(6):
        i = r // 2
        j = r % 2
        for c in range(rank1):
            U_full[i * 4 + j][c] = U_left[r][c]
    
    # A_right is rows 0-2, cols 2-3 of A (3x4)
    # U_right has 6 rows, map to U_full rows 6-11 (col 2-3 of A)
    for r in range(6):
        i = r // 2
        j = r % 2
        for c in range(rank2):
            U_full[i * 4 + (j + 2)][rank1 + c] = U_right[r][c]
    
    # B_top is rows 0-1, cols 0-5 of B (2x6)
    # V_top has 12 rows (2*6), map to V_full rows 0-11
    for r in range(12):
        j = r // 6
        k = r % 6
        for c in range(rank1):
            V_full[j * 6 + k][c] = V_top[r][c]
    
    # B_bot is rows 2-3, cols 0-5 of B (2x6)
    # V_bot has 12 rows, map to V_full rows 12-23
    for r in range(12):
        j = r // 6
        k = r % 6
        for c in range(rank2):
            V_full[(j + 2) * 6 + k][rank1 + c] = V_bot[r][c]
    
    # W is the same for both parts
    for r in range(18):
        for c in range(rank1):
            W_full[r][c] = W_left[r][c]
        for c in range(rank2):
            W_full[r][rank1 + c] = W_right[r][c]
    
    # Verify
    if _verify(U_full, V_full, W_full, n, m, p, total_rank):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
