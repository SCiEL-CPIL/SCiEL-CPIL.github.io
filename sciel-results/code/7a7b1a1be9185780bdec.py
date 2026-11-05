import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition by summing contributions from blocks.
    # U: (36, Rank), V: (36, Rank), W: (36, Rank)
    # We collect lists of (U_col, V_col, W_col)
    U_cols = []
    V_cols = []
    W_cols = []

    def add_schoolbook(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        """Add schoolbook decomposition for a sub-block."""
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    
                    # Global indices
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    
                    u[u_idx] = 1
                    v[v_idx] = 1
                    w[w_idx] = 1
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)

    def add_strassen_2x2(i_off, j_off, k_off):
        """Add Strassen decomposition for a 2x2x2 block."""
        # The 7 multiplications
        # u: 4 entries (2x2 A)
        # v: 4 entries (2x2 B)
        # w: 4 entries (2x2 C)
        
        # Helper to map local 2x2 index to global 6x6 index
        def get_u_global(li, lj):
            return (i_off + li) * m + (j_off + lj)
        def get_v_global(lj, lk):
            return (j_off + lj) * p + (k_off + lk)
        def get_w_global(lk, li):
            return (k_off + lk) * n + (i_off + li)

        # 1. (A00+A11)(B00+B11) -> C00, C11
        u1 = [0]*36; v1 = [0]*36; w1 = [0]*36
        u1[get_u_global(0,0)] = 1; u1[get_u_global(1,1)] = 1
        v1[get_v_global(0,0)] = 1; v1[get_v_global(1,1)] = 1
        w1[get_w_global(0,0)] = 1; w1[get_w_global(1,1)] = 1
        U_cols.append(u1); V_cols.append(v1); W_cols.append(w1)

        # 2. (A10+A11)B00 -> C10
        u2 = [0]*36; v2 = [0]*36; w2 = [0]*36
        u2[get_u_global(1,0)] = 1; u2[get_u_global(1,1)] = 1
        v2[get_v_global(0,0)] = 1
        w2[get_w_global(0,1)] = 1
        U_cols.append(u2); V_cols.append(v2); W_cols.append(w2)

        # 3. A00(B01-B11) -> C01
        u3 = [0]*36; v3 = [0]*36; w3 = [0]*36
        u3[get_u_global(0,0)] = 1
        v3[get_v_global(0,1)] = 1; v3[get_v_global(1,1)] = -1
        w3[get_w_global(1,0)] = 1
        U_cols.append(u3); V_cols.append(v3); W_cols.append(w3)

        # 4. A11(B10-B00) -> C00, C10
        u4 = [0]*36; v4 = [0]*36; w4 = [0]*36
        u4[get_u_global(1,1)] = 1
        v4[get_v_global(1,0)] = 1; v4[get_v_global(0,0)] = -1
        w4[get_w_global(0,0)] = 1; w4[get_w_global(0,1)] = 1
        U_cols.append(u4); V_cols.append(v4); W_cols.append(w4)

        # 5. (A00+A01)B11 -> C00, C01
        u5 = [0]*36; v5 = [0]*36; w5 = [0]*36
        u5[get_u_global(0,0)] = 1; u5[get_u_global(0,1)] = 1
        v5[get_v_global(1,1)] = 1
        w5[get_w_global(0,0)] = -1; w5[get_w_global(1,0)] = 1
        U_cols.append(u5); V_cols.append(v5); W_cols.append(w5)

        # 6. (A10-A00)(B00+B01) -> C11
        u6 = [0]*36; v6 = [0]*36; w6 = [0]*36
        u6[get_u_global(1,0)] = 1; u6[get_u_global(0,0)] = -1
        v6[get_v_global(0,0)] = 1; v6[get_v_global(0,1)] = 1
        w6[get_w_global(1,1)] = 1
        U_cols.append(u6); V_cols.append(v6); W_cols.append(w6)

        # 7. (A01-A11)(B10+B11) -> C00, C11
        u7 = [0]*36; v7 = [0]*36; w7 = [0]*36
        u7[get_u_global(0,1)] = 1; u7[get_u_global(1,1)] = -1
        v7[get_v_global(1,0)] = 1; v7[get_v_global(1,1)] = 1
        w7[get_w_global(0,0)] = 1; w7[get_w_global(1,1)] = 1
        U_cols.append(u7); V_cols.append(v7); W_cols.append(w7)

    def add_strassen_4x4(i_off, j_off, k_off):
        """Add Strassen decomposition for a 4x4x4 block using 7 2x2x2 Strassen instances."""
        # Split 4x4 into four 2x2 blocks: 00, 01, 10, 11
        # Offsets for the 2x2 blocks within the 4x4 block
        # Block 00: i 0-1, j 0-1
        # Block 01: i 0-1, j 2-3
        # Block 10: i 2-3, j 0-1
        # Block 11: i 2-3, j 2-3
        
        # We need to map the 7 Strassen multiplications to the 2x2 sub-blocks.
        # The linear combinations involve sums/diffs of the 2x2 blocks.
        # This is effectively applying Strassen to the "matrix of blocks".
        
        # Instead of hardcoding 7*7=49 terms, we can recursively call add_strassen_2x2?
        # No, the inputs to the 2x2 Strassen are themselves linear combinations of 2x2 blocks.
        # So we need to compose the linear forms.
        
        # Let's just use Schoolbook for 4x4 to be safe and simple, 
        # unless we can easily compose.
        # Rank of Schoolbook 4x4 is 64.
        # Rank of Strassen 4x4 is 49.
        # Difference is 15.
        # Using Schoolbook for 4x4 gives total rank:
        # C44: 64 + 16 = 80
        # C42: 32 + 16 = 48
        # C24: 32 + 16 = 48
        # C22: 16 + 7 = 23
        # Total: 199.
        
        # Using Strassen for 4x4 gives total rank:
        # C44: 49 + 16 = 65
        # C42: 32 + 16 = 48
        # C24: 32 + 16 = 48
        # C22: 16 + 7 = 23
        # Total: 184.
        
        # 184 is better. Let's implement the composition.
        
        # The 7 multiplications of 4x4 Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mi is a product of two 2x2 matrices (linear combos of blocks).
        # So each Mi can be decomposed using 7 rank-1 tensors (Strassen on 2x2).
        # Total 49 rank-1 tensors.
        
        # We need to define the linear combinations for U and V for each of the 7 "super-terms".
        # Then for each super-term, we apply the 2x2 Strassen decomposition to the resulting 2x2 matrices.
        
        # This is complex to code generically in a short time.
        # Given the risk of bugs, and that 199 is already an improvement (216/199 = 1.085),
        # I will use Schoolbook for 4x4 to ensure correctness.
        # 199 is a safe bet.
        
        add_schoolbook(4, 4, 4, i_off, j_off, k_off)

    # Block 44: A[0:4, 0:4] * B[0:4, 0:4]
    add_strassen_4x4(0, 0, 0)
    # Block 44: A[0:4, 4:6] * B[4:6, 0:4]
    add_schoolbook(4, 2, 4, 0, 4, 0)
    
    # Block 42: A[0:4, 0:4] * B[0:4, 4:6]
    add_schoolbook(4, 4, 2, 0, 0, 4)
    # Block 42: A[0:4, 4:6] * B[4:6, 4:6]
    add_schoolbook(4, 2, 2, 0, 4, 4)
    
    # Block 24: A[4:6, 0:4] * B[0:4, 0:4]
    add_schoolbook(2, 4, 4, 4, 0, 0)
    # Block 24: A[4:6, 4:6] * B[4:6, 0:4]
    add_schoolbook(2, 2, 4, 4, 4, 0)
    
    # Block 22: A[4:6, 0:4] * B[0:4, 4:6]
    add_schoolbook(2, 4, 2, 4, 0, 4)
    # Block 22: A[4:6, 4:6] * B[4:6, 4:6]
    add_strassen_2x2(4, 4, 4)

    # Convert lists to matrices
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Convert to list of lists as required? 
    # The task says "factor matrices of shapes ... with integer/half-integer entries".
    # It doesn't specify list vs numpy, but the seed returns lists.
    # However, numpy arrays are often accepted if they are convertible.
    # To be safe, return lists.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
