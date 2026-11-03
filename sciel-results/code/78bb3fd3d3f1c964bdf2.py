"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by summing contributions from sub-blocks.
    # Total rank will be the sum of ranks of sub-blocks.
    
    # Helper to add a term to the global decomposition
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We use a list of columns for U, V, W to make appending easy.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: list of length n*m
        v_vec: list of length m*p
        w_vec: list of length p*n
        """
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def get_zero_u():
        return [0] * (n * m)
    def get_zero_v():
        return [0] * (m * p)
    def get_zero_w():
        return [0] * (p * n)

    # --- Strassen 2x2 Decomposition ---
    # Maps local indices (0,1) to global indices.
    # A local (i,j), B local (j,k), C local (i,k)
    def strassen_2x2(a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_k_offset, c_i_offset):
        # a_i_offset, a_j_offset: start index of 2x2 block in A
        # b_j_offset, b_k_offset: start index of 2x2 block in B
        # c_k_offset, c_i_offset: start index of 2x2 block in C
        
        # M1 = (A00 + A11) * (B00 + B11)
        # w1 = C00 + C11
        u = get_zero_u(); v = get_zero_v(); w = get_zero_w()
        u[a_i_offset + 0, a_j_offset + 0] = 1 # A00
        u[a_i_offset + 1, a_j_offset + 1] = 1 # A11
        v[b_j_offset + 0, b_k_offset + 0] = 1 # B00
        v[b_j_offset + 1, b_k_offset + 1] = 1 # B11
        w[c_k_offset + 0, c_i_offset + 0] = 1 # C00
        w[c_k_offset + 1, c_i_offset + 1] = 1 # C11
        add_term(u, v, w)

        # M2 = (A10 + A11) * B00
        # w2 = C10 - C11 (Wait, C21 = M2 + M4, C22 = M1 - M2... so w2 = C21 - C22? No.)
        # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
        # Coeff of M2 in C21 is 1. Coeff in C22 is -1.
        u = get_zero_u(); v = get_zero_v(); w = get_zero_w()
        u[a_i_offset + 1, a_j_offset + 0] = 1 # A10
        u[a_i_offset + 1, a_j_offset + 1] = 1 # A11
        v[b_j_offset + 0, b_k_offset + 0] = 1 # B00
        w[c_k_offset + 0, c_i_offset + 1] = 1 # C10 (local index 1,0)
        w[c_k_offset + 1, c_i_offset + 1] = -1 # C11 (local index 1,1)
        add_term(u, v, w)

        # M3 = A00 * (B10 - B11)
        # w3 = C01 + C11
        u = get_zero_u(); v = get_zero_v(); w = get_zero_w()
        u[a_i_offset + 0, a_j_offset + 0] = 1 # A00
        v[b_j_offset + 1, b_k_offset + 0] = 1 # B10
        v[b_j_offset + 1, b_k_offset + 1] = -1 # B11
        w[c_k_offset + 0, c_i_offset + 0] = 1 # C00? No.
        # C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # Coeff M3 in C01 is 1. Coeff M3 in C11 is 1.
        w[c_k_offset + 0, c_i_offset + 0] = 0 # C00
        w[c_k_offset + 0, c_i_offset + 1] = 1 # C01 (local 0,1)
        w[c_k_offset + 1, c_i_offset + 1] = 1 # C11 (local 1,1)
        add_term(u, v, w)

        # M4 = A11 * (B01 - B00)
        # w4 = C00 + C10
        u = get_zero_u(); v = get_zero_v(); w = get_zero_w()
        u[a_i_offset + 1, a_j_offset + 1] = 1 # A11
        v[b_j_offset + 0, b_k_offset + 1] = 1 # B01
        v[b_j_offset + 0, b_k_offset + 0] = -1 # B00
        w[c_k_offset + 0, c_i_offset + 0] = 1 # C00
        w[c_k_offset + 0, c_i_offset + 1] = 1 # C10 (local 1,0)
        add_term(u, v, w)

        # M5 = (A00 + A01) * B11
        # w5 = -C00 + C01
        u = get_zero_u(); v = get_zero_v(); w = get_zero_w()
        u[a_i_offset + 0, a_j_offset + 0] = 1 # A00
        u[a_i_offset + 0, a_j_offset + 1] = 1 # A01
        v[b_j_offset + 1, b_k_offset + 1] = 1 # B11
        w[c_k_offset + 0, c_i_offset + 0] = -1 # C00
        w[c_k_offset + 0, c_i_offset + 1] = 1 # C01 (local 0,1)
        add_term(u, v, w)

        # M6 = (A10 - A00) * (B00 + B01)
        # w6 = C11
        u = get_zero_u(); v = get_zero_v(); w = get_zero_w()
        u[a_i_offset + 1, a_j_offset + 0] = 1 # A10
        u[a_i_offset + 0, a_j_offset + 0] = -1 # A00
        v[b_j_offset + 0, b_k_offset + 0] = 1 # B00
        v[b_j_offset + 0, b_k_offset + 1] = 1 # B01
        w[c_k_offset + 1, c_i_offset + 1] = 1 # C11 (local 1,1)
        add_term(u, v, w)

        # M7 = (A01 - A11) * (B10 + B11)
        # w7 = C00 - C10
        u = get_zero_u(); v = get_zero_v(); w = get_zero_w()
        u[a_i_offset + 0, a_j_offset + 1] = 1 # A01
        u[a_i_offset + 1, a_j_offset + 1] = -1 # A11
        v[b_j_offset + 1, b_k_offset + 0] = 1 # B10
        v[b_j_offset + 1, b_k_offset + 1] = 1 # B11
        w[c_k_offset + 0, c_i_offset + 0] = 1 # C00
        w[c_k_offset + 0, c_i_offset + 1] = -1 # C10 (local 1,0)
        add_term(u, v, w)

    # --- Trivial Decomposition ---
    # For n x m x p, rank n*m*p.
    def trivial_mm(n_sub, m_sub, p_sub, a_i_off, a_j_off, b_j_off, b_k_off, c_k_off, c_i_off):
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u = get_zero_u(); v = get_zero_v(); w = get_zero_w()
                    u[a_i_off + i, a_j_off + j] = 1
                    v[b_j_off + j, b_k_off + k] = 1
                    w[c_k_off + k, c_i_off + i] = 1
                    add_term(u, v, w)

    # --- Block 1: A_L (3x2) * B_T (2x6) ---
    # A_L is A[0:3, 0:2]. B_T is B[0:2, 0:6].
    # Split B_T into B_T1 (2x3) and B_T2 (2x3).
    
    # Sub-block 1.1: A_L (3x2) * B_T1 (2x3)
    # A_L indices: i=0..2, j=0..1.
    # B_T1 indices: j=0..1, k=0..2.
    # C indices: k=0..2, i=0..2.
    
    # Decompose 3x2x3 using R<=17 logic.
    # Split A_L into A_L1 (2x2, i=0..1) and A_L2 (1x2, i=2).
    # Split B_T1 into B_T1a (2x2, k=0..1) and B_T1b (2x1, k=2).
    
    # Term 1.1.1: A_L2 (1x2) * B_T1a (2x2). Rank 4.
    # A_L2 is row 2 of A_L. B_T1a is cols 0,1 of B_T1.
    trivial_mm(1, 2, 2, 2, 0, 0, 0, 0, 2) # c_i_off=2? No, c_i is global i.
    # Wait, trivial_mm expects local offsets.
    # a_i_off=2, a_j_off=0. n_sub=1.
    # b_j_off=0, b_k_off=0. m_sub=2.
    # c_k_off=0, c_i_off=2. p_sub=2, n_sub=1.
    # This maps to global:
    # U: i=2, j=0,1.
    # V: j=0,1, k=0,1.
    # W: k=0,1, i=2.
    # Correct.

    # Term 1.1.2: A_L1 (2x2) * B_T1a (2x2). Rank 7.
    # A_L1 is rows 0,1 of A_L. B_T1a is cols 0,1 of B_T1.
    strassen_2x2(0, 0, 0, 0, 0, 0)
    # a_i_off=0, a_j_off=0.
    # b_j_off=0, b_k_off=0.
    # c_k_off=0, c_i_off=0.
    # Correct.

    # Term 1.1.3: A_L (3x2) * B_T1b (2x1). Rank 6.
    # A_L is rows 0,1,2. B_T1b is col 2 of B_T1.
    trivial_mm(3, 2, 1, 0, 0, 0, 2, 2, 0)
    # a_i_off=0, a_j_off=0. n_sub=3, m_sub=2.
    # b_j_off=0, b_k_off=2. m_sub=2, p_sub=1.
    # c_k_off=2, c_i_off=0. p_sub=1, n_sub=3.
    # Correct.

    # Sub-block 1.2: A_L (3x2) * B_T2 (2x3)
    # B_T2 is B[0:2, 3:6].
    # Same structure as 1.1, but k offset is 3.
    
    # Term 1.2.1: A_L2 (1x2) * B_T2a (2x2, k=3,4).
    trivial_mm(1, 2, 2, 2, 0, 0, 3, 3, 2)
    
    # Term 1.2.2: A_L1 (2x2) * B_T2a (2x2, k=3,4).
    strassen_2x2(0, 0, 0, 3, 3, 0)
    
    # Term 1.2.3: A_L (3x2) * B_T2b (2x1, k=5).
    trivial_mm(3, 2, 1, 0, 0, 0, 5, 5, 0)

    # --- Block 2: A_R (3x3) * B_B (3x6) ---
    # A_R is A[0:3, 2:5]. B_B is B[2:5, 0:6].
    # Split B_B into B_B1 (3x3, k=0..2) and B_B2 (3x3, k=3..5).
    
    # Sub-block 2.1: A_R (3x3) * B_B1 (3x3)
    # Use R<=26 decomposition.
    # Split A_R into 2x2, 2x1, 1x2, 1x1.
    # Split B_B1 into 2x2, 2x1, 1x2, 1x1.
    
    # A_R indices: i=0..2, j=2..4.
    # B_B1 indices: j=2..4, k=0..2.
    # C indices: k=0..2, i=0..2.
    
    # 1. A_R22 (2x2) * B_B122 (2x2). Rank 7.
    # A_R22: i=0,1; j=2,3.
    # B_B122: j=2,3; k=0,1.
    strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # 2. A_R21 (2x1) * B_B112 (1x2). Rank 4.
    # A_R21: i=0,1; j=4.
    # B_B112: j=4; k=0,1.
    trivial_mm(2, 1, 2, 0, 4, 4, 0, 0, 0)
    
    # 3. A_R22 (2x2) * B_B121 (2x1). Rank 4.
    # A_R22: i=0,1; j=2,3.
    # B_B121: j=2,3; k=2.
    trivial_mm(2, 2, 1, 0, 2, 2, 2, 2, 0)
    
    # 4. A_R21 (2x1) * B_B111 (1x1). Rank 2.
    # A_R21: i=0,1; j=4.
    # B_B111: j=4; k=2.
    trivial_mm(2, 1, 1, 0, 4, 4, 2, 2, 0)
    
    # 5. A_R12 (1x2) * B_B122 (2x2). Rank 4.
    # A_R12: i=2; j=2,3.
    # B_B122: j=2,3; k=0,1.
    trivial_mm(1, 2, 2, 2, 2, 2, 0, 0, 2)
    
    # 6. A_R11 (1x1) * B_B112 (1x2). Rank 2.
    # A_R11: i=2; j=4.
    # B_B112: j=4; k=0,1.
    trivial_mm(1, 1, 2, 2, 4, 4, 0, 0, 2)
    
    # 7. A_R12 (1x2) * B_B121 (2x1). Rank 2.
    # A_R12: i=2; j=2,3.
    # B_B121: j=2,3; k=2.
    trivial_mm(1, 2, 1, 2, 2, 2, 2, 2, 2)
    
    # 8. A_R11 (1x1) * B_B111 (1x1). Rank 1.
    # A_R11: i=2; j=4.
    # B_B111: j=4; k=2.
    trivial_mm(1, 1, 1, 2, 4, 4, 2, 2, 2)

    # Sub-block 2.2: A_R (3x3) * B_B2 (3x3)
    # Same as 2.1 but k offset is 3.
    
    # 1. A_R22 (2x2) * B_B222 (2x2, k=3,4).
    strassen_2x2(0, 2, 2, 3, 3, 0)
    
    # 2. A_R21 (2x1) * B_B212 (1x2, k=3,4).
    trivial_mm(2, 1, 2, 0, 4, 4, 3, 3, 0)
    
    # 3. A_R22 (2x2) * B_B221 (2x1, k=5).
    trivial_mm(2, 2, 1, 0, 2, 2, 5, 5, 0)
    
    # 4. A_R21 (2x1) * B_B211 (1x1, k=5).
    trivial_mm(2, 1, 1, 0, 4, 4, 5, 5, 0)
    
    # 5. A_R12 (1x2) * B_B222 (2x2, k=3,4).
    trivial_mm(1, 2, 2, 2, 2, 2, 3, 3, 2)
    
    # 6. A_R11 (1x1) * B_B212 (1x2, k=3,4).
    trivial_mm(1, 1, 2, 2, 4, 4, 3, 3, 2)
    
    # 7. A_R12 (1x2) * B_B221 (2x1, k=5).
    trivial_mm(1, 2, 1, 2, 2, 2, 5, 5, 2)
    
    # 8. A_R11 (1x1) * B_B211 (1x1, k=5).
    trivial_mm(1, 1, 1, 2, 4, 4, 5, 5, 2)

    # Convert lists of columns to matrices
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
