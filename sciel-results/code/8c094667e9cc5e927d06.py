import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We are constructing a decomposition of rank 55
    # Split p=10 into 5 blocks of size 2
    # Each block has rank 11 (4 from 2x1x2, 7 from 2x2x2 Strassen)
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff_u, coeff_v, coeff_w
    def add_term(u_idx, v_idx, w_idx, cu, cv, cw):
        U.append([cu] + [0] * (len(U) if False else 0)) # This approach is inefficient for building lists
        # Better to build arrays or lists of lists and append
    
    # Let's use lists of lists for U, V, W and append columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vals, v_vals, w_vals):
        # u_vals: dict or list of (idx, val)
        # We need to fill the whole vector
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for idx, val in u_vals.items():
            u_vec[idx] = val
        for idx, val in v_vals.items():
            v_vec[idx] = val
        for idx, val in w_vals.items():
            w_vec[idx] = val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Process 5 blocks of 2 columns each
    for b in range(5):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # Part 1: A[2x1] * B[1x2] using j=2
        # A indices: i=0, j=2 -> 0*3+2=2; i=1, j=2 -> 1*3+2=5
        # B indices: j=2, k=k1 -> 2*10+k1; j=2, k=k2 -> 2*10+k2
        # C indices: i=0, k=k1 -> k1*2+0; i=0, k=k2 -> k2*2+0; i=1, k=k1 -> k1*2+1; i=1, k=k2 -> k2*2+1
        
        # T00: A[0,2] B[2,k1] -> C[0,k1]
        add_rank1({2: 1.0}, {2*p + k1: 1.0}, {k1*n + 0: 1.0})
        # T01: A[0,2] B[2,k2] -> C[0,k2]
        add_rank1({2: 1.0}, {2*p + k2: 1.0}, {k2*n + 0: 1.0})
        # T10: A[1,2] B[2,k1] -> C[1,k1]
        add_rank1({5: 1.0}, {2*p + k1: 1.0}, {k1*n + 1: 1.0})
        # T11: A[1,2] B[2,k2] -> C[1,k2]
        add_rank1({5: 1.0}, {2*p + k2: 1.0}, {k2*n + 1: 1.0})
        
        # Part 2: A[2x2] * B[2x2] using j=0,1
        # A indices:
        # A[0,0] -> 0, A[0,1] -> 1
        # A[1,0] -> 3, A[1,1] -> 4
        # B indices:
        # B[0,k1] -> k1, B[0,k2] -> k2
        # B[1,k1] -> p+k1, B[1,k2] -> p+k2
        
        # Strassen Multiplications
        # M1 = (A[0,0] + A[1,1]) * (B[0,k1] + B[1,k2])
        # M2 = (A[1,0] + A[1,1]) * B[0,k1]
        # M3 = A[0,0] * (B[0,k2] - B[1,k2])
        # M4 = A[1,1] * (B[1,k1] - B[0,k1])
        # M5 = (A[0,0] + A[0,1]) * B[1,k2]
        # M6 = (A[1,0] - A[0,0]) * (B[0,k1] + B[0,k2])
        # M7 = (A[0,1] - A[1,1]) * (B[1,k1] + B[1,k2])
        
        # Contributions to C:
        # C[0,k1] = M1 + M4 - M5 + M7
        # C[0,k2] = M3 + M5
        # C[1,k1] = M2 + M4
        # C[1,k2] = M1 - M2 + M3 + M6
        
        # We need to expand each M into rank-1 terms and add to respective C entries.
        
        # Helper to add Strassen term M to C[i,k] with coefficient alpha
        # M is defined by (sum c_a A_a) * (sum d_b B_b)
        def add_strassen_term_to_c(alpha, i, k, a_coeffs, b_coeffs):
            # a_coeffs: list of (A_idx, coeff)
            # b_coeffs: list of (B_idx, coeff)
            for a_idx, ca in a_coeffs:
                for b_idx, cb in b_coeffs:
                    add_rank1({a_idx: alpha * ca}, {b_idx: cb}, {k*n + i: 1.0})
        
        # M1
        a1 = [(0, 1.0), (4, 1.0)] # A[0,0], A[1,1]
        b1 = [(k1, 1.0), (p+k2, 1.0)] # B[0,k1], B[1,k2]
        add_strassen_term_to_c(1.0, 0, k1, a1, b1)
        add_strassen_term_to_c(1.0, 1, k2, a1, b1)
        
        # M2
        a2 = [(3, 1.0), (4, 1.0)] # A[1,0], A[1,1]
        b2 = [(k1, 1.0)] # B[0,k1]
        add_strassen_term_to_c(1.0, 1, k1, a2, b2)
        add_strassen_term_to_c(-1.0, 1, k2, a2, b2)
        
        # M3
        a3 = [(0, 1.0)] # A[0,0]
        b3 = [(k2, 1.0), (p+k2, -1.0)] # B[0,k2], -B[1,k2]
        add_strassen_term_to_c(1.0, 0, k2, a3, b3)
        add_strassen_term_to_c(1.0, 1, k2, a3, b3)
        
        # M4
        a4 = [(4, 1.0)] # A[1,1]
        b4 = [(p+k1, 1.0), (k1, -1.0)] # B[1,k1], -B[0,k1]
        add_strassen_term_to_c(1.0, 0, k1, a4, b4)
        add_strassen_term_to_c(1.0, 1, k1, a4, b4)
        
        # M5
        a5 = [(0, 1.0), (1, 1.0)] # A[0,0], A[0,1]
        b5 = [(p+k2, 1.0)] # B[1,k2]
        add_strassen_term_to_c(-1.0, 0, k1, a5, b5)
        add_strassen_term_to_c(1.0, 0, k2, a5, b5)
        
        # M6
        a6 = [(3, 1.0), (0, -1.0)] # A[1,0], -A[0,0]
        b6 = [(k1, 1.0), (k2, 1.0)] # B[0,k1], B[0,k2]
        add_strassen_term_to_c(1.0, 1, k2, a6, b6)
        
        # M7
        a7 = [(1, 1.0), (4, -1.0)] # A[0,1], -A[1,1]
        b7 = [(p+k1, 1.0), (p+k2, 1.0)] # B[1,k1], B[1,k2]
        add_strassen_term_to_c(1.0, 0, k1, a7, b7)

    # Convert lists of columns to matrices (rows x rank)
    rank = len(U_cols)
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
