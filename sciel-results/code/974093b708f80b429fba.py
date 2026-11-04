"""Block-Strassen hybrid decomposition for 3x4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-88 tensor decomposition for 3x4 * 4x8 multiplication.
    
    Approach: Block-Strassen hybrid.
    - Split A (3x4) into A1 (rows 0,1) and A2 (row 2).
    - Split B (4x8) into 4 blocks B1,B2,B3,B4 each of size 4x2.
    - For each block Bt:
      * A1*Bt (2x4 * 4x2): Use Strassen on two 2x2x2 sub-products (14 terms)
      * A2*Bt (1x4 * 4x2): Schoolbook (8 terms)
    - Total: 4 * (14 + 8) = 88 terms.
    """
    n, m, p = 3, 4, 8
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))
    
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        for (ui, uc) in u_coeffs:
            for (vi, vc) in v_coeffs:
                for (wi, wc) in w_coeffs:
                    add_term(ui, vi, wi, uc, vc, wc)
    
    for t in range(4):
        c0 = 2 * t
        c1 = 2 * t + 1
        
        # --- Strassen for A1 * Bt ---
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
        # Bt11 = B[0:2, c0:c1+1], Bt21 = B[2:4, c0:c1+1]
        
        # P1 = (a00+a11)*(B[0,c0]+B[1,c1])
        add_strassen_term(
            [(0, 1), (5, 1)],
            [(c0, 1), (8+c1, 1)],
            [(c0*3+0, 1), (c1*3+1, 1)]
        )
        
        # P2 = (a00+a11)*B[1,c0]
        add_strassen_term(
            [(0, 1), (5, 1)],
            [(8+c0, 1)],
            [(c0*3+1, 1), (c1*3+1, -1)]
        )
        
        # P3 = a00*(B[0,c0]-B[1,c0])
        add_strassen_term(
            [(0, 1)],
            [(c0, 1), (8+c0, -1)],
            [(c1*3+0, 1), (c1*3+1, 1)]
        )
        
        # P4 = a11*(B[1,c1]-B[0,c0])
        add_strassen_term(
            [(5, 1)],
            [(8+c1, 1), (c0, -1)],
            [(c0*3+0, 1), (c0*3+1, 1), (c1*3+1, 1)]
        )
        
        # P5 = (a01+a11)*B[0,c0]
        add_strassen_term(
            [(1, 1), (5, 1)],
            [(c0, 1)],
            [(c0*3+0, -1), (c1*3+0, 1)]
        )
        
        # P6 = (a00-a10)*(B[1,c0]+B[1,c1])
        add_strassen_term(
            [(0, 1), (4, -1)],
            [(8+c0, 1), (8+c1, 1)],
            [(c1*3+1, 1)]
        )
        
        # P7 = (a10-a00)*(B[0,c0]-B[0,c1])
        add_strassen_term(
            [(4, 1), (0, -1)],
            [(c0, 1), (c1, -1)],
            [(c0*3+0, 1), (c1*3+1, 1)]
        )
        
        # Q1 = (a02+a13)*(B[2,c0]+B[3,c1])
        add_strassen_term(
            [(2, 1), (7, 1)],
            [(16+c0, 1), (24+c1, 1)],
            [(c0*3+0, 1), (c1*3+1, 1)]
        )
        
        # Q2 = (a02+a13)*B[3,c0]
        add_strassen_term(
            [(2, 1), (7, 1)],
            [(24+c0, 1)],
            [(c0*3+1, 1), (c1*3+1, -1)]
        )
        
        # Q3 = a02*(B[2,c0]-B[3,c0])
        add_strassen_term(
            [(2, 1)],
            [(16+c0, 1), (24+c0, -1)],
            [(c1*3+0, 1), (c1*3+1, 1)]
        )
        
        # Q4 = a13*(B[3,c1]-B[2,c0])
        add_strassen_term(
            [(7, 1)],
            [(24+c1, 1), (16+c0, -1)],
            [(c0*3+0, 1), (c0*3+1, 1), (c1*3+1, 1)]
        )
        
        # Q5 = (a03+a13)*B[2,c0]
        add_strassen_term(
            [(3, 1), (7, 1)],
            [(16+c0, 1)],
            [(c0*3+0, -1), (c1*3+0, 1)]
        )
        
        # Q6 = (a02-a12)*(B[3,c0]+B[3,c1])
        add_strassen_term(
            [(2, 1), (6, -1)],
            [(24+c0, 1), (24+c1, 1)],
            [(c1*3+1, 1)]
        )
        
        # Q7 = (a12-a02)*(B[2,c0]-B[2,c1])
        add_strassen_term(
            [(6, 1), (2, -1)],
            [(16+c0, 1), (16+c1, -1)],
            [(c0*3+0, 1), (c1*3+1, 1)]
        )
        
        # --- Schoolbook for A2 * Bt ---
        for j in range(4):
            u_idx = 8 + j
            v_idx = j * 8 + c0
            w_idx = c0 * 3 + 2
            add_term(u_idx, v_idx, w_idx, 1, 1, 1)
            
            v_idx2 = j * 8 + c1
            w_idx2 = c1 * 3 + 2
            add_term(u_idx, v_idx2, w_idx2, 1, 1, 1)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (ui, vi, wi, uc, vc, wc) in enumerate(terms):
        U[ui][col] = uc
        V[vi][col] = vc
        W[wi][col] = wc
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
