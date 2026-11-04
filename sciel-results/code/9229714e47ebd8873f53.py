"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    columns = []  # Each entry is (u_dict, v_dict, w_dict)
    
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
    
    def add_schoolblock(n_start, m_start, p_start, ni, mi, pi):
        """Add schoolbook terms for a sub-block."""
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    u_idx = (n_start + i) * m + (m_start + j)
                    v_idx = (m_start + j) * p + (p_start + k)
                    w_idx = (p_start + k) * n + (n_start + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def add_laderman_234(n_start, m_start, p_start):
        """Add Laderman's rank-14 decomposition for 2x3x4 block.
        
        A is 2x3 (rows i=0,1; cols j=0,1,2)
        B is 3x4 (rows j=0,1,2; cols k=0,1,2,3)
        C is 2x4 (rows i=0,1; cols k=0,1,2,3)
        
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Uses 14 multiplications via Laderman's method.
        """
        def u_glob(li, lj):
            return (n_start + li) * m + (m_start + lj)
        def v_glob(li, lj):
            return (m_start + li) * p + (p_start + lj)
        def w_glob(li, lj):
            return (p_start + lj) * n + (n_start + li)
        
        # Laderman 2x3x4 rank-14 decomposition:
        # We split B into B_top (rows 0,1) and B_bot (row 2)
        # C = A_top * B_top + A_bot * B_bot
        # where A_top = A[:,0:2], A_bot = A[:,2]
        # B_top is 2x4, B_bot is 1x4
        #
        # For A_top * B_top (2x2x4): use Laderman 2x2x4 (rank 7)
        # For A_bot * B_bot (2x1x4): schoolbook (rank 8)
        # But we can share: total = 7 + 7 = 14
        #
        # Actually, let me use a different valid 14-term construction.
        # 
        # Here's the key insight for 2x3x4 rank 14:
        # Split into two 2x2x4 problems by pairing rows of B:
        # Problem 1: A[:,0:2] * B[0:2,:] -> contributes to C
        # Problem 2: A[:,1:3] * B[1:3,:] -> contributes to C
        # But these overlap on A[:,1]*B[1,:]
        #
        # Let me use a concrete verified construction:
        # 
        # Terms 1-4: A[0,0]*B[0,k] for k=0,1,2,3
        # Terms 5-8: A[0,1]*B[1,k] for k=0,1,2,3  
        # Terms 9-12: A[0,2]*B[2,k] for k=0,1,2,3
        # That's 12 for row 0 of C.
        # For row 1 of C, we need another 12. Total 24.
        #
        # To get 14, we need sharing between rows.
        # 
        # Laderman's trick: For 2x2x2, use (a+b)(c+d) type sharing.
        # For 2x3x4, apply this per column pair.
        #
        # Here's a valid rank-14 construction:
        # For each pair of output columns (k, k+1) for k=0,2:
        #   Use 2x3x2 subproblem with rank 7 (if achievable)
        # 
        # Actually, the rank of 2x3x2 is 8, not 7.
        # So 2*(2x3x2) = 16, not 14.
        #
        # Let me just use a known correct 14-term decomposition.
        # After research, here's one valid construction:
        
        # M1 = A[0,0] * B[0,0]
        # M2 = A[0,0] * B[0,1]
        # M3 = A[0,0] * B[0,2]
        # M4 = A[0,0] * B[0,3]
        # M5 = A[0,1] * B[1,0]
        # M6 = A[0,1] * B[1,1]
        # M7 = A[0,1] * B[1,2]
        # M8 = A[0,1] * B[1,3]
        # M9 = A[0,2] * B[2,0]
        # M10 = A[0,2] * B[2,1]
        # M11 = A[0,2] * B[2,2]
        # M12 = A[0,2] * B[2,3]
        # M13 = (A[1,0]+A[1,1]+A[1,2]) * (B[0,0]+B[1,0]+B[2,0])
        # M14 = (A[1,0]+A[1,1]+A[1,2]) * (B[0,1]+B[1,1]+B[2,1])
        #
        # But M13 expands to 9 terms, and we only want the diagonal ones.
        # This doesn't work.
        #
        # Let me try yet another approach. I'll use the fact that:
        # For 2x3x4, we can write it as:
        # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:]
        # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:]
        #
        # Group by B row:
        # = (A[0,0]*B[0,:] + A[1,0]*B[0,:]) + (A[0,1]*B[1,:] + A[1,1]*B[1,:]) + (A[0,2]*B[2,:] + A[1,2]*B[2,:])
        # = B[0,:] * [A[0,0], A[1,0]] + B[1,:] * [A[0,1], A[1,1]] + B[2,:] * [A[0,2], A[1,2]]
        #
        # Each term B[j,:] * [A[0,j], A[1,j]] is a 1x4 times 2x1 = 2x4, needing 8 mults.
        # Total: 3*8 = 24. No improvement.
        #
        # I give up on finding rank 14 analytically right now.
        # Let me use schoolbook (24 terms) for correctness, and note that
        # the total will be 96.
        
        # Schoolbook for 2x3x4:
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    add_term(
                        {u_glob(i, j): 1},
                        {v_glob(j, k): 1},
                        {w_glob(i, k): 1}
                    )
    
    # Block 1: n=[0], m=[0,1,2], p=[0,1,2,3]
    add_schoolblock(0, 0, 0, 1, 3, 4)  # 12 terms
    
    # Block 2: n=[0], m=[3], p=[0,1,2,3]
    add_schoolblock(0, 3, 0, 1, 1, 4)  # 4 terms
    
    # Block 3: n=[0], m=[0,1,2], p=[4,5,6,7]
    add_schoolblock(0, 0, 4, 1, 3, 4)  # 12 terms
    
    # Block 4: n=[0], m=[3], p=[4,5,6,7]
    add_schoolblock(0, 3, 4, 1, 1, 4)  # 4 terms
    
    # Block 5: n=[1,2], m=[0,1,2], p=[0,1,2,3]
    add_laderman_234(1, 0, 0)  # 24 terms (schoolbook fallback)
    
    # Block 6: n=[1,2], m=[3], p=[0,1,2,3]
    add_schoolblock(1, 3, 0, 2, 1, 4)  # 8 terms
    
    # Block 7: n=[1,2], m=[0,1,2], p=[4,5,6,7]
    add_laderman_234(1, 0, 4)  # 24 terms (schoolbook fallback)
    
    # Block 8: n=[1,2], m=[3], p=[4,5,6,7]
    add_schoolblock(1, 3, 4, 2, 1, 4)  # 8 terms
    
    rank = len(columns)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(columns):
        for idx, val in u_dict.items():
            U[idx][col_idx] = val
        for idx, val in v_dict.items():
            V[idx][col_idx] = val
        for idx, val in w_dict.items():
            W[idx][col_idx] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
