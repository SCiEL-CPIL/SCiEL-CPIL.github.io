"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We build the decomposition by splitting m into two 2s and p into three 2s and one 1.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Strassen 2x2x2 decomposition
    # m1 = (a+d)(e+h) -> C11 += 1, C22 += 1
    # m2 = (c+d)e    -> C21 += 1, C22 += -1
    # m3 = a(f-h)    -> C12 += 1, C22 += 1
    # m4 = d(g-e)    -> C11 += 1, C21 += 1
    # m5 = (a+b)h    -> C11 += -1, C12 += 1
    # m6 = (c-a)(e+f) -> C22 += 1
    # m7 = (b-d)(g+h) -> C11 += 1, C22 += -1
    
    # Local indices for A (2x2): (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # Local indices for B (2x2): (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # Local indices for C (2x2): (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    # Note: W index is k*n + i. C is n x p. C[i,k].
    # In local 2x2 C:
    # C11 is i=0, k=0. W idx = 0*2 + 0 = 0.
    # C12 is i=0, k=1. W idx = 1*2 + 0 = 2.
    # C21 is i=1, k=0. W idx = 0*2 + 1 = 1.
    # C22 is i=1, k=1. W idx = 1*2 + 1 = 3.
    
    strassen_comps = [
        # (U_dict, V_dict, W_dict)
        # U_dict: {(i,j): coeff}
        # V_dict: {(j,k): coeff}
        # W_dict: {(i,k): coeff}
        # m1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # m6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # m7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
    ]
    
    # Schoolbook 2x2x1 decomposition
    # A (2x2), B (2x1), C (2x1)
    # B indices: (0,0)=e, (1,0)=g.
    # C indices: (0,0)=C1, (1,0)=C2.
    # W idx for C1 (i=0,k=0) = 0.
    # W idx for C2 (i=1,k=0) = 1.
    
    schoolbook_comps = [
        # m1 = a e -> C1
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        # m2 = b g -> C1
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        # m3 = c e -> C2
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),
        # m4 = d g -> C2
        ({(1,1): 1}, {(1,0): 1}, {(1,0): 1}),
    ]
    
    # Iterate over blocks
    # j_block: 0, 1 (cols of A, rows of B)
    # k_block: 0, 1, 2, 3 (cols of B)
    for j_block in range(2):
        for k_block in range(4):
            # Determine block size for k
            if k_block < 3:
                # 2x2x2 block
                # Global J = j_block * 2 + j_loc
                # Global K = k_block * 2 + k_loc
                for u_dict, v_dict, w_dict in strassen_comps:
                    U_col = [0] * (n * m)
                    V_col = [0] * (m * p)
                    W_col = [0] * (p * n)
                    
                    for (i_loc, j_loc), coeff in u_dict.items():
                        i_glob = i_loc
                        j_glob = j_block * 2 + j_loc
                        idx_u = i_glob * m + j_glob
                        U_col[idx_u] = coeff
                    
                    for (j_loc, k_loc), coeff in v_dict.items():
                        j_glob = j_block * 2 + j_loc
                        k_glob = k_block * 2 + k_loc
                        idx_v = j_glob * p + k_glob
                        V_col[idx_v] = coeff
                    
                    for (i_loc, k_loc), coeff in w_dict.items():
                        i_glob = i_loc
                        k_glob = k_block * 2 + k_loc
                        idx_w = k_glob * n + i_glob
                        W_col[idx_w] = coeff
                        
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
            else:
                # 2x2x1 block
                # Global J = j_block * 2 + j_loc
                # Global K = 6
                for u_dict, v_dict, w_dict in schoolbook_comps:
                    U_col = [0] * (n * m)
                    V_col = [0] * (m * p)
                    W_col = [0] * (p * n)
                    
                    for (i_loc, j_loc), coeff in u_dict.items():
                        i_glob = i_loc
                        j_glob = j_block * 2 + j_loc
                        idx_u = i_glob * m + j_glob
                        U_col[idx_u] = coeff
                    
                    for (j_loc, k_loc), coeff in v_dict.items():
                        j_glob = j_block * 2 + j_loc
                        k_glob = 6
                        idx_v = j_glob * p + k_glob
                        V_col[idx_v] = coeff
                        
                    for (i_loc, k_loc), coeff in w_dict.items():
                        i_glob = i_loc
                        k_glob = 6
                        idx_w = k_glob * n + i_glob
                        W_col[idx_w] = coeff
                        
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)
                    
    # Convert lists of columns to matrices (lists of rows)
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for i in range(m * p):
            V[i][r] = V_cols[r][i]
        for i in range(p * n):
            W[i][r] = W_cols[r][i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
