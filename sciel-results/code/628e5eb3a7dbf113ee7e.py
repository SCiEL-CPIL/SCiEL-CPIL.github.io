"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition by block partitioning:
    # n=3 -> [1, 2]
    # m=4 -> [3, 1]  
    # p=8 -> [4, 4]
    
    # Sub-blocks and their ranks:
    # (1,3,4): rank 12
    # (1,1,4): rank 4
    # (2,3,4): rank 14 (Laderman)
    # (2,1,4): rank 8
    # Total: 2*(12+4+14+8) = 76
    
    U = []
    V = []
    W = []
    
    def add_block(u_entries, v_entries, w_entries):
        """Add a block to the decomposition.
        u_entries: list of (index_in_U, value) for this block's U columns
        v_entries: list of (index_in_V, value) for this block's V columns  
        w_entries: list of (index_in_W, value) for this block's W columns
        """
        pass
    
    # Let me build it differently. I'll construct the full U, V, W matrices.
    
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    # Let me build column by column.
    
    columns = []  # Each entry is (U_col, V_col, W_col) as sparse dicts
    
    # Helper to add a rank-1 term
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
    
    # Block 1: (1,3,4) at n=[0], m=[0,1,2], p=[0,1,2,3]
    # Schoolbook: for i=0, j=0,1,2, k=0,1,2,3
    # U index: i*m + j = 0*4 + j = j
    # V index: j*p + k = j*8 + k
    # W index: k*n + i = k*3 + 0 = k*3
    for j in range(3):
        for k in range(4):
            u = {j: 1}
            v = {j * 8 + k: 1}
            w = {k * 3 + 0: 1}
            add_term(u, v, w)
    
    # Block 2: (1,1,4) at n=[0], m=[3], p=[0,1,2,3]
    # U index: 0*4 + 3 = 3
    # V index: 3*8 + k = 24 + k
    # W index: k*3 + 0 = k*3
    for k in range(4):
        u = {3: 1}
        v = {24 + k: 1}
        w = {k * 3 + 0: 1}
        add_term(u, v, w)
    
    # Block 3: (1,3,4) at n=[0], m=[0,1,2], p=[4,5,6,7]
    for j in range(3):
        for k in range(4):
            pk = k + 4  # p index 4,5,6,7
            u = {j: 1}
            v = {j * 8 + pk: 1}
            w = {pk * 3 + 0: 1}
            add_term(u, v, w)
    
    # Block 4: (1,1,4) at n=[0], m=[3], p=[4,5,6,7]
    for k in range(4):
        pk = k + 4
        u = {3: 1}
        v = {24 + pk: 1}
        w = {pk * 3 + 0: 1}
        add_term(u, v, w)
    
    # Block 5: (2,3,4) at n=[1,2], m=[0,1,2], p=[0,1,2,3] - Laderman rank 14
    # This is the tricky part. I need to implement Laderman's 2×3×4.
    # Let me add 14 terms here.
    # I'll implement a known rank-14 decomposition.
    
    # For now, let me use schoolbook (24 terms) and replace with Laderman later.
    # Actually, let me just implement Laderman directly.
    
    # Laderman's 2x3x4 algorithm:
    # A is 2x3, B is 3x4, C is 2x4
    # I'll use a verified rank-14 construction.
    
    # Here's a construction using 14 bilinear forms:
    # I'll define them as (u_coefficients, v_coefficients, w_coefficients)
    # where u is over A's 6 elements, v over B's 12 elements, w over C's 8 elements.
    
    # Let me use a specific known decomposition.
    # A[i,j] for i=0,1 j=0,1,2 -> indices 0-5
    # B[i,j] for i=0,1,2 j=0,1,2,3 -> indices 0-11
    # C[i,j] for i=0,1 j=0,1,2,3 -> indices 0-7
    
    # Local to global mapping:
    # A local (i,j) -> U global index: (1+i)*4 + j = 4+i*4+j
    # B local (i,j) -> V global index: j*8 + (0+i) = j*8+i  (since m starts at 0)
    #   Wait, m=[0,1,2] for this block, so B row index is 0,1,2
    #   V global: m_idx * p + p_idx = m_idx * 8 + p_idx
    #   m_idx is 0,1,2 and p_idx is 0,1,2,3
    # C local (i,j) -> W global index: p_idx * n + n_idx
    #   n_idx is 1,2 (local i=0,1 maps to global n=1,2)
    #   p_idx is 0,1,2,3
    #   W global: p_idx * 3 + (1+i)
    
    laderman_terms = []
    
    # I'll use the following rank-14 decomposition for 2x3x4:
    # This is based on Laderman's original algorithm.
    
    # Let me define 14 terms. Each term is (u_dict, v_dict, w_dict) in local coords.
    # u_dict: {local_A_index: coeff}
    # v_dict: {local_B_index: coeff}
    # w_dict: {local_C_index: coeff}
    
    # Local A: 2x3, index = i*3+j
    # Local B: 3x4, index = i*4+j
    # Local C: 2x4, index = i*4+j
    
    # Term 1: A[0,0]*B[0,0] -> C[0,0]
    laderman_terms.append(({0: 1}, {0: 1}, {0: 1}))
    # Term 2: A[0,0]*B[0,1] -> C[0,1]
    laderman_terms.append(({0: 1}, {1: 1}, {1: 1}))
    # Term 3: A[0,1]*B[1,0] -> C[0,2]
    laderman_terms.append(({1: 1}, {4: 1}, {2: 1}))
    # Term 4: A[0,1]*B[1,1] -> C[0,3]
    laderman_terms.append(({1: 1}, {5: 1}, {3: 1}))
    # Term 5: A[0,2]*B[2,0] -> C[0,4]... wait C only has 8 elements (2x4)
    # C local index: i*4+j, so max is 7.
    # Let me redo. C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3, C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
    
    # I need a proper rank-14 construction. Let me use a different approach.
    # I'll just use schoolbook for 2x3x4 (rank 24) for now to get a working solution,
    # then optimize.
    
    # Actually, let me just implement schoolbook for all blocks first to verify correctness,
    # then replace with optimized versions.
    
    # Clear and restart with a cleaner approach
    columns = []
    
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
    
    # For each sub-block, add schoolbook terms (will optimize 2x3x4 later)
    
    # Block (n_range, m_range, p_range) with sizes (ni, mi, pi)
    def add_schoolblock(n_start, m_start, p_start, ni, mi, pi):
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    u_idx = (n_start + i) * m + (m_start + j)
                    v_idx = (m_start + j) * p + (p_start + k)
                    w_idx = (p_start + k) * n + (n_start + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Block 1: n=[0], m=[0,1,2], p=[0,1,2,3]
    add_schoolblock(0, 0, 0, 1, 3, 4)  # 12 terms
    
    # Block 2: n=[0], m=[3], p=[0,1,2,3]
    add_schoolblock(0, 3, 0, 1, 1, 4)  # 4 terms
    
    # Block 3: n=[0], m=[0,1,2], p=[4,5,6,7]
    add_schoolblock(0, 0, 4, 1, 3, 4)  # 12 terms
    
    # Block 4: n=[0], m=[3], p=[4,5,6,7]
    add_schoolblock(0, 3, 4, 1, 1, 4)  # 4 terms
    
    # Block 5: n=[1,2], m=[0,1,2], p=[0,1,2,3] - use Laderman (14 terms)
    # Block 6: n=[1,2], m=[3], p=[0,1,2,3] - schoolbook (8 terms)
    # Block 7: n=[1,2], m=[0,1,2], p=[4,5,6,7] - use Laderman (14 terms)
    # Block 8: n=[1,2], m=[3], p=[4,5,6,7] - schoolbook (8 terms)
    
    # For blocks 5 and 7, I'll add Laderman terms
    # For blocks 6 and 8, schoolbook
    
    # Add Laderman for block 5
    def add_laderman(n_start, m_start, p_start):
        """Add a rank-22 decomposition for 2x3x4 block (2x2x4 + 2x1x4)."""
        def u_glob(li, lj):
            return (n_start + li) * m + (m_start + lj)
        def v_glob(li, lj):
            return (m_start + li) * p + (p_start + lj)
        def w_glob(li, lj):
            return (p_start + lj) * n + (n_start + li)
            
        # Part 1: 2x2x4 (m=0,1)
        # Use Laderman 2x2x2 for cols 0,1 and cols 2,3
        # Laderman 2x2x2 terms for A(2x2) B(2x2) C(2x2)
        # A indices: 0,1,2,3. B indices: 0,1,2,3. C indices: 0,1,2,3.
        
        # Sub-block 1: cols 0,1
        # M1 = A00*B00
        add_term({u_glob(0,0):1}, {v_glob(0,0):1}, {w_glob(0,0):1})
        add_term({u_glob(0,0):1}, {v_glob(0,1):1}, {w_glob(0,1):1})
        add_term({u_glob(0,1):1}, {v_glob(1,0):1}, {w_glob(0,0):1})
        add_term({u_glob(0,1):1}, {v_glob(1,1):1}, {w_glob(0,1):1})
        add_term({u_glob(1,0):1}, {v_glob(0,0):1}, {w_glob(1,0):1})
        add_term({u_glob(1,0):1}, {v_glob(0,1):1}, {w_glob(1,1):1})
        add_term({u_glob(1,1):1}, {v_glob(1,0):1}, {w_glob(1,0):1})
        add_term({u_glob(1,1):1}, {v_glob(1,1):1}, {w_glob(1,1):1})
        
        # Sub-block 2: cols 2,3
        add_term({u_glob(0,0):1}, {v_glob(0,2):1}, {w_glob(0,2):1})
        add_term({u_glob(0,0):1}, {v_glob(0,3):1}, {w_glob(0,3):1})
        add_term({u_glob(0,1):1}, {v_glob(1,2):1}, {w_glob(0,2):1})
        add_term({u_glob(0,1):1}, {v_glob(1,3):1}, {w_glob(0,3):1})
        add_term({u_glob(1,0):1}, {v_glob(0,2):1}, {w_glob(1,2):1})
        add_term({u_glob(1,0):1}, {v_glob(0,3):1}, {w_glob(1,3):1})
        add_term({u_glob(1,1):1}, {v_glob(1,2):1}, {w_glob(1,2):1})
        add_term({u_glob(1,1):1}, {v_glob(1,3):1}, {w_glob(1,3):1})
        
        # Part 2: 2x1x4 (m=2)
        for i in range(2):
            for k in range(4):
                add_term({u_glob(i,2):1}, {v_glob(2,k):1}, {w_glob(i,k):1})
    
    add_laderman(1, 0, 0)  # Block 5 - using rank 22
    
    # Block 6: n=[1,2], m=[3], p=[0,1,2,3]
    add_schoolblock(1, 3, 0, 2, 1, 4)  # 8 terms
    
    # Block 7: n=[1,2], m=[0,1,2], p=[4,5,6,7]
    add_laderman(1, 0, 4)  # Block 7 - using rank 22
    
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
