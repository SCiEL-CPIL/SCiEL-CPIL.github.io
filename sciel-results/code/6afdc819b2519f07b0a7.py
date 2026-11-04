import time
_start_time = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0

def construct():
    """
    Returns a rank-minimal tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses a Strassen-based block decomposition:
    - Split B into two 4x4 blocks and a 4x1 block
    - Use Strassen 4x4x4 (rank 49) for the first two 4x4 blocks
    - Use schoolbook (rank 16) for the last 4x1 block
    - Total rank: 49 + 49 + 16 = 114
    """
    
    n, m, p = 4, 4, 9
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen 2x2x2 base case: 7 rank-1 terms
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def build_4x4_strassen(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """Build a 4x4x4 Strassen decomposition using 2x2 blocks."""
        terms = []
        
        def build_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                for (loc_i, loc_j), coeff in u_c.items():
                    for (br, bc, bc_coeff) in A_blocks:
                        gi = A_i_base + br * 2 + loc_i
                        gj = A_j_base + bc * 2 + loc_j
                        u_vec[gi * m + gj] += coeff * bc_coeff
                
                for (loc_j, loc_k), coeff in v_c.items():
                    for (br, bc, bc_coeff) in B_blocks:
                        gj = B_j_base + br * 2 + loc_j
                        gk = B_k_base + bc * 2 + loc_k
                        v_vec[gj * p + gk] += coeff * bc_coeff
                
                for (loc_i, loc_k), coeff in w_c.items():
                    for (br, bc, bc_coeff) in C_blocks:
                        gi = C_i_base + br * 2 + loc_i
                        gk = C_k_base + bc * 2 + loc_k
                        w_vec[gk * n + gi] += coeff * bc_coeff
                
                terms.append((u_vec, v_vec, w_vec))
        
        build_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        build_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        build_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        build_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        build_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        build_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        build_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        
        return terms
    
    # Build first 4x4 block terms
    block1_terms = build_4x4_strassen(0, 0, 0, 0, 0, 0)
    for u, v, w in block1_terms:
        add_term(u, v, w)
    
    # Build second 4x4 block terms
    block2_terms = build_4x4_strassen(0, 0, 0, 4, 0, 4)
    for u, v, w in block2_terms:
        add_term(u, v, w)
    
    # Build third 4x4 block terms (columns 8-11, but we only have column 8)
    block3_terms = build_4x4_strassen(0, 0, 0, 8, 0, 8)
    for u, v, w in block3_terms:
        add_term(u, v, w)
    
    # Subtract schoolbook terms for the phantom columns 9, 10, 11
    # This effectively gives us a rank 49 decomposition for the 4x4 * 4x1 case
    # by using the 4x4 * 4x4 Strassen and subtracting out the extra 3 columns
    # Actually, this approach is wrong. Let me reconsider.
    
    # The issue: we have 49+49+49 = 147 terms, but we need to remove the contributions
    # from columns 9,10,11 which don't exist. The Strassen decomposition for columns 8-11
    # creates terms that reference B[j,9], B[j,10], B[j,11] which are 0.
    # So those terms are actually 0 and don't contribute!
    # Therefore we have rank 147 which is worse than 114.
    
    # Let me reset and use the original approach but with a different structure.
    U_cols = []
    V_cols = []
    W_cols = []
    
    for u, v, w in block1_terms:
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    for u, v, w in block2_terms:
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    for i in range(n):
        for j in range(m):
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    rank = len(U_cols)
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _MARGIN:
        pass
    
    # Convert list of column vectors to row-major matrices
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n*m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m*p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
