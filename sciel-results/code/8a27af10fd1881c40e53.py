"""
Program to construct a rank-minimal exact tensor decomposition for matrix multiplication.
Target: 3x4 matrix by 4x4 matrix.
Dimensions: n=3, m=4, p=4.
Schoolbook rank: 3*4*4 = 48.
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for matrix multiplication C = A * B.
    
    Shapes:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta (reconstructs C_ik)
    """
    n, m, p = DIMENSIONS
    
    # We will build the decomposition term by term.
    # Each term is a dictionary mapping (row_index, col_index) to value for U, V, W.
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    
    terms = []
    
    # Strassen's algorithm for 2x2x2
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # 7 products M1..M7
    # We define the coefficients for U, V, W for each of the 7 products.
    # U corresponds to A, V to B, W to C.
    
    # Local indices for 2x2: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    strassen_terms = []
    
    # M1 = (a00 + a11) * (b00 + b11)
    # U: a00=1, a11=1. V: b00=1, b11=1.
    # W: c00 += 1, c11 += 1
    u1 = {0: 1, 3: 1} # a00, a11
    v1 = {0: 1, 3: 1} # b00, b11
    w1 = {0: 1, 3: 1} # c00, c11
    strassen_terms.append((u1, v1, w1))
    
    # M2 = (a10 + a11) * b00
    # U: a10=1, a11=1. V: b00=1.
    # W: c10 += 1, c11 -= 1
    u2 = {2: 1, 3: 1} # a10, a11
    v2 = {0: 1}       # b00
    w2 = {2: 1, 3: -1} # c10, c11
    strassen_terms.append((u2, v2, w2))
    
    # M3 = a00 * (b10 - b11)
    # U: a00=1. V: b10=1, b11=-1.
    # W: c01 += 1, c11 += 1
    u3 = {0: 1}       # a00
    v3 = {2: 1, 3: -1} # b10, b11
    w3 = {1: 1, 3: 1}  # c01, c11
    strassen_terms.append((u3, v3, w3))
    
    # M4 = a11 * (b01 - b00)
    # U: a11=1. V: b01=1, b00=-1.
    # W: c00 += 1, c10 += 1
    u4 = {3: 1}       # a11
    v4 = {1: 1, 0: -1} # b01, b00
    w4 = {0: 1, 2: 1}  # c00, c10
    strassen_terms.append((u4, v4, w4))
    
    # M5 = (a00 + a01) * b11
    # U: a00=1, a01=1. V: b11=1.
    # W: c00 -= 1, c01 += 1
    u5 = {0: 1, 1: 1} # a00, a01
    v5 = {3: 1}       # b11
    w5 = {0: -1, 1: 1} # c00, c01
    strassen_terms.append((u5, v5, w5))
    
    # M6 = (a10 - a00) * (b00 + b01)
    # U: a10=1, a00=-1. V: b00=1, b01=1.
    # W: c11 += 1
    u6 = {2: 1, 0: -1} # a10, a00
    v6 = {0: 1, 1: 1}  # b00, b01
    w6 = {3: 1}         # c11
    strassen_terms.append((u6, v6, w6))
    
    # M7 = (a01 - a11) * (b10 + b11)
    # U: a01=1, a11=-1. V: b10=1, b11=1.
    # W: c00 += 1
    u7 = {1: 1, 3: -1} # a01, a11
    v7 = {2: 1, 3: 1}  # b10, b11
    w7 = {0: 1}         # c00
    strassen_terms.append((u7, v7, w7))
    
    # Schoolbook terms for 1x2x2
    # Inputs: A (1x2), B (2x2)
    # Outputs: C (1x2)
    # 4 products
    # U index: i*2 + j. i=0. j in {0,1}.
    # V index: j*2 + k. j in {0,1}, k in {0,1}.
    # W index: k*2 + i. i=0. k in {0,1}.
    
    schoolbook_1x2x2_terms = []
    
    # C0 = A0 B00 + A1 B10
    # Term 1: A0 B00 -> C0
    u_s1 = {0: 1} # A0
    v_s1 = {0: 1} # B00
    w_s1 = {0: 1} # C0
    schoolbook_1x2x2_terms.append((u_s1, v_s1, w_s1))
    
    # Term 2: A1 B10 -> C0
    u_s2 = {1: 1} # A1
    v_s2 = {2: 1} # B10
    w_s2 = {0: 1} # C0
    schoolbook_1x2x2_terms.append((u_s2, v_s2, w_s2))
    
    # C1 = A0 B01 + A1 B11
    # Term 3: A0 B01 -> C1
    u_s3 = {0: 1} # A0
    v_s3 = {1: 1} # B01
    w_s3 = {1: 1} # C1
    schoolbook_1x2x2_terms.append((u_s3, v_s3, w_s3))
    
    # Term 4: A1 B11 -> C1
    u_s4 = {1: 1} # A1
    v_s4 = {3: 1} # B11
    w_s4 = {1: 1} # C1
    schoolbook_1x2x2_terms.append((u_s4, v_s4, w_s4))
    
    # Helper to add a block of terms with offsets
    def add_block(block_terms, i_offset, j_offset, k_offset, n_block, m_block, p_block):
        """
        Adds terms for a block of size n_block x m_block x p_block.
        i_offset, j_offset, k_offset are the starting indices in the global dimensions.
        """
        for u_local, v_local, w_local in block_terms:
            u_global = {}
            v_global = {}
            w_global = {}
            
            for u_idx, val in u_local.items():
                # u_idx is i*m_block + j
                i_loc, j_loc = divmod(u_idx, m_block)
                i_glob = i_offset + i_loc
                j_glob = j_offset + j_loc
                u_global[i_glob * m + j_glob] = val
            
            for v_idx, val in v_local.items():
                # v_idx is j*p_block + k
                j_loc, k_loc = divmod(v_idx, p_block)
                j_glob = j_offset + j_loc
                k_glob = k_offset + k_loc
                v_global[j_glob * p + k_glob] = val
            
            for w_idx, val in w_local.items():
                # w_idx is k*n_block + i
                k_loc, i_loc = divmod(w_idx, n_block)
                k_glob = k_offset + k_loc
                i_glob = i_offset + i_loc
                w_global[k_glob * n + i_glob] = val
            
            terms.append((u_global, v_global, w_global))
            
    # Part 1: Rows 0,1 of A (2x4x4)
    # Sub-problem 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_block(strassen_terms, i_offset=0, j_offset=0, k_offset=0, n_block=2, m_block=2, p_block=2)
    
    # Sub-problem 2: i in {0,1}, j in {2,3}, k in {0,1}
    add_block(strassen_terms, i_offset=0, j_offset=2, k_offset=0, n_block=2, m_block=2, p_block=2)
    
    # Sub-problem 3: i in {0,1}, j in {0,1}, k in {2,3}
    add_block(strassen_terms, i_offset=0, j_offset=0, k_offset=2, n_block=2, m_block=2, p_block=2)
    
    # Sub-problem 4: i in {0,1}, j in {2,3}, k in {2,3}
    add_block(strassen_terms, i_offset=0, j_offset=2, k_offset=2, n_block=2, m_block=2, p_block=2)
    
    # Part 2: Row 2 of A (1x4x4)
    # Sub-problem 5: i in {2}, j in {0,1}, k in {0,1}
    add_block(schoolbook_1x2x2_terms, i_offset=2, j_offset=0, k_offset=0, n_block=1, m_block=2, p_block=2)
    
    # Sub-problem 6: i in {2}, j in {2,3}, k in {0,1}
    add_block(schoolbook_1x2x2_terms, i_offset=2, j_offset=2, k_offset=0, n_block=1, m_block=2, p_block=2)
    
    # Sub-problem 7: i in {2}, j in {0,1}, k in {2,3}
    add_block(schoolbook_1x2x2_terms, i_offset=2, j_offset=0, k_offset=2, n_block=1, m_block=2, p_block=2)
    
    # Sub-problem 8: i in {2}, j in {2,3}, k in {2,3}
    add_block(schoolbook_1x2x2_terms, i_offset=2, j_offset=2, k_offset=2, n_block=1, m_block=2, p_block=2)
    
    rank = len(terms)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for u_idx, val in u_dict.items():
            U[u_idx][r] = val
        for v_idx, val in v_dict.items():
            V[v_idx][r] = val
        for w_idx, val in w_dict.items():
            W[w_idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
