"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Build decomposition using block Strassen + schoolbook
    # A (2x4) = [A1 (2x2), A2 (2x2)]
    # B (4x5) = [B1 (2x5); B2 (2x5)]
    # C = A1*B1 + A2*B2
    # Each B split into (2x2, 2x2, 2x1)
    
    # Pre-compute the Strassen and schoolbook templates for 2x2x2 and 2x2x1
    # Strassen terms for 2x2x2 (7 terms)
    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    # U index: i*2+j, V index: j*2+k, W index: k*2+i
    
    strassen_terms = [
        # Term 0: M1 = (A00+A11)(B00+B11), C00+1, C11+1
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # Term 1: M2 = (A10+A11)B00, C10+1, C11-1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        # Term 2: M3 = A00(B10-B11), C01+1, C11+1
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
        # Term 3: M4 = A11(B01-B00), C00+1, C10+1
        ([0,0,0,1], [-1,1,0,0], [1,0,1,0]),
        # Term 4: M5 = (A00+A01)B11, C00-1, C01+1
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # Term 5: M6 = (A10-A00)(B00+B10), C11+1
        ([ -1,0,1,0], [1,0,1,0], [0,0,0,1]),
        # Term 6: M7 = (A01-A11)(B10+B11), C00+1
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    schoolbook_221_terms = [
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        ([1,0,0,0], [1,0], [1,0]),  # A00*B00 -> C00
        ([0,1,0,0], [0,1], [1,0]),  # A01*B10 -> C00
        ([0,0,1,0], [1,0], [0,1]),  # A10*B00 -> C10
        ([0,0,0,1], [0,1], [0,1]),  # A11*B10 -> C10
    ]
    
    # U shape: (8, rank), V shape: (20, rank), W shape: (10, rank)
    # We'll build column by column
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    def add_term(U_coeffs, V_coeffs, W_coeffs, a_row_offset, a_col_offset, b_row_offset, b_col_offset, w_k_offset, w_i_offset):
        """Add one rank-1 term to the decomposition."""
        col_U = [0.0] * (n*m)
        col_V = [0.0] * (m*p)
        col_W = [0.0] * (p*n)
        
        for idx, coeff in enumerate(U_coeffs):
            if coeff != 0:
                i_local = idx // 2
                j_local = idx % 2
                u_row = (a_row_offset + i_local) * m + (a_col_offset + j_local)
                col_U[u_row] = coeff
        
        for idx, coeff in enumerate(V_coeffs):
            if coeff != 0:
                j_local = idx // 2
                k_local = idx % 2
                v_row = (b_row_offset + j_local) * p + (b_col_offset + k_local)
                col_V[v_row] = coeff
        
        for idx, coeff in enumerate(W_coeffs):
            if coeff != 0:
                k_local = idx // 2
                i_local = idx % 2
                w_row = (w_k_offset + k_local) * n + (w_i_offset + i_local)
                col_W[w_row] = coeff
        
        cols_U.append(col_U)
        cols_V.append(col_V)
        cols_W.append(col_W)
    
    def add_term_221(U_coeffs, V_coeffs, W_coeffs, a_row_offset, a_col_offset, b_row_offset, b_col_offset, w_k_offset, w_i_offset):
        """Add one rank-1 term for 2x2x1."""
        col_U = [0.0] * (n*m)
        col_V = [0.0] * (m*p)
        col_W = [0.0] * (p*n)
        
        for idx, coeff in enumerate(U_coeffs):
            if coeff != 0:
                i_local = idx // 2
                j_local = idx % 2
                u_row = (a_row_offset + i_local) * m + (a_col_offset + j_local)
                col_U[u_row] = coeff
        
        for idx, coeff in enumerate(V_coeffs):
            if coeff != 0:
                j_local = idx // 2  # 0 or 1
                k_local = 0         # only one column
                v_row = (b_row_offset + j_local) * p + (b_col_offset + k_local)
                col_V[v_row] = coeff
        
        for idx, coeff in enumerate(W_coeffs):
            if coeff != 0:
                i_local = idx  # 0 or 1, only one k
                w_row = w_k_offset * n + (w_i_offset + i_local)
                col_W[w_row] = coeff
        
        cols_U.append(col_U)
        cols_V.append(col_V)
        cols_W.append(col_W)
    
    # Block 1: A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-4)
    # A1: a_row_offset=0, a_col_offset=0
    # B1: b_row_offset=0, b_col_offset varies
    # W: k from 0-4, i from 0-1
    
    # B1_a: k=0,1
    for U_c, V_c, W_c in strassen_terms:
        add_term(U_c, V_c, W_c, 0, 0, 0, 0, 0, 0)
    
    # B1_b: k=2,3
    for U_c, V_c, W_c in strassen_terms:
        add_term(U_c, V_c, W_c, 0, 0, 0, 2, 2, 0)
    
    # B1_c: k=4
    for U_c, V_c, W_c in schoolbook_221_terms:
        add_term_221(U_c, V_c, W_c, 0, 0, 0, 4, 4, 0)
    
    # Block 2: A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 0-4)
    # A2: a_row_offset=0, a_col_offset=2
    # B2: b_row_offset=2, b_col_offset varies
    
    # B2_a: k=0,1
    for U_c, V_c, W_c in strassen_terms:
        add_term(U_c, V_c, W_c, 0, 2, 2, 0, 0, 0)
    
    # B2_b: k=2,3
    for U_c, V_c, W_c in strassen_terms:
        add_term(U_c, V_c, W_c, 0, 2, 2, 2, 2, 0)
    
    # B2_c: k=4
    for U_c, V_c, W_c in schoolbook_221_terms:
        add_term_221(U_c, V_c, W_c, 0, 2, 2, 4, 4, 0)
    
    rank = len(cols_U)
    
    U = list(zip(*cols_U))
    V = list(zip(*cols_V))
    W = list(zip(*cols_W))
    
    # Convert to list of lists of integers (all values are 0, 1, or -1)
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
