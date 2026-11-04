"""Program for the suite task; Strassen-style recursive tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Build a rank-r decomposition of the (n x m) @ (m x p) tensor.

    Strategy: Decompose the 2x4x6 multiplication into 6 independent 2x2x2
    multiplications, each solved using Strassen's algorithm (rank 7).
    Total rank = 6 * 7 = 42.
    """
    n, m, p = DIMENSIONS
    
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We will build a list of terms first, then assemble.
    # Each term is a dict: {'U': {idx: val}, 'V': {idx: val}, 'W': {idx: val}}
    terms = []
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict {(i,j): val}
        # B_coeffs: dict {(j,k): val}
        # C_coeffs: dict {(k,i): val}
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}), # M2 -> C10
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}), # M3 -> C01
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # M4 -> C00, C10
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # M5 -> C00, C01
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # M6 -> C11
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M7 -> C00, C11
    ]
    
    # Loop over the 6 blocks
    # j_block: 0, 1 (splits m=4 into two 2s)
    # k_block: 0, 1, 2 (splits p=6 into three 2s)
    for jb in range(2):
        for kb in range(3):
            # Offset for A (cols), B (rows), B (cols), C (cols)
            a_j_off = jb * 2
            b_j_off = jb * 2
            b_k_off = kb * 2
            c_k_off = kb * 2
            
            for a_coefs, b_coefs, c_coefs in strassen_terms:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                # Map A indices
                for (i, j), val in a_coefs.items():
                    global_i = i
                    global_j = a_j_off + j
                    u_idx = global_i * m + global_j
                    u_dict[u_idx] = val
                
                # Map B indices
                for (j, k), val in b_coefs.items():
                    global_j = b_j_off + j
                    global_k = b_k_off + k
                    v_idx = global_j * p + global_k
                    v_dict[v_idx] = val
                
                # Map C indices
                # Note: C index in W is k*n + i
                for (k, i), val in c_coefs.items():
                    global_k = c_k_off + k
                    global_i = i
                    w_idx = global_k * n + global_i
                    w_dict[w_idx] = val
                
                terms.append((u_dict, v_dict, w_dict))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
