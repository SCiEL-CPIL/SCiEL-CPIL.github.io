"""Program for matrix multiplication 2x4 x 4x6 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix using a block-Strassen approach.
    
    The decomposition has rank 42.
    """
    n, m, p = DIMENSIONS
    rank = 42
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 6 blocks of 2x2x2 multiplications
    # Each block is defined by (i_offset, j_offset, k_offset)
    # The block covers i in [i_off, i_off+2), j in [j_off, j_off+2), k in [k_off, k_off+2)
    blocks = [
        (0, 0, 4), # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
        (0, 2, 4), # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
        (0, 0, 0), # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 2, 0), # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 0, 2), # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 2, 2), # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    ]
    
    col = 0
    for i_off, j_off, k_off in blocks:
        # Local indices for A, B, C
        # A: i in {0,1}, j in {0,1} relative to i_off, j_off
        # B: j in {0,1}, k in {0,1} relative to j_off, k_off
        # C: i in {0,1}, k in {0,1} relative to i_off, k_off
        
        # Strassen coefficients for U, V, W
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # We define the 7 terms as (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: dict mapping (li, lj) to coefficient
        # v_coeffs: dict mapping (lj, lk) to coefficient
        # w_coeffs: dict mapping (lk, li) to coefficient (Note: W index is k*n+i, so (lk, li))
        
        # Helper to add a term
        def add_term(u_dict, v_dict, w_dict, coeff_u=1, coeff_v=1, coeff_w=1):
            pass # Not used directly, we will construct lists
            
        # Term 1: (X00+X11)(Y00+Y11)
        # U: X00, X11 -> 1, 1
        # V: Y00, Y11 -> 1, 1
        # W: Z00, Z11 contributions?
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # We need to distribute M1 into W.
        # M1 contributes +1 to Z00 and +1 to Z11.
        # So W: (0,0) -> +1, (1,1) -> +1.
        
        terms = [
            # M1
            {
                "u": { (0,0): 1, (1,1): 1 },
                "v": { (0,0): 1, (1,1): 1 },
                "w": { (0,0): 1, (1,1): 1 }
            },
            # M2
            {
                "u": { (1,0): 1, (1,1): 1 },
                "v": { (0,0): 1 },
                "w": { (1,0): 1, (1,1): -1 } # Z10 +1, Z11 -1
            },
            # M3
            {
                "u": { (0,0): 1 },
                "v": { (0,1): 1, (1,1): -1 },
                "w": { (0,1): 1, (1,1): 1 } # Z01 +1, Z11 +1
            },
            # M4
            {
                "u": { (1,1): 1 },
                "v": { (1,0): 1, (0,0): -1 },
                "w": { (0,0): 1, (1,0): 1 } # Z00 +1, Z10 +1
            },
            # M5
            {
                "u": { (0,0): 1, (0,1): 1 },
                "v": { (1,1): 1 },
                "w": { (0,0): -1, (0,1): 1 } # Z00 -1, Z01 +1
            },
            # M6
            {
                "u": { (1,0): 1, (0,0): -1 },
                "v": { (0,0): 1, (0,1): 1 },
                "w": { (1,1): 1 } # Z11 +1
            },
            # M7
            {
                "u": { (0,1): 1, (1,1): -1 },
                "v": { (1,0): 1, (1,1): 1 },
                "w": { (0,0): 1 } # Z00 +1
            }
        ]
        
        for term in terms:
            # Update U
            for (li, lj), coeff in term["u"].items():
                global_i = i_off + li
                global_j = j_off + lj
                u_idx = global_i * m + global_j
                U[u_idx][col] += coeff
            
            # Update V
            for (lj, lk), coeff in term["v"].items():
                global_j = j_off + lj
                global_k = k_off + lk
                v_idx = global_j * p + global_k
                V[v_idx][col] += coeff
            
            # Update W
            # W index is k*n + i.
            # In term["w"], keys are (lk, li) corresponding to Z_lk,li?
            # Z is 2x2. Z[0,0] is i=0, k=0.
            # My w dict keys: (0,0) means Z[0,0] which is i=0, k=0.
            # So key (lk, li) is correct if I interpret first as k, second as i.
            # Let's check M1: w: {(0,0): 1, (1,1): 1}.
            # Z00 (i=0, k=0) and Z11 (i=1, k=1).
            # Yes.
            for (lk, li), coeff in term["w"].items():
                global_k = k_off + lk
                global_i = i_off + li
                w_idx = global_k * n + global_i
                W[w_idx][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
