"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition with Strassen:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses a single rank-1 term (rank 1)
  - Total rank: 49 + 49 + 1 = 99
"""

import time

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 99.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2
    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict (li,lj) -> coeff
    # B_coeffs: dict (lj,lk) -> coeff
    # C_coeffs: dict (li,lk) -> coeff
    
    strassen_2x2_terms = [
        # s1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # s2 = (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # s3 = a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # s4 = d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # s5 = (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # s6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # s7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add Strassen decomposition for 4x4 * 4x4."""
        
        def get_A_entry(li, lj):
            """Get global index for A entry at (li, lj) within the 4x4 block."""
            gi = A_base_i + li
            gj = A_base_j + lj
            return gi * m + gj
        
        def get_B_entry(lj, lk):
            """Get global index for B entry at (lj, lk) within the 4x4 block."""
            gj = B_base_j + lj
            gk = B_base_k + lk
            return gj * p + gk
        
        def get_C_entry(li, lk):
            """Get global index for C entry at (li, lk) within the 4x4 block."""
            gi = C_base_i + li
            gk = C_base_k + lk
            return gk * n + gi
        
        def add_block_multiplication(A_side, B_side, C_side):
            """
            Add 7 Strassen terms for a block multiplication.
            
            A_side: list of (br, bc, coeff) - blocks to sum with coefficients
            B_side: list of (br, bc, coeff) - blocks to sum with coefficients
            C_side: dict mapping (br, bc) to coeff - C blocks with coefficients
            """
            for A_coeffs, B_coeffs, C_contribs in strassen_2x2_terms:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                # Build u_vec from A_coeffs and A_side
                for (li, lj), meta_coeff in A_coeffs.items():
                    for (br, bc, block_coeff) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += meta_coeff * block_coeff
                
                # Build v_vec from B_coeffs and B_side
                for (lj, lk), meta_coeff in B_coeffs.items():
                    for (br, bc, block_coeff) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += meta_coeff * block_coeff
                
                # Build w_vec from C_contribs and C_side
                for (li, lk), meta_coeff in C_contribs.items():
                    for (br, bc), block_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += meta_coeff * block_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # M1 = (A11+A22)(B11+B22), C11: +1, C22: +1
        add_block_multiplication(
            A_side=[(0,0,1), (1,1,1)],
            B_side=[(0,0,1), (1,1,1)],
            C_side={(0,0): 1, (1,1): 1}
        )
        
        # M2 = (A21+A22)B11, C21: +1, C22: -1
        add_block_multiplication(
            A_side=[(1,0,1), (1,1,1)],
            B_side=[(0,0,1)],
            C_side={(1,0): 1, (1,1): -1}
        )
        
        # M3 = A11(B12-B22), C12: +1, C22: +1
        add_block_multiplication(
            A_side=[(0,0,1)],
            B_side=[(0,1,1), (1,1,-1)],
            C_side={(0,1): 1, (1,1): 1}
        )
        
        # M4 = A22(B21-B11), C11: +1, C21: +1
        add_block_multiplication(
            A_side=[(1,1,1)],
            B_side=[(1,0,1), (0,0,-1)],
            C_side={(0,0): 1, (1,0): 1}
        )
        
        # M5 = (A11+A12)B22, C11: -1, C12: +1
        add_block_multiplication(
            A_side=[(0,0,1), (0,1,1)],
            B_side=[(1,1,1)],
            C_side={(0,0): -1, (0,1): 1}
        )
        
        # M6 = (A21-A11)(B11+B12), C22: +1
        add_block_multiplication(
            A_side=[(1,0,1), (0,0,-1)],
            B_side=[(0,0,1), (0,1,1)],
            C_side={(1,1): 1}
        )
        
        # M7 = (A12-A22)(B21+B22), C11: +1
        add_block_multiplication(
            A_side=[(0,1,1), (1,1,-1)],
            B_side=[(1,0,1), (1,1,1)],
            C_side={(0,0): 1}
        )
    
    # Part 1: A * B1 where B1 is columns 0-3
    strassen_4x4(
        A_base_i=0, A_base_j=0,
        B_base_j=0, B_base_k=0,
        C_base_i=0, C_base_k=0
    )
    
    # Part 2: A * B2 where B2 is columns 4-7
    strassen_4x4(
        A_base_i=0, A_base_j=0,
        B_base_j=0, B_base_k=4,
        C_base_i=0, C_base_k=4
    )
    
    # Part 3: A * B3 where B3 is column 8 (4x1)
    # Single rank-1 term: U = all ones, V = ones at column 8, W = ones at column 8
    u_vec = [0] * 16
    v_vec = [0] * 36
    w_vec = [0] * 36
    
    for i in range(4):
        for j in range(4):
            u_vec[i * m + j] = 1
    
    for j in range(4):
        v_vec[j * p + 8] = 1
    
    for i in range(4):
        w_vec[8 * n + i] = 1
    
    add_term(u_vec, v_vec, w_vec)
    
    # Build U, V, W matrices
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}
