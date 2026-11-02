import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # We will build the decomposition as a list of terms.
    # Each term is a tuple (u_idx, v_idx, w_idx, coeff)
    # But we need to handle linear combinations (Strassen).
    # So a term is a dict: { 'U': {idx: coeff}, 'V': {idx: coeff}, 'W': {idx: coeff} }
    # Or simpler: just accumulate into U, V, W matrices directly.
    
    # Rank estimate: 66
    # Let's allocate U, V, W with rank 66.
    # If we find a smaller rank, we can trim.
    # But 66 is the target.
    
    rank = 66
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise OverflowError("Rank exceeded")
        for idx, coeff in u_coeffs.items():
            U[idx, col] += coeff
        for idx, coeff in v_coeffs.items():
            V[idx, col] += coeff
        for idx, coeff in w_coeffs.items():
            W[idx, col] += coeff
        col += 1
        
    def strassen_2x2x2(a_offset, b_offset, c_offset):
        """
        Adds Strassen terms for a 2x2x2 block.
        a_offset: (i0, j0) top-left of A block (2x2)
        b_offset: (j0, k0) top-left of B block (2x2)
        c_offset: (i0, k0) top-left of C block (2x2)
        """
        i0, j0 = a_offset
        j0_b, k0 = b_offset
        i0_c, k0_c = c_offset
        
        # M1 = (A11+A22)(B11+B22)
        a1 = {(i0, j0): 1, (i0+1, j0+1): 1}
        b1 = {(j0, k0): 1, (j0+1, k0+1): 1}
        c1 = {(i0, k0): 1, (i0+1, k0+1): 1}
        
        # M2 = (A21+A22)B11
        a2 = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        b2 = {(j0, k0): 1}
        c2 = {(i0+1, k0): 1, (i0+1, k0+1): -1}
        
        # M3 = A11(B12-B22)
        a3 = {(i0, j0): 1}
        b3 = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        c3 = {(i0, k0+1): 1, (i0+1, k0+1): 1}
        
        # M4 = A22(B21-B11)
        a4 = {(i0+1, j0+1): 1}
        b4 = {(j0+1, k0): 1, (j0, k0): -1}
        c4 = {(i0, k0): 1, (i0+1, k0): 1}
        
        # M5 = (A11+A12)B22
        a5 = {(i0, j0): 1, (i0, j0+1): 1}
        b5 = {(j0+1, k0+1): 1}
        c5 = {(i0, k0): -1, (i0, k0+1): 1}
        
        # M6 = (A21-A11)(B11+B12)
        a6 = {(i0+1, j0): 1, (i0, j0): -1}
        b6 = {(j0, k0): 1, (j0, k0+1): 1}
        c6 = {(i0+1, k0+1): 1}
        
        # M7 = (A12-A22)(B21+B22)
        a7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        b7 = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        c7 = {(i0, k0): 1}
        
        terms_list = [
            (a1, b1, c1),
            (a2, b2, c2),
            (a3, b3, c3),
            (a4, b4, c4),
            (a5, b5, c5),
            (a6, b6, c6),
            (a7, b7, c7)
        ]
        
        for a, b, c in terms_list:
            # Convert local (i,j) to global U index i*m+j
            u_coeffs = {i * m + j: coeff for (i, j), coeff in a.items()}
            # Convert local (j,k) to global V index j*p+k
            v_coeffs = {j * p + k: coeff for (j, k), coeff in b.items()}
            # Convert local (i,k) to global W index k*n+i
            w_coeffs = {k * n + i: coeff for (i, k), coeff in c.items()}
            
            add_term(u_coeffs, v_coeffs, w_coeffs)

    def schoolbook_1x2x2(a_offset, b_offset, c_offset):
        """
        Adds schoolbook terms for a 1x2x2 block.
        a_offset: (i0, j0) top-left of A block (1x2)
        b_offset: (j0, k0) top-left of B block (2x2)
        c_offset: (i0, k0) top-left of C block (1x2)
        """
        i0, j0 = a_offset
        j0_b, k0 = b_offset
        i0_c, k0_c = c_offset
        
        # A is 1x2: A1 at (i0, j0), A2 at (i0, j0+1)
        # B is 2x2: B11 at (j0, k0), B12 at (j0, k0+1), B21 at (j0+1, k0), B22 at (j0+1, k0+1)
        # C is 1x2: C1 at (i0, k0), C2 at (i0, k0+1)
        
        # C1 = A1*B11 + A2*B21
        # C2 = A1*B12 + A2*B22
        
        # Term 1: A1*B11 -> C1
        a1 = {(i0, j0): 1}
        b1 = {(j0, k0): 1}
        c1 = {(i0, k0): 1}
        
        # Term 2: A2*B21 -> C1
        a2 = {(i0, j0+1): 1}
        b2 = {(j0+1, k0): 1}
        c2 = {(i0, k0): 1}
        
        # Term 3: A1*B12 -> C2
        a3 = {(i0, j0): 1}
        b3 = {(j0, k0+1): 1}
        c3 = {(i0, k0+1): 1}
        
        # Term 4: A2*B22 -> C2
        a4 = {(i0, j0+1): 1}
        b4 = {(j0+1, k0+1): 1}
        c4 = {(i0, k0+1): 1}
        
        terms_list = [
            (a1, b1, c1),
            (a2, b2, c2),
            (a3, b3, c3),
            (a4, b4, c4)
        ]
        
        for a, b, c in terms_list:
            u_coeffs = {i * m + j: coeff for (i, j), coeff in a.items()}
            v_coeffs = {j * p + k: coeff for (j, k), coeff in b.items()}
            w_coeffs = {k * n + i: coeff for (i, k), coeff in c.items()}
            
            add_term(u_coeffs, v_coeffs, w_coeffs)

    # Split B (4x6) into three 4x2 blocks
    # B_1: cols 0,1
    # B_2: cols 2,3
    # B_3: cols 4,5
    
    # Top 2 rows of A (rows 0,1)
    # A_top is 2x4
    # Split A_top into A_top_L (cols 0,1) and A_top_R (cols 2,3)
    
    for block_idx in range(3):
        k_start = block_idx * 2
        # B_block is 4x2, starting at col k_start
        # B_block rows 0,1 is B_T, rows 2,3 is B_B
        
        # Top part: 2x2x2 blocks
        # A_top_L (rows 0,1; cols 0,1) * B_T (rows 0,1; cols k_start, k_start+1)
        # Result adds to C_top (rows 0,1; cols k_start, k_start+1)
        strassen_2x2x2((0, 0), (0, k_start), (0, k_start))
        
        # A_top_R (rows 0,1; cols 2,3) * B_B (rows 2,3; cols k_start, k_start+1)
        # Result adds to C_top (rows 0,1; cols k_start, k_start+1)
        strassen_2x2x2((0, 2), (2, k_start), (0, k_start))
        
        # Bottom part: 1x2x2 blocks
        # A_bot (row 2; cols 0,1) * B_T (rows 0,1; cols k_start, k_start+1)
        # Result adds to C_bot (row 2; cols k_start, k_start+1)
        schoolbook_1x2x2((2, 0), (0, k_start), (2, k_start))
        
        # A_bot (row 2; cols 2,3) * B_B (rows 2,3; cols k_start, k_start+1)
        # Result adds to C_bot (row 2; cols k_start, k_start+1)
        schoolbook_1x2x2((2, 2), (2, k_start), (2, k_start))

    # Trim unused columns if any
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
