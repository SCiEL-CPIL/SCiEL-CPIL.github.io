"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get index in flattened 4x4 matrix
    def get_idx(r, c):
        return r * 4 + c

    # Define the 7 outer Strassen terms
    # Each term is defined by (X_blocks, Y_blocks)
    # X_blocks is a dict: {(r,c): (block_r, block_c, coeff)}
    # Actually, it's easier to define the linear combinations of the 4x4 matrix entries directly.
    
    # Let's define the linear combinations for X and Y for each of the 7 outer terms.
    # X is 2x2, Y is 2x2.
    # X[r][c] is a linear combination of A entries.
    # Y[r][c] is a linear combination of B entries.
    
    # Outer Term 1: M1 = (A11 + A22)(B11 + B22)
    # X1 = A11 + A22
    # Y1 = B11 + B22
    
    # We will represent X and Y as 2x2 lists of lists of (index, coeff)
    
    def make_block_combination(blocks):
        # blocks is a list of (block_row, block_col, coeff)
        # Returns a 2x2 list of lists of (index, coeff)
        res = [[[] for _ in range(2)] for _ in range(2)]
        for br, bc, coeff in blocks:
            for r in range(2):
                for c in range(2):
                    idx = get_idx((br-1)*2 + r, (bc-1)*2 + c)
                    res[r][c].append((idx, coeff))
        return res

    # Define the 7 outer terms
    # X_k and Y_k are 2x2 matrices of linear combinations
    outer_terms = []
    
    # M1 = (A11 + A22)(B11 + B22)
    X1 = make_block_combination([(1,1,1), (2,2,1)])
    Y1 = make_block_combination([(1,1,1), (2,2,1)])
    outer_terms.append((X1, Y1))
    
    # M2 = (A21 + A22)B11
    X2 = make_block_combination([(2,1,1), (2,2,1)])
    Y2 = make_block_combination([(1,1,1)])
    outer_terms.append((X2, Y2))
    
    # M3 = A11(B12 - B22)
    X3 = make_block_combination([(1,1,1)])
    Y3 = make_block_combination([(1,2,1), (2,2,-1)])
    outer_terms.append((X3, Y3))
    
    # M4 = A22(B21 - B11)
    X4 = make_block_combination([(2,2,1)])
    Y4 = make_block_combination([(2,1,1), (1,1,-1)])
    outer_terms.append((X4, Y4))
    
    # M5 = (A11 + A12)B22
    X5 = make_block_combination([(1,1,1), (1,2,1)])
    Y5 = make_block_combination([(2,2,1)])
    outer_terms.append((X5, Y5))
    
    # M6 = (A21 - A11)(B11 + B12)
    X6 = make_block_combination([(2,1,1), (1,1,-1)])
    Y6 = make_block_combination([(1,1,1), (1,2,1)])
    outer_terms.append((X6, Y6))
    
    # M7 = (A12 - A22)(B21 + B22)
    X7 = make_block_combination([(1,2,1), (2,2,-1)])
    Y7 = make_block_combination([(2,1,1), (2,2,1)])
    outer_terms.append((X7, Y7))
    
    # Inner Strassen terms for 2x2 matrices
    # For X, Y, the 7 products P1..P7 are:
    # P1 = (X11 + X22)(Y11 + Y22)
    # P2 = (X21 + X22)Y11
    # P3 = X11(Y12 - Y22)
    # P4 = X22(Y21 - Y11)
    # P5 = (X11 + X12)Y22
    # P6 = (X21 - X11)(Y11 + Y12)
    # P7 = (X12 - X22)(Y21 + Y22)
    
    # We define the linear combinations for the factors of Pj
    # Factor 1 of Pj is a linear combination of X entries.
    # Factor 2 of Pj is a linear combination of Y entries.
    
    # X entries are themselves linear combinations of A entries.
    # So we need to substitute.
    
    # Let's define the inner terms structure
    # inner_terms[j] = (coeffs_X, coeffs_Y, coeffs_Z)
    # coeffs_X: dict mapping (r,c) in X to coeff
    # coeffs_Y: dict mapping (r,c) in Y to coeff
    # coeffs_Z: dict mapping (r,c) in Z to coeff
    
    inner_terms = []
    
    # P1
    coeffs_X1 = {(0,0): 1, (1,1): 1}
    coeffs_Y1 = {(0,0): 1, (1,1): 1}
    coeffs_Z1 = {(0,0): 1, (1,1): 1} # Z11 += P1, Z22 += P1
    inner_terms.append((coeffs_X1, coeffs_Y1, coeffs_Z1))
    
    # P2
    coeffs_X2 = {(1,0): 1, (1,1): 1}
    coeffs_Y2 = {(0,0): 1}
    coeffs_Z2 = {(1,0): 1, (1,1): -1} # Z21 += P2, Z22 -= P2
    inner_terms.append((coeffs_X2, coeffs_Y2, coeffs_Z2))
    
    # P3
    coeffs_X3 = {(0,0): 1}
    coeffs_Y3 = {(0,1): 1, (1,1): -1}
    coeffs_Z3 = {(0,1): 1, (1,1): 1} # Z12 += P3, Z22 += P3
    inner_terms.append((coeffs_X3, coeffs_Y3, coeffs_Z3))
    
    # P4
    coeffs_X4 = {(1,1): 1}
    coeffs_Y4 = {(1,0): 1, (0,0): -1}
    coeffs_Z4 = {(0,0): 1, (1,0): 1} # Z11 += P4, Z21 += P4
    inner_terms.append((coeffs_X4, coeffs_Y4, coeffs_Z4))
    
    # P5
    coeffs_X5 = {(0,0): 1, (0,1): 1}
    coeffs_Y5 = {(1,1): 1}
    coeffs_Z5 = {(0,0): -1, (0,1): 1} # Z11 -= P5, Z12 += P5
    inner_terms.append((coeffs_X5, coeffs_Y5, coeffs_Z5))
    
    # P6
    coeffs_X6 = {(1,0): 1, (0,0): -1}
    coeffs_Y6 = {(0,0): 1, (0,1): 1}
    coeffs_Z6 = {(1,1): 1} # Z22 += P6
    inner_terms.append((coeffs_X6, coeffs_Y6, coeffs_Z6))
    
    # P7
    coeffs_X7 = {(0,1): 1, (1,1): -1}
    coeffs_Y7 = {(1,0): 1, (1,1): 1}
    coeffs_Z7 = {(0,0): 1} # Z11 += P7
    inner_terms.append((coeffs_X7, coeffs_Y7, coeffs_Z7))
    
    col = 0
    for k in range(7):
        Xk, Yk = outer_terms[k]
        for j in range(7):
            coeffs_Xj, coeffs_Yj, coeffs_Zj = inner_terms[j]
            
            # Build U and V contributions
            # U row is A index, V row is B index
            # The term is sum_{a} u_a A_a * sum_{b} v_b B_b
            # We need to expand coeffs_Xj (on X) and Xk (on A)
            
            # First factor: X combination
            # X_rc = sum_{a} (Xk[rc][a_idx]) A_a
            # Factor = sum_{rc} coeffs_Xj[rc] * X_rc
            # Factor = sum_{rc} coeffs_Xj[rc] * sum_{a} (Xk[rc][a_idx]) A_a
            # Factor = sum_{a} (sum_{rc} coeffs_Xj[rc] * Xk[rc][a_idx]) A_a
            
            u_contrib = {}
            for rc, cx in coeffs_Xj.items():
                for a_idx, ca in Xk[rc[0]][rc[1]]:
                    u_contrib[a_idx] = u_contrib.get(a_idx, 0) + cx * ca
            
            # Second factor: Y combination
            v_contrib = {}
            for rc, cy in coeffs_Yj.items():
                for b_idx, cb in Yk[rc[0]][rc[1]]:
                    v_contrib[b_idx] = v_contrib.get(b_idx, 0) + cy * cb
            
            # Add to U and V
            for a_idx, val in u_contrib.items():
                if val != 0:
                    U[a_idx][col] = val
            
            for b_idx, val in v_contrib.items():
                if val != 0:
                    V[b_idx][col] = val
            
            # Build W contribution
            # Z_rc = sum_{j} coeffs_Zj[rc] P_j
            # Z is the result of M_k
            # M_k contributes to C blocks.
            # We need to map Z indices (0,1) to C block indices.
            # The outer term k corresponds to a specific combination of C blocks?
            # No, the outer Strassen terms M1..M7 are intermediate matrices.
            # The final C blocks are linear combinations of M1..M7.
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for a given outer term k, we need to know which C blocks it contributes to and with what coefficient.
            # Let's define outer_coeffs[k] = { (c_block_r, c_block_c): coeff }
            
            # We'll handle this outside the inner loop or by pre-computing.
            
            col += 1
            
    # We need to re-do the W part correctly.
    # The above loop only filled U and V.
    # We need to fill W based on how M_k contributes to C.
    
    # Let's restart the loop or just fill W now.
    # We have 49 columns.
    # col index in U/V/W corresponds to (k, j) where k is 0..6 and j is 0..6.
    # col = k * 7 + j
    
    # Define outer contributions to C blocks
    # C_block (br, bc) is a 2x2 matrix.
    # C_block[rc] is an entry.
    # C index = get_idx((br-1)*2 + rc_r, (bc-1)*2 + rc_c)
    
    outer_contribs = [
        {(1,1): 1, (2,2): 1},      # M1 -> C11, C22
        {(2,1): 1, (2,2): -1},     # M2 -> C21, C22
        {(1,2): 1, (2,2): 1},      # M3 -> C12, C22
        {(1,1): 1, (2,1): 1},      # M4 -> C11, C21
        {(1,1): -1, (1,2): 1},     # M5 -> C11, C12
        {(2,2): 1},                # M6 -> C22
        {(1,1): 1}                 # M7 -> C11
    ]
    
    # Re-iterate to fill W
    for k in range(7):
        for j in range(7):
            col = k * 7 + j
            coeffs_Zj = inner_terms[j][2]
            
            for (rc_r, rc_c), cz in coeffs_Zj.items():
                # Z entry (rc_r, rc_c) of M_k
                # M_k contributes to C blocks defined in outer_contribs[k]
                for (br, bc), co in outer_contribs[k].items():
                    # C entry is at row (br-1)*2 + rc_r, col (bc-1)*2 + rc_c
                    c_row = (br-1)*2 + rc_r
                    c_col = (bc-1)*2 + rc_c
                    c_idx = c_col * 4 + c_row # W[k*n+i]
                    coeff = cz * co
                    if coeff != 0:
                        W[c_idx][col] = coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
