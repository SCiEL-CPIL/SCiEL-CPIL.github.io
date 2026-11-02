import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen's algorithm on 2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 matrices
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x10)y11
    # M6 = (x11-x00)(y00+y10)
    # M7 = (x01-x11)(y11+y10)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We have 4 block products:
    # P1 = A_L B_LL (A_L is A[:, 0:2], B_LL is B[0:2, 0:2]) -> C_L[:, 0:2]
    # P2 = A_R B_RL (A_R is A[:, 2:4], B_RL is B[2:4, 0:2]) -> C_L[:, 0:2]
    # P3 = A_L B_LR (A_L is A[:, 0:2], B_LR is B[0:2, 2:4]) -> C_R[:, 2:4]
    # P4 = A_R B_RR (A_R is A[:, 2:4], B_RR is B[2:4, 2:4]) -> C_R[:, 2:4]
    
    def add_strassen(col_offset, i_base, j_base_A, j_base_B, k_base):
        # i_base: row offset in A and C (0 or 1? No, A is 2x4, so i is 0..1)
        # Actually, A is 2x4. i is 0..1. j is 0..3.
        # B is 4x4. j is 0..3. k is 0..3.
        # C is 2x4. i is 0..1. k is 0..3.
        
        # j_base_A is the starting column of A for this block (0 or 2)
        # j_base_B is the starting row of B for this block (0 or 2)
        # k_base is the starting column of B for this block (0 or 2)
        # The output goes to C at columns k_base .. k_base+1
        
        # Strassen terms
        # M1: (A[i, j0] + A[i, j1]) * (B[j0, k0] + B[j1, k1])
        # We need to map these to the global U, V, W.
        
        # Let's define the indices for the 2x2 block
        # A indices: (i, j0), (i, j0+1), (i+1, j0), (i+1, j0+1)
        # But i is fixed? No, the block is 2x2 in A? No, A is 2x4.
        # The block A_L is 2x2. So i is 0..1, j is 0..1.
        # So for each i in 0..1, we have A[i, j0] and A[i, j0+1].
        
        # Wait, Strassen's algorithm works on 2x2 matrices.
        # Here, for each i (0 and 1), we have a row of A.
        # But Strassen combines rows of A.
        # So we must apply Strassen to the 2x2 block A_L and B_LL.
        # This produces a 2x2 matrix C_L.
        # The rank-1 terms of Strassen are linear combinations of entries of A_L and B_LL.
        # These linear combinations correspond to entries in U and V.
        
        # Let's explicitly write the 7 terms for the block.
        # The block A has indices (i, j) where i in {0, 1} relative to block, j in {0, 1}.
        # Global A indices: (i_base + i, j_base_A + j)
        # The block B has indices (j, k) where j in {0, 1}, k in {0, 1}.
        # Global B indices: (j_base_B + j, k_base + k)
        # The block C has indices (i, k) where i in {0, 1}, k in {0, 1}.
        # Global C indices: (i_base + i, k_base + k) -> W index: (k_base + k) * n + (i_base + i)
        
        # Strassen M1: (A00 + A11)(B00 + B11)
        # U part: A00 + A11. This is a linear combination of A[0,0] and A[1,1].
        # In the global U, we need to set U[i, j] for the relevant i, j.
        # But U is a matrix of size (n*m, rank).
        # Each rank-1 term corresponds to a column in U, V, W.
        # For M1, we have one column.
        # U column: 1 at (0, j0), 1 at (1, j0+1). (Assuming i_base=0)
        # V column: 1 at (j0, k0), 1 at (j0+1, k0+1).
        # W column: 1 at (k0, 0), 1 at (k0, 1), 1 at (k0+1, 0), 1 at (k0+1, 1)?
        # No, W is for the output C.
        # C = M1 * (something)? No.
        # Strassen computes Z = X Y.
        # Z00 = M1 + M4 - M5 + M7.
        # So M1 contributes +1 to Z00, +1 to Z11.
        # M4 contributes +1 to Z00, +1 to Z10.
        # M5 contributes -1 to Z00, +1 to Z01.
        # M7 contributes +1 to Z00, -1 to Z11.
        # M3 contributes +1 to Z01, +1 to Z11.
        # M2 contributes +1 to Z10, -1 to Z11.
        # M6 contributes +1 to Z11.
        
        # So for each M, we know its contribution to Z00, Z01, Z10, Z11.
        # This contribution goes into W.
        # W is indexed by (k, i).
        # Z00 is C[0,0] relative to block. Global C[i_base+0, k_base+0]. W index: (k_base+0)*n + (i_base+0).
        # Z01 is C[0,1] relative to block. Global C[i_base+0, k_base+1]. W index: (k_base+1)*n + (i_base+0).
        # Z10 is C[1,0] relative to block. Global C[i_base+1, k_base+0]. W index: (k_base+0)*n + (i_base+1).
        # Z11 is C[1,1] relative to block. Global C[i_base+1, k_base+1]. W index: (k_base+1)*n + (i_base+1).
        
        # Now for U and V.
        # M1: (X00 + X11)(Y00 + Y11)
        # U part: X00 + X11. Global A indices: (i_base+0, j_base_A+0) and (i_base+1, j_base_A+1).
        # So U[col][ (i_base+0)*m + (j_base_A+0) ] = 1
        # U[col][ (i_base+1)*m + (j_base_A+1) ] = 1
        # V part: Y00 + Y11. Global B indices: (j_base_B+0, k_base+0) and (j_base_B+1, k_base+1).
        # V[col][ (j_base_B+0)*p + (k_base+0) ] = 1
        # V[col][ (j_base_B+1)*p + (k_base+1) ] = 1
        # W part: +1 to Z00, +1 to Z11.
        # W[col][ (k_base+0)*n + (i_base+0) ] = 1
        # W[col][ (k_base+1)*n + (i_base+1) ] = 1
        
        # We can generalize this for all 7 terms.
        
        terms = [
            # (U_indices, V_indices, W_contribs)
            # U_indices: list of (i, j) relative to block
            # V_indices: list of (j, k) relative to block
            # W_contribs: dict {(i, k): coeff}
            
            # M1: (X00 + X11)(Y00 + Y11)
            {
                "U": [(0, 0), (1, 1)],
                "V": [(0, 0), (1, 1)],
                "W": {(0, 0): 1, (1, 1): 1}
            },
            # M2: (X10 + X11)Y00
            {
                "U": [(1, 0), (1, 1)],
                "V": [(0, 0)],
                "W": {(1, 0): 1, (1, 1): -1}
            },
            # M3: X00(Y10 - Y11)
            {
                "U": [(0, 0)],
                "V": [(1, 0), (1, 1)],
                "W": {(0, 1): 1, (1, 1): 1}
            },
            # M4: X11(Y01 - Y00)
            {
                "U": [(1, 1)],
                "V": [(0, 1), (0, 0)],
                "W": {(0, 0): 1, (1, 0): 1}
            },
            # M5: (X00 + X10)Y11
            {
                "U": [(0, 0), (1, 0)],
                "V": [(1, 1)],
                "W": {(0, 0): -1, (0, 1): 1}
            },
            # M6: (X11 - X00)(Y00 + Y10)
            {
                "U": [(1, 1), (0, 0)],
                "V": [(0, 0), (1, 0)],
                "W": {(1, 1): 1}
            },
            # M7: (X01 - X11)(Y11 + Y10)
            {
                "U": [(0, 1), (1, 1)],
                "V": [(1, 1), (1, 0)],
                "W": {(0, 0): 1, (1, 1): -1}
            }
        ]
        
        # Note: M6 U part: X11 - X00. So (1,1) coeff 1, (0,0) coeff -1.
        # My dict above uses lists. I need to handle coefficients.
        # Let's redefine terms with coefficients.
        
        terms = [
            # M1
            (
                { (0,0): 1, (1,1): 1 },
                { (0,0): 1, (1,1): 1 },
                { (0,0): 1, (1,1): 1 }
            ),
            # M2
            (
                { (1,0): 1, (1,1): 1 },
                { (0,0): 1 },
                { (1,0): 1, (1,1): -1 }
            ),
            # M3
            (
                { (0,0): 1 },
                { (1,0): 1, (1,1): -1 },
                { (0,1): 1, (1,1): 1 }
            ),
            # M4
            (
                { (1,1): 1 },
                { (0,1): 1, (0,0): -1 },
                { (0,0): 1, (1,0): 1 }
            ),
            # M5
            (
                { (0,0): 1, (1,0): 1 },
                { (1,1): 1 },
                { (0,0): -1, (0,1): 1 }
            ),
            # M6
            (
                { (1,1): 1, (0,0): -1 },
                { (0,0): 1, (1,0): 1 },
                { (1,1): 1 }
            ),
            # M7
            (
                { (0,1): 1, (1,1): -1 },
                { (1,1): 1, (1,0): 1 },
                { (0,0): 1, (1,1): -1 }
            )
        ]
        
        for t in terms:
            U_dict, V_dict, W_dict = t
            # Fill U
            for (ri, rj), coeff in U_dict.items():
                gi = i_base + ri
                gj = j_base_A + rj
                U[gi * m + gj][col_offset] += coeff
            # Fill V
            for (rj, rk), coeff in V_dict.items():
                gj = j_base_B + rj
                gk = k_base + rk
                V[gj * p + gk][col_offset] += coeff
            # Fill W
            for (ri, rk), coeff in W_dict.items():
                gi = i_base + ri
                gk = k_base + rk
                W[gk * n + gi][col_offset] += coeff
            col_offset += 1
        
        return col_offset

    col = 0
    # P1: A_L (cols 0,1) * B_LL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # i_base = 0 (rows 0,1 of A and C)
    # j_base_A = 0
    # j_base_B = 0
    # k_base = 0
    col = add_strassen(col, 0, 0, 0, 0)
    
    # P2: A_R (cols 2,3) * B_RL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # i_base = 0
    # j_base_A = 2
    # j_base_B = 2
    # k_base = 0
    col = add_strassen(col, 0, 2, 2, 0)
    
    # P3: A_L (cols 0,1) * B_LR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # i_base = 0
    # j_base_A = 0
    # j_base_B = 0
    # k_base = 2
    col = add_strassen(col, 0, 0, 0, 2)
    
    # P4: A_R (cols 2,3) * B_RR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    # i_base = 0
    # j_base_A = 2
    # j_base_B = 2
    # k_base = 2
    col = add_strassen(col, 0, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
