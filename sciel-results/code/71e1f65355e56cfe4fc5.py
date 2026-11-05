import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Helper to add a schoolbook term for a single (i, j, k)
    def add_schoolbook(i, j, k):
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[i*5 + j] = 1
        v_vec[j*5 + k] = 1
        w_vec[k*5 + i] = 1
        add_col(u_vec, v_vec, w_vec)

    # Helper to add the 7 Strassen factors for a 2x2x2 block at offsets (io, jo, ko)
    # This is used for the 2x2x2 blocks in the 2+2+1 partition, 
    # and for the inner 2x2x2 decompositions in the 4x4x4 block.
    def add_strassen_2x2(io, jo, ko):
        # Factor 0: M1 = (A00 + A11)(B00 + B11) -> C00, C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(io+0)*5 + (jo+0)] = 1
        u_vec[(io+1)*5 + (jo+1)] = 1
        v_vec[(jo+0)*5 + (ko+0)] = 1
        v_vec[(jo+1)*5 + (ko+1)] = 1
        w_vec[(ko+0)*5 + (io+0)] = 1
        w_vec[(ko+1)*5 + (io+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 1: M2 = (A10 + A11)B00 -> C10, C11(-1)
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(io+1)*5 + (jo+0)] = 1
        u_vec[(io+1)*5 + (jo+1)] = 1
        v_vec[(jo+0)*5 + (ko+0)] = 1
        w_vec[(ko+0)*5 + (io+1)] = 1
        w_vec[(ko+1)*5 + (io+1)] = -1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 2: M3 = A00(B01 - B11) -> C01, C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(io+0)*5 + (jo+0)] = 1
        v_vec[(jo+0)*5 + (ko+1)] = 1
        v_vec[(jo+1)*5 + (ko+1)] = -1
        w_vec[(ko+1)*5 + (io+0)] = 1
        w_vec[(ko+1)*5 + (io+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 3: M4 = A11(B10 - B00) -> C00, C10
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(io+1)*5 + (jo+1)] = 1
        v_vec[(jo+1)*5 + (ko+0)] = 1
        v_vec[(jo+0)*5 + (ko+0)] = -1
        w_vec[(ko+0)*5 + (io+0)] = 1
        w_vec[(ko+0)*5 + (io+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 4: M5 = (A00 + A01)B11 -> C00(-1), C01
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(io+0)*5 + (jo+0)] = 1
        u_vec[(io+0)*5 + (jo+1)] = 1
        v_vec[(jo+1)*5 + (ko+1)] = 1
        w_vec[(ko+0)*5 + (io+0)] = -1
        w_vec[(ko+1)*5 + (io+0)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 5: M6 = (A10 - A00)(B00 + B01) -> C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(io+1)*5 + (jo+0)] = 1
        u_vec[(io+0)*5 + (jo+0)] = -1
        v_vec[(jo+0)*5 + (ko+0)] = 1
        v_vec[(jo+0)*5 + (ko+1)] = 1
        w_vec[(ko+1)*5 + (io+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 6: M7 = (A01 - A11)(B10 + B11) -> C00
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(io+0)*5 + (jo+1)] = 1
        u_vec[(io+1)*5 + (jo+1)] = -1
        v_vec[(jo+1)*5 + (ko+0)] = 1
        v_vec[(jo+1)*5 + (ko+1)] = 1
        w_vec[(ko+0)*5 + (io+0)] = 1
        add_col(u_vec, v_vec, w_vec)

    # Function to add the rank-49 decomposition for a 4x4x4 block at offsets (io, jo, ko)
    def add_4x4_strassen(io, jo, ko):
        # The 4x4x4 block is treated as 2x2 blocks of 2x2 matrices.
        # We apply Strassen to the 2x2 block structure.
        # There are 7 Strassen terms.
        # For each term, we identify the linear combinations of the 2x2 blocks of A and B.
        # Then we apply the 7 Strassen factors to the 2x2 internal structure.
        
        # Define the 7 Strassen terms for the 2x2 block level.
        # Each term is defined by:
        # - U_blocks: list of (block_row, block_col, coeff) for A
        # - V_blocks: list of (block_row, block_col, coeff) for B
        # - W_blocks: list of (block_row, block_col, coeff) for C (reconstruction signs)
        
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        # Contributes to C00 (+1) and C11 (+1)
        terms = [
            {
                "u": [(0,0,1), (1,1,1)],
                "v": [(0,0,1), (1,1,1)],
                "w": [(0,0,1), (1,1,1)]
            },
            {
                "u": [(1,0,1), (1,1,1)],
                "v": [(0,0,1)],
                "w": [(1,0,1), (1,1,-1)]
            },
            {
                "u": [(0,0,1)],
                "v": [(0,1,1), (1,1,-1)],
                "w": [(0,1,1), (1,1,1)]
            },
            {
                "u": [(1,1,1)],
                "v": [(1,0,1), (0,0,-1)],
                "w": [(0,0,1), (1,0,1)]
            },
            {
                "u": [(0,0,1), (0,1,1)],
                "v": [(1,1,1)],
                "w": [(0,0,-1), (0,1,1)]
            },
            {
                "u": [(1,0,1), (0,0,-1)],
                "v": [(0,0,1), (0,1,1)],
                "w": [(1,1,1)]
            },
            {
                "u": [(0,1,1), (1,1,-1)],
                "v": [(1,0,1), (1,1,1)],
                "w": [(0,0,1)]
            }
        ]
        
        # For each term, we need to decompose the product of the two 2x2 matrices.
        # The "matrices" here are 2x2 blocks.
        # Let's say for a term, U is formed by blocks $A_{b1}, A_{b2}$ etc.
        # The actual multiplication is between the 2x2 matrices formed by these blocks.
        # We can iterate over the 7 Strassen factors for the 2x2 internal structure.
        
        # To simplify, we can just call a helper that adds the 7 columns for a 2x2x2 product
        # where the "scalars" are 2x2 blocks.
        
        def add_block_strassen_term(u_defs, v_defs, w_defs):
            # u_defs: list of (br, bc, coeff)
            # v_defs: list of (br, bc, coeff)
            # w_defs: list of (br, bc, coeff)
            
            # We need to add 7 columns.
            # Each column corresponds to one of the 7 Strassen factors for the 2x2 internal structure.
            
            # Let's define the 7 factors for a 2x2 matrix product X*Y -> Z.
            # X, Y, Z are 2x2 matrices.
            # The factors are (u_factor, v_factor, w_factor) where each is a 2x2 matrix of coefficients.
            
            factors = [
                # Factor 1: (X00+X11)(Y00+Y11) -> Z00, Z11
                (
                    [[1, 0], [0, 1]], 
                    [[1, 0], [0, 1]], 
                    [[1, 0], [0, 1]]
                ),
                # Factor 2: (X10+X11)Y00 -> Z10, Z11(-1)
                (
                    [[0, 0], [1, 1]], 
                    [[1, 0], [0, 0]], 
                    [[0, 0], [1, -1]]
                ),
                # Factor 3: X00(Y01-Y11) -> Z01, Z11
                (
                    [[1, 0], [0, 0]], 
                    [[0, 1], [0, -1]], 
                    [[0, 1], [0, 1]]
                ),
                # Factor 4: X11(Y10-Y00) -> Z00, Z10
                (
                    [[0, 0], [0, 1]], 
                    [[0, -1], [1, 0]], 
                    [[1, 0], [1, 0]]
                ),
                # Factor 5: (X00+X01)Y11 -> Z00(-1), Z01
                (
                    [[1, 1], [0, 0]], 
                    [[0, 0], [0, 1]], 
                    [[-1, 0], [1, 0]]
                ),
                # Factor 6: (X10-X00)(Y00+Y01) -> Z11
                (
                    [[-1, 0], [1, 0]], 
                    [[1, 1], [0, 0]], 
                    [[0, 0], [0, 1]]
                ),
                # Factor 7: (X01-X11)(Y10+Y11) -> Z00
                (
                    [[0, 1], [0, -1]], 
                    [[0, 0], [1, 1]], 
                    [[1, 0], [0, 0]]
                )
            ]
            
            for f_idx in range(7):
                u_mat, v_mat, w_mat = factors[f_idx]
                
                u_vec = [0]*25
                v_vec = [0]*25
                w_vec = [0]*25
                
                # Fill U_vec
                # u_mat is 2x2. u_mat[r][c] is coeff for X_{r,c}
                # X is formed by linear combo of A blocks.
                # But wait, X is a 2x2 matrix of 2x2 blocks? No.
                # In the block Strassen, X is a 2x2 matrix where each entry is a 2x2 block of A.
                # So X_{r,c} is the 2x2 block $A_{r,c}$ (if coeff is 1).
                # The factor u_mat[r][c] multiplies the entire block $A_{r,c}$.
                
                # So for each (br, bc, coeff) in u_defs:
                # The block is at global offset io + 2*br, jo + 2*bc.
                # The entries in this block are multiplied by coeff * u_mat[r][c] where r,c are the block indices?
                # No. The Strassen factor u_mat is for the 2x2 structure of the blocks.
                # Let's clarify.
                # The "matrix" X in the inner Strassen is a 2x2 matrix of scalars? No.
                # The "matrix" X is a 2x2 matrix where the entries are the 2x2 blocks of A.
                # So X_{0,0} is the 2x2 block A_{0,0}.
                # The factor u_mat is a 2x2 matrix of coefficients for these blocks.
                # So if u_mat[0][0] is 1, it means we take the block A_{0,0} with coeff 1.
                # And the entire 2x2 block A_{0,0} is used.
                
                # So, for a given factor f_idx:
                # We iterate over the blocks defined in u_defs.
                # For each block (br, bc, coeff_block) in u_defs:
                #   We look at u_mat[br][bc]. Let this be coeff_inner.
                #   The total coefficient for the entries in the global block (io+2*br, jo+2*bc) is coeff_block * coeff_inner.
                #   We add this coefficient to all 4 entries in that 2x2 block?
                #   No. The Strassen factor u_mat[r][c] applies to the entry X_{r,c}.
                #   X_{r,c} IS the block A_{r,c}.
                #   So the coefficient applies to the entire block?
                #   Yes, because the tensor decomposition is linear.
                #   The rank-1 term is $u \otimes v \otimes w$.
                #   If $u$ has a coefficient $\alpha$ at position $(i,j)$, it means $A_{i,j}$ is multiplied by $\alpha$.
                #   Since the block $A_{br,bc}$ consists of 4 entries, and the factor $u_{br,bc}$ applies to the "block" as a unit in the outer Strassen,
                #   does it apply to all 4 entries?
                #   Yes. The outer Strassen treats the block as a scalar.
                #   So $u_{br,bc}$ is the coefficient for the block $A_{br,bc}$.
                #   This coefficient multiplies all 4 entries in that block.
                
                for br, bc, coeff_block in u_defs:
                    coeff_inner = u_mat[br][bc]
                    if coeff_inner != 0:
                        total_coeff = coeff_block * coeff_inner
                        if total_coeff != 0:
                            # Add to the 2x2 block at (io+2*br, jo+2*bc)
                            for r in range(2):
                                for c in range(2):
                                    gi = io + 2*br + r
                                    gj = jo + 2*bc + c
                                    u_vec[gi*5 + gj] += total_coeff
                
                for br, bc, coeff_block in v_defs:
                    coeff_inner = v_mat[br][bc]
                    if coeff_inner != 0:
                        total_coeff = coeff_block * coeff_inner
                        if total_coeff != 0:
                            for r in range(2):
                                for c in range(2):
                                    gj = jo + 2*br + r
                                    gk = ko + 2*bc + c
                                    v_vec[gj*5 + gk] += total_coeff

                for br, bc, coeff_block in w_defs:
                    coeff_inner = w_mat[br][bc]
                    if coeff_inner != 0:
                        total_coeff = coeff_block * coeff_inner
                        if total_coeff != 0:
                            for r in range(2):
                                for c in range(2):
                                    gk = ko + 2*br + r
                                    gi = io + 2*bc + c
                                    w_vec[gk*5 + gi] += total_coeff

                add_col(u_vec, v_vec, w_vec)

        # Call for each of the 7 terms
        for t in terms:
            add_block_strassen_term(t["u"], t["v"], t["w"])

    # Partition 4+1
    # Block (0,0,0) is 4x4x4
    add_4x4_strassen(0, 0, 0)
    
    # Mixed blocks:
    # (0,0,1): 4x4x1
    for i in range(4):
        for j in range(4):
            add_schoolbook(i, j, 4)
            
    # (0,1,0): 4x1x4
    for i in range(4):
        for k in range(4):
            add_schoolbook(i, 4, k)
            
    # (0,1,1): 4x1x1
    for i in range(4):
        add_schoolbook(i, 4, 4)
        
    # (1,0,0): 1x4x4
    for j in range(4):
        for k in range(4):
            add_schoolbook(4, j, k)
            
    # (1,0,1): 1x4x1
    for j in range(4):
        add_schoolbook(4, j, 4)
        
    # (1,1,0): 1x1x4
    for k in range(4):
        add_schoolbook(4, 4, k)
        
    # (1,1,1): 1x1x1
    add_schoolbook(4, 4, 4)

    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
