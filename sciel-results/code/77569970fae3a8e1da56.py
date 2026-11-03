"""Improved program using block-Strassen decomposition for 2x4 * 4x8 matrix multiplication."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    
    # We decompose the multiplication into 8 Strassen 2x2x2 blocks.
    # Structure:
    # A is 2x4, split into A0 (2x2), A1 (2x2)
    # B is 4x8, split into B0 (4x4), B1 (4x4)
    # B0 split into B0T (2x4), B0B (2x4)
    # B1 split into B1T (2x4), B1B (2x4)
    # 
    # C = A0 * B0T + A1 * B0B + A0 * B1T + A1 * B1B
    #
    # Each term is 2x2 * 2x4, which we split into two 2x2 * 2x2 multiplications
    # and apply Strassen to each.
    #
    # Total: 8 Strassen calls, each rank 7, so total rank = 56.
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition:
    # For M = X * Y where X, Y are 2x2:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M = [[m00, m01], [m10, m11]]
    #
    # Strassen products:
    # P1 = x00 * (y11 - y01)
    # P2 = (x10 + x00) * y01
    # P3 = (x01 + x11) * y10
    # P4 = x11 * (y00 - y10)
    # P5 = (x00 + x11) * (y00 + y11)
    # P6 = (x11 - x10) * (y00 + y10)
    # P7 = (x00 - x01) * (y01 + y11)
    #
    # m00 = P5 + P4 - P2 - P7
    # m01 = P1 + P5
    # m10 = P3 + P4
    # m11 = P5 + P1 - P3 - P6
    
    def strassen_block(offset, x_idx, y_idx, m_idx):
        """
        Add one Strassen 2x2x2 block to U, V, W.
        
        x_idx: list of 4 global U indices for the 2x2 X matrix [x00, x01, x10, x11]
        y_idx: list of 4 global V indices for the 2x2 Y matrix [y00, y01, y10, y11]
        m_idx: list of 4 global W indices for the 2x2 M matrix [m00, m01, m10, m11]
        offset: starting column in U, V, W for this Strassen block
        """
        x00, x01, x10, x11 = x_idx
        y00, y01, y10, y11 = y_idx
        m00, m01, m10, m11 = m_idx
        
        # P1 = x00 * (y11 - y01)
        c = offset
        U[x00][c] = 1
        V[y11][c] = 1
        V[y01][c] = -1
        W[m01][c] = 1
        W[m11][c] = 1
        c += 1
        
        # P2 = (x10 + x00) * y01
        U[x10][c] = 1
        U[x00][c] = 1
        V[y01][c] = 1
        W[m00][c] = -1
        c += 1
        
        # P3 = (x01 + x11) * y10
        U[x01][c] = 1
        U[x11][c] = 1
        V[y10][c] = 1
        W[m10][c] = 1
        W[m11][c] = -1
        c += 1
        
        # P4 = x11 * (y00 - y10)
        U[x11][c] = 1
        V[y00][c] = 1
        V[y10][c] = -1
        W[m00][c] = 1
        W[m10][c] = 1
        c += 1
        
        # P5 = (x00 + x11) * (y00 + y11)
        U[x00][c] = 1
        U[x11][c] = 1
        V[y00][c] = 1
        V[y11][c] = 1
        W[m00][c] = 1
        W[m01][c] = 1
        W[m10][c] = 1
        W[m11][c] = 1
        c += 1
        
        # P6 = (x11 - x10) * (y00 + y10)
        U[x11][c] = 1
        U[x10][c] = -1
        V[y00][c] = 1
        V[y10][c] = 1
        W[m11][c] = -1
        c += 1
        
        # P7 = (x00 - x01) * (y01 + y11)
        U[x00][c] = 1
        U[x01][c] = -1
        V[y01][c] = 1
        V[y11][c] = 1
        W[m00][c] = -1
        W[m11][c] = -1
        c += 1
    
    # Global index conventions:
    # U index: i*m + j where i in {0,1}, j in {0,1,2,3}
    #   A[i,j] -> U[i*4 + j]
    # V index: j*p + k where j in {0,1,2,3}, k in {0,...,7}
    #   B[j,k] -> V[j*8 + k]
    # W index: k*n + i where k in {0,...,7}, i in {0,1}
    #   C[k,i] -> W[k*2 + i]
    
    # Define the 8 Strassen blocks:
    # For y in {0,1} (which B half: B0 or B1)
    #   For x in {0,1} (which A half: A0 or A1)
    #     For z in {T, B} (top or bottom of B block)
    #       For side in {L, R} (left or right 2x2 of the 2x4 B subblock)
    #
    # A_x: rows i=0,1; cols j = 2*x + {0,1}
    #   A0: A[:, 0:2] -> U indices [0, 1, 4, 5] for [x00, x01, x10, x11]
    #   A1: A[:, 2:4] -> U indices [2, 3, 6, 7]
    #
    # B_{y,z,side}: 
    #   B0 = B[:, 0:4], B1 = B[:, 4:8]
    #   B0T = B0[0:2, :] = B[0:2, 0:4]
    #   B0B = B0[2:4, :] = B[2:4, 0:4]
    #   B1T = B1[0:2, :] = B[0:2, 4:8]
    #   B1B = B1[2:4, :] = B[2:4, 4:8]
    #   Each 2x4 split into left (cols 0:2 of that block) and right (cols 2:4)
    #
    # For B_{y,z,side}, the 2x2 Y matrix has:
    #   Rows: determined by z (top or bottom of B block)
    #   Cols: determined by y and side
    #
    # Let's enumerate all 8 blocks:
    
    # Block parameters: (A_half, B_half, B_row_block, B_col_side)
    # A_half: 0 -> A0, 1 -> A1
    # B_half: 0 -> B0 (cols 0:4), 1 -> B1 (cols 4:8)
    # B_row_block: 0 -> top (rows 0:2), 1 -> bottom (rows 2:4)
    # B_col_side: 0 -> left (local cols 0:2), 1 -> right (local cols 2:4)
    
    # Mapping from A_half to U indices [x00, x01, x10, x11]:
    # A[i, 2*A_half + local_j]
    # x00 = U[0*4 + 2*A_half + 0]
    # x01 = U[0*4 + 2*A_half + 1]
    # x10 = U[1*4 + 2*A_half + 0]
    # x11 = U[1*4 + 2*A_half + 1]
    
    # Mapping from (B_half, B_row_block, B_col_side) to V indices [y00, y01, y10, y11]:
    # B[2*B_row_block + local_i, 4*B_half + 2*B_col_side + local_j]
    # y00 = V[(2*B_row_block + 0)*8 + 4*B_half + 2*B_col_side + 0]
    # y01 = V[(2*B_row_block + 0)*8 + 4*B_half + 2*B_col_side + 1]
    # y10 = V[(2*B_row_block + 1)*8 + 4*B_half + 2*B_col_side + 0]
    # y11 = V[(2*B_row_block + 1)*8 + 4*B_half + 2*B_col_side + 1]
    
    # Mapping from (B_half, B_row_block, B_col_side) to W indices [m00, m01, m10, m11]:
    # C[k, i] where k = 4*B_half + 2*B_col_side + local_j, i = local_i
    # m00 = W[(4*B_half + 2*B_col_side + 0)*2 + 0]
    # m01 = W[(4*B_half + 2*B_col_side + 1)*2 + 0]
    # m10 = W[(4*B_half + 2*B_col_side + 0)*2 + 1]
    # m11 = W[(4*B_half + 2*B_col_side + 1)*2 + 1]
    
    # But wait: the mapping between A_half and B_row_block is constrained!
    # A0 multiplies B_{y,T}, A1 multiplies B_{y,B}
    # So for each B_half y:
    #   A0 (x=0) pairs with B_row_block=0 (top)
    #   A1 (x=1) pairs with B_row_block=1 (bottom)
    
    offset = 0
    for B_half in range(2):  # y: 0 for B0, 1 for B1
        for B_col_side in range(2):  # 0 for left, 1 for right
            # A0 * B_{B_half, top, B_col_side}
            A_half = 0
            B_row_block = 0
            
            x_idx = [
                0 * 4 + 2 * A_half + 0,  # x00
                0 * 4 + 2 * A_half + 1,  # x01
                1 * 4 + 2 * A_half + 0,  # x10
                1 * 4 + 2 * A_half + 1,  # x11
            ]
            y_idx = [
                (2 * B_row_block + 0) * 8 + 4 * B_half + 2 * B_col_side + 0,  # y00
                (2 * B_row_block + 0) * 8 + 4 * B_half + 2 * B_col_side + 1,  # y01
                (2 * B_row_block + 1) * 8 + 4 * B_half + 2 * B_col_side + 0,  # y10
                (2 * B_row_block + 1) * 8 + 4 * B_half + 2 * B_col_side + 1,  # y11
            ]
            m_idx = [
                (4 * B_half + 2 * B_col_side + 0) * 2 + 0,  # m00
                (4 * B_half + 2 * B_col_side + 1) * 2 + 0,  # m01
                (4 * B_half + 2 * B_col_side + 0) * 2 + 1,  # m10
                (4 * B_half + 2 * B_col_side + 1) * 2 + 1,  # m11
            ]
            strassen_block(offset, x_idx, y_idx, m_idx)
            offset += 7
            
            # A1 * B_{B_half, bottom, B_col_side}
            A_half = 1
            B_row_block = 1
            
            x_idx = [
                0 * 4 + 2 * A_half + 0,  # x00
                0 * 4 + 2 * A_half + 1,  # x01
                1 * 4 + 2 * A_half + 0,  # x10
                1 * 4 + 2 * A_half + 1,  # x11
            ]
            y_idx = [
                (2 * B_row_block + 0) * 8 + 4 * B_half + 2 * B_col_side + 0,  # y00
                (2 * B_row_block + 0) * 8 + 4 * B_half + 2 * B_col_side + 1,  # y01
                (2 * B_row_block + 1) * 8 + 4 * B_half + 2 * B_col_side + 0,  # y10
                (2 * B_row_block + 1) * 8 + 4 * B_half + 2 * B_col_side + 1,  # y11
            ]
            m_idx = [
                (4 * B_half + 2 * B_col_side + 0) * 2 + 0,  # m00
                (4 * B_half + 2 * B_col_side + 1) * 2 + 0,  # m01
                (4 * B_half + 2 * B_col_side + 0) * 2 + 1,  # m10
                (4 * B_half + 2 * B_col_side + 1) * 2 + 1,  # m11
            ]
            strassen_block(offset, x_idx, y_idx, m_idx)
            offset += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
