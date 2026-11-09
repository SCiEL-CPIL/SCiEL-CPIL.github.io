import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # 2x2 Strassen decomposition
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 2: (A10+A11)B00 -> C10, -C11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 3: A00(B01-B11) -> C01, C11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 4: A11(B10-B00) -> C00, C10
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 5: (A00+A01)B11 -> -C00, C01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01) -> C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11) -> C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 4x4 Strassen decomposition
    # 7 block terms
    # L_A[k], L_B[k] are 2x2 matrices of coefficients for the 4 blocks of A/B
    # Blocks: 00, 01, 10, 11
    # Indices in 4x4:
    # A00: 0-3, A01: 4-7, A10: 8-11, A11: 12-15
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Define the 7 Strassen block combinations
    # k=0: M1 = (A00+A11)(B00+B11)
    # k=1: M2 = (A10+A11)B00
    # k=2: M3 = A00(B01-B11)
    # k=3: M4 = A11(B10-B00)
    # k=4: M5 = (A00+A01)B11
    # k=5: M6 = (A10-A00)(B00+B01)
    # k=6: M7 = (A01-A11)(B10+B11)
    
    # For each k, we have a 2x2 product of linear combinations
    # We use the 2x2 decomposition U2, V2, W2
    
    for k in range(7):
        # Determine which blocks are involved and their signs
        # L_A[k] is 2x2 matrix of block indices and signs
        # L_B[k] is 2x2 matrix of block indices and signs
        
        # Block indices: 00->0, 01->1, 10->2, 11->3
        # Offset for 4x4 indices:
        # Block 00: 0, Block 01: 4, Block 10: 8, Block 11: 12
        
        # Define L_A and L_B as lists of (block_idx, sign) for each of the 4 entries of the 2x2 matrix
        # Entry (i,l) of the 2x2 matrix
        
        if k == 0: # M1
            LA = [[(0, 1), (1, 0)], [(2, 0), (3, 1)]] # A00+A11 is not a 2x2 matrix, it's a sum of blocks
            # Wait, M1 = (A00+A11)(B00+B11).
            # A00+A11 is a 2x2 matrix where each entry is sum of corresponding entries of A00 and A11.
            # So L_A[0] is a 2x2 matrix where each entry is a linear combination of blocks.
            # Entry (0,0) of the 2x2 result comes from A00(0,0) + A11(0,0).
            # So for each entry (i,l) of the 2x2 matrix, we sum the contributions from the blocks.
            
            # Let's define the contribution of each block to the 2x2 matrix for L_A[k]
            # contrib_A[k][block_idx][i][l] = sign
            contrib_A = [[[0.0]*2 for _ in range(2)] for _ in range(4)]
            contrib_B = [[[0.0]*2 for _ in range(2)] for _ in range(4)]
            
            if k == 0:
                # A00 + A11
                for i in range(2):
                    for l in range(2):
                        contrib_A[0][i][l] = 1 # Block 00
                        contrib_A[3][i][l] = 1 # Block 11
                for i in range(2):
                    for l in range(2):
                        contrib_B[0][i][l] = 1 # Block 00
                        contrib_B[3][i][l] = 1 # Block 11
            elif k == 1:
                # (A10+A11) B00
                for i in range(2):
                    for l in range(2):
                        contrib_A[2][i][l] = 1 # Block 10
                        contrib_A[3][i][l] = 1 # Block 11
                for i in range(2):
                    for l in range(2):
                        contrib_B[0][i][l] = 1 # Block 00
            elif k == 2:
                # A00 (B01-B11)
                for i in range(2):
                    for l in range(2):
                        contrib_A[0][i][l] = 1 # Block 00
                for i in range(2):
                    for l in range(2):
                        contrib_B[1][i][l] = 1 # Block 01
                        contrib_B[3][i][l] = -1 # Block 11
            elif k == 3:
                # A11 (B10-B00)
                for i in range(2):
                    for l in range(2):
                        contrib_A[3][i][l] = 1 # Block 11
                for i in range(2):
                    for l in range(2):
                        contrib_B[2][i][l] = 1 # Block 10
                        contrib_B[0][i][l] = -1 # Block 00
            elif k == 4:
                # (A00+A01) B11
                for i in range(2):
                    for l in range(2):
                        contrib_A[0][i][l] = 1 # Block 00
                        contrib_A[1][i][l] = 1 # Block 01
                for i in range(2):
                    for l in range(2):
                        contrib_B[3][i][l] = 1 # Block 11
            elif k == 5:
                # (A10-A00) (B00+B01)
                for i in range(2):
                    for l in range(2):
                        contrib_A[2][i][l] = 1 # Block 10
                        contrib_A[0][i][l] = -1 # Block 00
                for i in range(2):
                    for l in range(2):
                        contrib_B[0][i][l] = 1 # Block 00
                        contrib_B[1][i][l] = 1 # Block 01
            elif k == 6:
                # (A01-A11) (B10+B11)
                for i in range(2):
                    for l in range(2):
                        contrib_A[1][i][l] = 1 # Block 01
                        contrib_A[3][i][l] = -1 # Block 11
                for i in range(2):
                    for l in range(2):
                        contrib_B[2][i][l] = 1 # Block 10
                        contrib_B[3][i][l] = 1 # Block 11
            
            # Now combine with 2x2 decomposition
            for s in range(7):
                r = k * 7 + s
                # U2[i*2+l, s] is coeff for A_{i,l} in term s
                # V2[l*2+j, s] is coeff for B_{l,j} in term s
                # W2[j*2+i, s] is coeff for C_{j,i} in term s
                
                for i in range(2):
                    for l in range(2):
                        u_coeff = U2[i*2+l, s]
                        if u_coeff != 0:
                            # Sum contributions from blocks to A_{i,l}
                            for b_idx in range(4):
                                c = contrib_A[b_idx][i][l]
                                if c != 0:
                                    # Block offset
                                    offset = b_idx * 4
                                    # A index: i*4 + l? No.
                                    # Block 00 is A[0:2, 0:2].
                                    # Entry (i,l) in block b_idx corresponds to global index:
                                    # row = (b_idx // 2) * 2 + i
                                    # col = (b_idx % 2) * 2 + l
                                    row = (b_idx // 2) * 2 + i
                                    col = (b_idx % 2) * 2 + l
                                    U[row * 4 + col, r] += u_coeff * c
                                    
                        v_coeff = V2[l*2+j, s] if False else 0 # j is not defined here
                        # We need to iterate over j as well for V and W
                        # Actually, the sum is over i, l, j
                        # Let's restructure the loop

    # Corrected Loop
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for k in range(7):
        contrib_A = [[[0.0]*2 for _ in range(2)] for _ in range(4)]
        contrib_B = [[[0.0]*2 for _ in range(2)] for _ in range(4)]
        
        if k == 0:
            for i in range(2):
                for l in range(2):
                    contrib_A[0][i][l] = 1
                    contrib_A[3][i][l] = 1
                    contrib_B[0][i][l] = 1
                    contrib_B[3][i][l] = 1
        elif k == 1:
            for i in range(2):
                for l in range(2):
                    contrib_A[2][i][l] = 1
                    contrib_A[3][i][l] = 1
                    contrib_B[0][i][l] = 1
        elif k == 2:
            for i in range(2):
                for l in range(2):
                    contrib_A[0][i][l] = 1
                    contrib_B[1][i][l] = 1
                    contrib_B[3][i][l] = -1
        elif k == 3:
            for i in range(2):
                for l in range(2):
                    contrib_A[3][i][l] = 1
                    contrib_B[2][i][l] = 1
                    contrib_B[0][i][l] = -1
        elif k == 4:
            for i in range(2):
                for l in range(2):
                    contrib_A[0][i][l] = 1
                    contrib_A[1][i][l] = 1
                    contrib_B[3][i][l] = 1
        elif k == 5:
            for i in range(2):
                for l in range(2):
                    contrib_A[2][i][l] = 1
                    contrib_A[0][i][l] = -1
                    contrib_B[0][i][l] = 1
                    contrib_B[1][i][l] = 1
        elif k == 6:
            for i in range(2):
                for l in range(2):
                    contrib_A[1][i][l] = 1
                    contrib_A[3][i][l] = -1
                    contrib_B[2][i][l] = 1
                    contrib_B[3][i][l] = 1

        for s in range(7):
            r = k * 7 + s
            for i in range(2):
                for l in range(2):
                    for j in range(2):
                        u_coeff = U2[i*2+l, s]
                        v_coeff = V2[l*2+j, s]
                        w_coeff = W2[j*2+i, s]
                        
                        if u_coeff != 0:
                            for b_idx in range(4):
                                c = contrib_A[b_idx][i][l]
                                if c != 0:
                                    row = (b_idx // 2) * 2 + i
                                    col = (b_idx % 2) * 2 + l
                                    U[row * 4 + col, r] += u_coeff * c
                                    
                        if v_coeff != 0:
                            for b_idx in range(4):
                                c = contrib_B[b_idx][l][j]
                                if c != 0:
                                    row = (b_idx // 2) * 2 + l
                                    col = (b_idx % 2) * 2 + j
                                    V[row * 4 + col, r] += v_coeff * c
                                    
                        if w_coeff != 0:
                            # W index is k*n + i. Here n=4.
                            # The "k" in W index is the row of C, "i" is the col of C?
                            # Contract: W[k*n+i, r].
                            # In 2x2, W[j*2+i, s].
                            # So global row = j, global col = i.
                            # But W is for C.
                            # C is 4x4.
                            # The term contributes to C[row, col].
                            # In 2x2, W[j*2+i] corresponds to C[j, i].
                            # So global row = j, global col = i.
                            # But wait, the blocks of C are also 2x2.
                            # M_k is a 2x2 matrix of blocks? No, M_k is a 2x2 matrix.
                            # So the result is a 2x2 matrix.
                            # This 2x2 matrix is one of the blocks of C?
                            # No, M_k is a linear combination of blocks of C.
                            # C = sum of M_k with coefficients.
                            # But in Strassen, M_k are the products, and C is formed by adding/subtracting them.
                            # So the "W" part should map the 2x2 result to the 4x4 C.
                            # But my W2 already maps to the 2x2 C indices.
                            # And the 4x4 C is just the 4 blocks placed together?
                            # No, C is the sum of the 7 M_k matrices (which are 2x2) multiplied by coefficients?
                            # No, the 7 M_k are 2x2 matrices.
                            # C is a 4x4 matrix.
                            # How do 2x2 matrices sum to 4x4?
                            # They don't.
                            # In Strassen for 4x4 using 2x2 blocks:
                            # The 7 products are 2x2 matrices.
                            # The final 4x4 matrix C is formed by combining these 2x2 matrices.
                            # But C has 4 blocks.
                            # So we need 4 sets of 7 products?
                            # No, Strassen for 4x4 has rank 49.
                            # This means 49 scalar multiplications.
                            # My construction: 7 block terms, each rank 7. Total 49.
                            # Each block term $M_k$ is a 2x2 matrix.
                            # But $C$ is 4x4.
                            # The formula is $C_{block} = \sum_k c_k M_k$.
                            # So each entry of $C$ is a linear combination of entries of $M_k$.
                            # So $W$ should map the 2x2 entries of $M_k$ to the 4x4 entries of $C$.
                            # But $M_k$ is a 2x2 matrix.
                            # So $W$ should have 16 rows (for 4x4 C) and 49 cols.
                            # For a given $k$ and $s$, $W$ should add $w_{j,i}^{(2)}$ to the appropriate entry of $C$.
                            # But which entry?
                            # The 2x2 matrix $M_k$ contributes to ALL 4 blocks of $C$?
                            # No, $M_k$ is a 2x2 matrix.
                            # In the standard Strassen, $M_k$ are 2x2 matrices.
                            # $C_{00} = M_1 + M_4 - M_5 + M_7$
                            # Here $C_{00}$ is a 2x2 block.
                            # So $M_1$ (2x2) is added to $C_{00}$ (2x2).
                            # So the entry $(j,i)$ of $M_k$ contributes to the block $C_{block}$ at position $(j,i)$.
                            # Which block?
                            # The formula $C_{00} = \dots$ means the top-left block of C is the sum.
                            # So $M_k$ contributes to the top-left block with coefficient 1.
                            # $C_{01}$ is top-right. $M_3+M_5$.
                            # So $M_k$ contributes to different blocks with different coefficients.
                            # So for each $k$, we need to know which block of C it contributes to and with what coefficient.
                            # And within that block, which 2x2 position.
                            
                            # Let's define block_contrib[k][block_idx] = coeff
                            # And within the block, the 2x2 position (j,i) is given by W2.
                            
                            # k=0 (M1): C00 (+1), C11 (+1)
                            # k=1 (M2): C10 (+1), C11 (-1)
                            # k=2 (M3): C01 (+1), C11 (+1)
                            # k=3 (M4): C00 (+1), C10 (+1)
                            # k=4 (M5): C00 (-1), C01 (+1)
                            # k=5 (M6): C11 (+1)
                            # k=6 (M7): C00 (+1)
                            
                            if k == 0:
                                blocks = [(0, 1), (3, 1)]
                            elif k == 1:
                                blocks = [(2, 1), (3, -1)]
                            elif k == 2:
                                blocks = [(1, 1), (3, 1)]
                            elif k == 3:
                                blocks = [(0, 1), (2, 1)]
                            elif k == 4:
                                blocks = [(0, -1), (1, 1)]
                            elif k == 5:
                                blocks = [(3, 1)]
                            elif k == 6:
                                blocks = [(0, 1)]
                            else:
                                blocks = []
                                
                            for b_idx, coeff in blocks:
                                # b_idx is block index in C (0:00, 1:01, 2:10, 3:11)
                                # The 2x2 matrix M_k has entries (j,i).
                                # These go to C block b_idx at position (j,i).
                                # Global row = (b_idx // 2) * 2 + j
                                # Global col = (b_idx % 2) * 2 + i
                                # W index in contract: k*n + i.
                                # Here "k" is row, "i" is col.
                                # So W[row*4+col, r] += w_coeff * coeff
                                row = (b_idx // 2) * 2 + j
                                col = (b_idx % 2) * 2 + i
                                W[row * 4 + col, r] += w_coeff * coeff

    return {"U": U, "V": V, "W": W}
