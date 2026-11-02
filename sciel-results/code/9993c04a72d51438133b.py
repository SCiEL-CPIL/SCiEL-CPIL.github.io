import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a 2x2 Strassen decomposition on 3x3 blocks.
    # Rank = 7 * (3*3*3) = 189
    rank = 189
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block indices for 3x3 blocks
    # A is 6x6, split into 4 blocks of 3x3
    # A_TL: rows 0-2, cols 0-2
    # A_TR: rows 0-2, cols 3-5
    # A_BL: rows 3-5, cols 0-2
    # A_BR: rows 3-5, cols 3-5
    
    # Strassen terms
    # M1 = (A_TL + A_BR)(B_TL + B_BR)
    # M2 = (A_BL + A_BR)B_TL
    # M3 = A_TL(B_TR - B_BR)
    # M4 = A_BR(B_BL - B_TL)
    # M5 = (A_TL + A_TR)B_BR
    # M6 = (A_BL - A_TL)(B_TL + B_TR)
    # M7 = (A_TR - A_BR)(B_BL + B_BR)
    
    # For each M, we have a 3x3 multiplication.
    # Each 3x3 multiplication has rank 27.
    # Total 7 * 27 = 189 terms.
    
    col = 0
    for m_idx in range(7):
        # Define the linear combinations for A and B for this M
        # A_comb: list of (block, coeff)
        # B_comb: list of (block, coeff)
        if m_idx == 0:
            A_comb = [('TL', 1), ('BR', 1)]
            B_comb = [('TL', 1), ('BR', 1)]
            W_comb = [('TL', 1), ('TR', 0), ('BL', 0), ('BR', 1)] # C_TL += M1, C_BR += M1
        elif m_idx == 1:
            A_comb = [('BL', 1), ('BR', 1)]
            B_comb = [('TL', 1)]
            W_comb = [('BL', 1), ('BR', 1)] # C_BL += M2, C_BR += M2
        elif m_idx == 2:
            A_comb = [('TL', 1)]
            B_comb = [('TR', 1), ('BR', -1)]
            W_comb = [('TR', 1), ('BR', 1)] # C_TR += M3, C_BR += M3
        elif m_idx == 3:
            A_comb = [('BR', 1)]
            B_comb = [('BL', 1), ('TL', -1)]
            W_comb = [('TL', 1), ('BL', 1)] # C_TL += M4, C_BL += M4
        elif m_idx == 4:
            A_comb = [('TL', 1), ('TR', 1)]
            B_comb = [('BR', 1)]
            W_comb = [('TR', -1), ('BR', 1)] # C_TR -= M5, C_BR += M5
        elif m_idx == 5:
            A_comb = [('BL', 1), ('TL', -1)]
            B_comb = [('TL', 1), ('TR', 1)]
            W_comb = [('BR', 1)] # C_BR += M6
        elif m_idx == 6:
            A_comb = [('TR', 1), ('BR', -1)]
            B_comb = [('BL', 1), ('BR', 1)]
            W_comb = [('TL', 1), ('TR', 1)] # C_TL += M7, C_TR += M7
        else:
            continue
            
        # For each of the 27 terms in the 3x3 product
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    # Term: (sum a_x A_x_ik) * (sum b_y B_y_kj)
                    # This contributes to C_ij
                    # We need to add this to U, V, W
                    
                    # U vector: 1s at A positions
                    for a_block, a_coeff in A_comb:
                        # Map block to row/col offset
                        if a_block == 'TL':
                            r_off, c_off = 0, 0
                        elif a_block == 'TR':
                            r_off, c_off = 0, 3
                        elif a_block == 'BL':
                            r_off, c_off = 3, 0
                        elif a_block == 'BR':
                            r_off, c_off = 3, 3
                        else:
                            continue
                        row = (r_off + i) * 6 + (c_off + k)
                        U[row, col] += a_coeff
                        
                    # V vector: 1s at B positions
                    for b_block, b_coeff in B_comb:
                        if b_block == 'TL':
                            r_off, c_off = 0, 0
                        elif b_block == 'TR':
                            r_off, c_off = 0, 3
                        elif b_block == 'BL':
                            r_off, c_off = 3, 0
                        elif b_block == 'BR':
                            r_off, c_off = 3, 3
                        else:
                            continue
                        row = (r_off + k) * 6 + (c_off + j)
                        V[row, col] += b_coeff
                        
                    # W vector: 1s at C positions
                    for w_block, w_coeff in W_comb:
                        if w_block == 'TL':
                            r_off, c_off = 0, 0
                        elif w_block == 'TR':
                            r_off, c_off = 0, 3
                        elif w_block == 'BL':
                            r_off, c_off = 3, 0
                        elif w_block == 'BR':
                            r_off, c_off = 3, 3
                        else:
                            continue
                        # W index is k*n + i ? No, W[k*n+i, r] is the convention.
                        # The task says: "output factor follows the official trace convention W[k*n+i,r]"
                        # Wait, the seed code uses W[k*n+i][column] = 1.
                        # And U[i*m+j], V[j*p+k].
                        # So C[i,k] = sum U[i*m+j] V[j*p+k] W[k*n+i].
                        # Here i is row of A, k is col of A (row of B), j is col of B.
                        # So C[i,k] gets contribution from U[i,j] V[j,k] W[k,i].
                        # In our loop, i is row of block, k is col of block (row of B block), j is col of B block.
                        # So C_block[i,j] gets contribution.
                        # The global row of C is r_off + i.
                        # The global col of C is c_off + j.
                        # So W index is (c_off + j) * 6 + (r_off + i).
                        row = (c_off + j) * 6 + (r_off + i)
                        W[row, col] += w_coeff
                        
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
