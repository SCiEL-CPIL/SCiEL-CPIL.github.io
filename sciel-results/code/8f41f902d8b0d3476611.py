import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * 27 = 189
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen products definition
    # Each product is a tuple (X_blocks, Y_blocks, C_contributions)
    # X_blocks: list of (block_index, coeff)
    # Y_blocks: list of (block_index, coeff)
    # C_contributions: list of (I, K, coeff)
    
    # Block indices: (row_block, col_block)
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    strassen_products = [
        # M1
        {
            'X': [((0, 0), 1), ((1, 1), 1)],
            'Y': [((0, 0), 1), ((1, 1), 1)],
            'C': [((0, 0), 1), ((1, 1), 1)]
        },
        # M2
        {
            'X': [((1, 0), 1), ((1, 1), 1)],
            'Y': [((0, 0), 1)],
            'C': [((1, 0), 1), ((1, 1), -1)]
        },
        # M3
        {
            'X': [((0, 0), 1)],
            'Y': [((0, 1), 1), ((1, 1), -1)],
            'C': [((0, 1), 1), ((1, 1), 1)]
        },
        # M4
        {
            'X': [((1, 1), 1)],
            'Y': [((1, 0), 1), ((0, 0), -1)],
            'C': [((0, 0), 1), ((1, 0), 1)]
        },
        # M5
        {
            'X': [((0, 0), 1), ((0, 1), 1)],
            'Y': [((1, 1), 1)],
            'C': [((0, 0), -1), ((0, 1), 1)]
        },
        # M6
        {
            'X': [((1, 0), 1), ((0, 0), -1)],
            'Y': [((0, 0), 1), ((0, 1), 1)],
            'C': [((1, 1), 1)]
        },
        # M7
        {
            'X': [((0, 1), 1), ((1, 1), -1)],
            'Y': [((1, 0), 1), ((1, 1), 1)],
            'C': [((0, 0), 1)]
        }
    ]
    
    col_idx = 0
    for prod in strassen_products:
        X_blocks = prod['X']
        Y_blocks = prod['Y']
        C_contribs = prod['C']
        
        # Iterate over local indices for 3x3 multiplication
        for i_prime in range(3):
            for j_prime in range(3):
                for k_prime in range(3):
                    # Determine U coefficients
                    # X[i', j'] is sum of A[I, J][i', j'] * coeff
                    # Global A index: (I*3 + i', J*3 + j')
                    # U row: i*6 + j
                    for (I, J), coeff_A in X_blocks:
                        i_global = I * 3 + i_prime
                        j_global = J * 3 + j_prime
                        u_row = i_global * 6 + j_global
                        U[u_row, col_idx] += coeff_A
                    
                    # Determine V coefficients
                    # Y[j', k'] is sum of B[J, K][j', k'] * coeff
                    # Global B index: (J*3 + j', K*3 + k')
                    # V row: j*6 + k
                    for (J, K), coeff_B in Y_blocks:
                        j_global = J * 3 + j_prime
                        k_global = K * 3 + k_prime
                        v_row = j_global * 6 + k_global
                        V[v_row, col_idx] += coeff_B
                        
                    # Determine W coefficients
                    # C[I, K] gets contribution from this product
                    # Local index in C block: (i', k')
                    # Global C index: (I*3 + i', K*3 + k')
                    # W row: k*6 + i
                    for (I, K), coeff_C in C_contribs:
                        i_global = I * 3 + i_prime
                        k_global = K * 3 + k_prime
                        w_row = k_global * 6 + i_global
                        W[w_row, col_idx] += coeff_C
                        
                    col_idx += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
