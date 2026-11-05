import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = 100  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    # Rank is 7 * 27 = 189 (Strassen 2x2 blocks with schoolbook 3x3)
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen products for 2x2 blocks of 3x3 matrices
    strassen_products = [
        # M1 = (A11 + A22) * (B11 + B22)
        {'X': [((0, 0), 1), ((1, 1), 1)], 'Y': [((0, 0), 1), ((1, 1), 1)], 'C': [((0, 0), 1), ((1, 1), 1)]},
        # M2 = (A21 + A22) * B11
        {'X': [((1, 0), 1), ((1, 1), 1)], 'Y': [((0, 0), 1)], 'C': [((1, 0), 1), ((1, 1), -1)]},
        # M3 = A11 * (B12 - B22)
        {'X': [((0, 0), 1)], 'Y': [((0, 1), 1), ((1, 1), -1)], 'C': [((0, 1), 1), ((1, 1), 1)]},
        # M4 = A22 * (B21 - B11)
        {'X': [((1, 1), 1)], 'Y': [((1, 0), 1), ((0, 0), -1)], 'C': [((0, 0), 1), ((1, 0), 1)]},
        # M5 = (A11 + A12) * B22
        {'X': [((0, 0), 1), ((0, 1), 1)], 'Y': [((1, 1), 1)], 'C': [((0, 0), -1), ((0, 1), 1)]},
        # M6 = (A21 - A11) * (B11 + B12)
        {'X': [((1, 0), 1), ((0, 0), -1)], 'Y': [((0, 0), 1), ((0, 1), 1)], 'C': [((1, 1), 1)]},
        # M7 = (A12 - A22) * (B21 + B22)
        {'X': [((0, 1), 1), ((1, 1), -1)], 'Y': [((1, 0), 1), ((1, 1), 1)], 'C': [((0, 0), 1)]}
    ]
    
    col_idx = 0
    for prod in strassen_products:
        X_blocks = prod['X']
        Y_blocks = prod['Y']
        C_contribs = prod['C']
        
        if time.time() > deadline:
            break
        
        for i_prime in range(3):
            for j_prime in range(3):
                for k_prime in range(3):
                    if time.time() > deadline:
                        break
                    
                    for (I, J), coeff_A in X_blocks:
                        i_global = I * 3 + i_prime
                        j_global = J * 3 + j_prime
                        u_row = i_global * 6 + j_global
                        U[u_row, col_idx] += coeff_A
                    
                    for (J, K), coeff_B in Y_blocks:
                        j_global = J * 3 + j_prime
                        k_global = K * 3 + k_prime
                        v_row = j_global * 6 + k_global
                        V[v_row, col_idx] += coeff_B
                        
                    for (I, K), coeff_C in C_contribs:
                        i_global = I * 3 + i_prime
                        k_global = K * 3 + k_prime
                        w_row = k_global * 6 + i_global
                        W[w_row, col_idx] += coeff_C
                        
                    col_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
