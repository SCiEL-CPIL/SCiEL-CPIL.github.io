import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block Strassen with Schoolbook inner blocks
    # Split 6x6 into 2x2 grid of 3x3 blocks
    # Rank = 7 * (3*3*3) = 189
    
    # Strassen coefficients for 2x2 blocks of size 3x3
    # t: 0..6
    # A_coeffs[t]: dict {(block_row, block_col): coeff}
    # B_coeffs[t]: dict {(block_row, block_col): coeff}
    # W_coeffs[t]: dict {(block_row, block_col): coeff} for C reconstruction
    
    A_coeffs = []
    B_coeffs = []
    W_coeffs = []
    
    # M1 = (A00 + A11)(B00 + B11)
    A_coeffs.append({(0,0): 1, (1,1): 1})
    B_coeffs.append({(0,0): 1, (1,1): 1})
    W_coeffs.append({(0,0): 1, (1,1): 1})
    
    # M2 = (A10 + A11)B00
    A_coeffs.append({(1,0): 1, (1,1): 1})
    B_coeffs.append({(0,0): 1})
    W_coeffs.append({(1,0): 1, (1,1): -1})
    
    # M3 = A00(B01 - B11)
    A_coeffs.append({(0,0): 1})
    B_coeffs.append({(0,1): 1, (1,1): -1})
    W_coeffs.append({(0,1): 1, (1,1): 1})
    
    # M4 = A11(B10 - B00)
    A_coeffs.append({(1,1): 1})
    B_coeffs.append({(1,0): 1, (0,0): -1})
    W_coeffs.append({(0,0): 1, (1,0): 1})
    
    # M5 = (A00 + A01)B11
    A_coeffs.append({(0,0): 1, (0,1): 1})
    B_coeffs.append({(1,1): 1})
    W_coeffs.append({(0,1): -1, (1,1): -1})
    
    # M6 = (A10 - A00)(B00 + B01)
    A_coeffs.append({(1,0): 1, (0,0): -1})
    B_coeffs.append({(0,0): 1, (0,1): 1})
    W_coeffs.append({(1,1): 1})
    
    # M7 = (A01 - A11)(B10 + B11)
    A_coeffs.append({(0,1): 1, (1,1): -1})
    B_coeffs.append({(1,0): 1, (1,1): 1})
    W_coeffs.append({(0,0): -1, (1,1): 1})
    
    rank = 189
    block_size = 3
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(7):
        a_c = A_coeffs[t]
        b_c = B_coeffs[t]
        w_c = W_coeffs[t]
        
        # Iterate over local indices in the 3x3 blocks
        for r in range(block_size):
            for c in range(block_size):
                for k in range(block_size):
                    # U part: A coefficients
                    for (bi, bj), coeff_a in a_c.items():
                        if coeff_a != 0:
                            u_row = bi * block_size + r
                            u_col = bj * block_size + k
                            U[u_row * 6 + u_col][col] += coeff_a
                            
                    # V part: B coefficients
                    for (bi, bj), coeff_b in b_c.items():
                        if coeff_b != 0:
                            v_row = bi * block_size + k
                            v_col = bj * block_size + c
                            V[v_row * 6 + v_col][col] += coeff_b
                            
                    # W part: C coefficients
                    for (ri, rj), coeff_w in w_c.items():
                        if coeff_w != 0:
                            w_row = ri * block_size + r
                            w_col = rj * block_size + c
                            W[w_row * 6 + w_col][col] += coeff_w
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
