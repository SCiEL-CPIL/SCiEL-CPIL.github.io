import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block Strassen with Schoolbook inner blocks
    # Rank 189
    
    # Strassen coefficients for 2x2 blocks
    # t: 1..7
    # A_coeffs[t][alpha][beta]
    # B_coeffs[t][beta][gamma]
    # W_coeffs[t][alpha][gamma] (for reconstruction)
    
    A_coeffs = {}
    B_coeffs = {}
    W_coeffs = {}
    
    # M1 = (A00 + A11)(B00 + B11)
    A_coeffs[1] = {(0,0): 1, (1,1): 1}
    B_coeffs[1] = {(0,0): 1, (1,1): 1}
    W_coeffs[1] = {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}
    
    # M2 = (A10 + A11)B00
    A_coeffs[2] = {(1,0): 1, (1,1): 1}
    B_coeffs[2] = {(0,0): 1}
    W_coeffs[2] = {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}
    
    # M3 = A00(B01 - B11)
    A_coeffs[3] = {(0,0): 1}
    B_coeffs[3] = {(0,1): 1, (1,1): -1}
    W_coeffs[3] = {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}
    
    # M4 = A11(B10 - B00)
    A_coeffs[4] = {(1,1): 1}
    B_coeffs[4] = {(1,0): 1, (0,0): -1}
    W_coeffs[4] = {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}
    
    # M5 = (A00 + A01)B11
    A_coeffs[5] = {(0,0): 1, (0,1): 1}
    B_coeffs[5] = {(1,1): 1}
    W_coeffs[5] = {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): -1}
    
    # M6 = (A10 - A00)(B00 + B01)
    A_coeffs[6] = {(1,0): 1, (0,0): -1}
    B_coeffs[6] = {(0,0): 1, (0,1): 1}
    W_coeffs[6] = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}
    
    # M7 = (A01 - A11)(B10 + B11)
    A_coeffs[7] = {(0,1): 1, (1,1): -1}
    B_coeffs[7] = {(1,0): 1, (1,1): 1}
    W_coeffs[7] = {(0,0): -1, (0,1): 0, (1,0): 0, (1,1): 1}
    
    # Rank is 7 * 3^3 = 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(1, 8):
        for a in range(3): # row index in 3x3 block
            for x in range(3): # inner index
                for b in range(3): # col index in 3x3 block
                    # U part
                    for alpha in range(2):
                        for beta in range(2):
                            ca = A_coeffs[t].get((alpha, beta), 0)
                            if ca != 0:
                                u_row = (2 * alpha + a) * 6 + (2 * beta + x)
                                U[u_row][col] += ca
                                
                    # V part
                    for beta in range(2):
                        for gamma in range(2):
                            cb = B_coeffs[t].get((beta, gamma), 0)
                            if cb != 0:
                                v_row = (2 * beta + x) * 6 + (2 * gamma + b)
                                V[v_row][col] += cb
                                
                    # W part
                    for alpha in range(2):
                        for gamma in range(2):
                            cw = W_coeffs[t].get((alpha, gamma), 0)
                            if cw != 0:
                                w_row = (2 * gamma + b) * 6 + (2 * alpha + a)
                                W[w_row][col] += cw
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
