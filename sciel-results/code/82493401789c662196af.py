"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    
    # Rank is 7 blocks * 7 Strassen terms = 49
    rank = 49
    
    U = np.zeros((size_a, rank), dtype=np.float64)
    V = np.zeros((size_b, rank), dtype=np.float64)
    W = np.zeros((size_c, rank), dtype=np.float64)
    
    # Strassen's 7 products for 2x2 matrices (0-indexed)
    # M1 = (A00 - A11)(B01 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B00 - B10)
    # M4 = A11(B11 - B01)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the coefficients for each term t (0 to 6)
    # U_coeffs[t][a_idx] = coeff
    # V_coeffs[t][b_idx] = coeff
    # W_coeffs[t][c_idx] = coeff
    
    # We will iterate over blocks b (0 to 6) and terms t (0 to 6)
    
    for b in range(7):
        for t in range(7):
            r = b * 7 + t
            
            # Helper to add to U, V, W
            # U index for A[i,j] is 2*i + j
            # V index for B[j,k] is 2*j + (2*b + k)
            # W index for C[i,k] is (2*b + k)*2 + i = 4*b + 2*k + i
            
            if t == 0: # M1
                # A: 00 (+1), 11 (-1)
                U[0, r] += 1
                U[3, r] += -1
                # B: 01 (+1), 11 (+1)
                V[2*0 + 2*b + 1, r] += 1
                V[2*1 + 2*b + 1, r] += 1
                # C: 00 (+1), 11 (+1)
                W[4*b + 2*0 + 0, r] += 1
                W[4*b + 2*1 + 1, r] += 1
                
            elif t == 1: # M2
                # A: 10 (+1), 11 (+1)
                U[2, r] += 1
                U[3, r] += 1
                # B: 00 (+1)
                V[2*0 + 2*b + 0, r] += 1
                # C: 10 (+1), 11 (-1)
                W[4*b + 2*0 + 1, r] += 1
                W[4*b + 2*1 + 1, r] += -1
                
            elif t == 2: # M3
                # A: 00 (+1)
                U[0, r] += 1
                # B: 00 (+1), 10 (-1)
                V[2*0 + 2*b + 0, r] += 1
                V[2*1 + 2*b + 0, r] += -1
                # C: 01 (+1), 11 (+1)
                W[4*b + 2*0 + 1, r] += 1
                W[4*b + 2*1 + 1, r] += 1
                
            elif t == 3: # M4
                # A: 11 (+1)
                U[3, r] += 1
                # B: 11 (+1), 01 (-1)
                V[2*1 + 2*b + 1, r] += 1
                V[2*0 + 2*b + 1, r] += -1
                # C: 00 (+1), 10 (+1)
                W[4*b + 2*0 + 0, r] += 1
                W[4*b + 2*0 + 1, r] += 1
                
            elif t == 4: # M5
                # A: 00 (+1), 01 (+1)
                U[0, r] += 1
                U[1, r] += 1
                # B: 11 (+1)
                V[2*1 + 2*b + 1, r] += 1
                # C: 00 (-1), 01 (+1)
                W[4*b + 2*0 + 0, r] += -1
                W[4*b + 2*0 + 1, r] += 1
                
            elif t == 5: # M6
                # A: 10 (+1), 00 (-1)
                U[2, r] += 1
                U[0, r] += -1
                # B: 00 (+1), 01 (+1)
                V[2*0 + 2*b + 0, r] += 1
                V[2*0 + 2*b + 1, r] += 1
                # C: 11 (+1)
                W[4*b + 2*1 + 1, r] += 1
                
            elif t == 6: # M7
                # A: 01 (+1), 11 (-1)
                U[1, r] += 1
                U[3, r] += -1
                # B: 10 (+1), 11 (+1)
                V[2*1 + 2*b + 0, r] += 1
                V[2*1 + 2*b + 1, r] += 1
                # C: 00 (+1)
                W[4*b + 2*0 + 0, r] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
