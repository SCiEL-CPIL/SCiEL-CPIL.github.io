"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Construct the rank-42 block-Strassen decomposition
    rank_42 = 42
    U_base = np.zeros((n * m, rank_42), dtype=np.float64)
    V_base = np.zeros((m * p, rank_42), dtype=np.float64)
    W_base = np.zeros((p * n, rank_42), dtype=np.float64)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U corresponds to A (2x2 blocks)
    # V corresponds to B (2x2 blocks)
    # W corresponds to C (2x2 blocks)
    
    # Note: The indices in the code below are (row, col) for the 2x2 block.
    # U_coeffs[s] maps (i_loc, j_loc) to coeff
    # V_coeffs[s] maps (j_loc, k_loc) to coeff
    # W_contrib[s] maps (i_loc, k_loc) to coeff
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # M1
        {(1, 0): 1, (1, 1): 1},   # M2
        {(0, 0): 1},              # M3
        {(1, 1): 1},              # M4
        {(0, 0): 1, (0, 1): 1},   # M5
        {(1, 0): 1, (0, 0): -1},  # M6
        {(0, 1): 1, (1, 1): -1}   # M7
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # M1
        {(0, 0): 1},              # M2
        {(0, 1): 1, (1, 1): -1},  # M3
        {(1, 0): 1, (0, 0): -1},  # M4
        {(1, 1): 1},              # M5
        {(0, 0): 1, (0, 1): 1},   # M6
        {(1, 0): 1, (1, 1): 1}    # M7
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},   # M1
        {(1, 0): 1, (1, 1): -1},  # M2
        {(0, 1): 1, (1, 1): 1},   # M3
        {(0, 0): 1, (1, 0): 1},   # M4
        {(0, 0): -1, (0, 1): 1},  # M5
        {(1, 1): 1},              # M6
        {(0, 0): 1}               # M7
    ]
    
    col = 0
    # a: block index for A's columns (0, 1) -> j in 0..1, 2..3
    # b: block index for B's rows (0, 1, 2) -> j in 0..1, 2..3, 4..5
    #     Wait, B is 4x6. Inner dim is 4.
    #     A is 2x4. Inner dim is 4.
    #     The inner dimension is split into two blocks of size 2.
    #     So a=0 corresponds to j=0,1 in A and j=0,1 in B.
    #     a=1 corresponds to j=2,3 in A and j=2,3 in B.
    #     But B has 6 columns. So for each 'a' (inner block), we have 3 output blocks 'b' (k=0..1, 2..3, 4..5).
    
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U_base[i * m + j, col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    # j_loc is the row in B's block (0,1). Global row is a*2 + j_loc
                    # k_loc is the col in B's block (0,1). Global col is b*2 + k_loc
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V_base[j * p + k, col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    # i_loc is row in C (0,1). Global row is i_loc.
                    # k_loc is col in C's block (0,1). Global col is b*2 + k_loc
                    i = i_loc
                    k = b * 2 + k_loc
                    W_base[k * n + i, col] += coeff
                
                col += 1
    
    # Return the decomposition
    return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
# EVOLVE-BLOCK-END
