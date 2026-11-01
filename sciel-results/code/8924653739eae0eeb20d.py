"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    num_blocks = p // 2  # 5 blocks of 2 columns each
    
    rank = 7 * num_blocks  # 35
    
    # Laderman's 7-term decomposition for 2x2 matrix multiplication
    # Terms defined with local indices for a 2x2 block
    # U_dict: {(i,j): coeff} for A[i][j]
    # V_dict: {(j,l): coeff} for B[j][l] (l is local col 0 or 1)
    # W_dict: {(k,i): coeff} for C[i][k] (k is local col, i is row)
    
    terms_local = [
        # P0 = (A00 + A11)(B00 + B11)
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1}),
        # P1 = (A10 + A11)B00
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(0,0): 1, (0,1): -1}),
        # P2 = A00(B10 - B11)
        ({(0,0): 1},
         {(1,0): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        # P3 = A11(B01 - B00)
        ({(1,1): 1},
         {(0,1): 1, (0,0): -1},
         {(0,0): 1}),
        # P4 = (A00 + A01)B11
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,1): 1, (0,0): -1}),
        # P5 = (A10 - A00)(B00 + B01)
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(0,0): 1}),
        # P6 = (A01 + A11)(B10 + B11)
        ({(0,1): 1, (1,1): 1},
         {(1,0): 1, (1,1): 1},
         {(0,0): -1, (1,1): 1}),
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        for t in range(7):
            c = b * 7 + t
            u_dict, v_dict, w_dict = terms_local[t]
            
            for (i, j), coeff in u_dict.items():
                U[i * m + j][c] = coeff
            
            for (j, l), coeff in v_dict.items():
                global_k = 2 * b + l
                V[j * p + global_k][c] = coeff
            
            for (k, i), coeff in w_dict.items():
                global_k = 2 * b + k
                W[global_k * n + i][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
