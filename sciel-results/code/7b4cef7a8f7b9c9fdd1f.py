"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen on 2x2 blocks
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # b: block of A (0, 1) -> columns 0,1 and 2,3
    # c: block of B (0, 1, 2, 3) -> cols 0,1; 2,3; 4,5; 6,7
    for b in range(2):
        for c in range(4):
            # Local indices for A (2x2)
            # a: i=0, j=2b
            # b_val: i=0, j=2b+1
            # c_val: i=1, j=2b
            # d: i=1, j=2b+1
            idx_a = 0 * m + 2 * b
            idx_b = 0 * m + 2 * b + 1
            idx_c = 1 * m + 2 * b
            idx_d = 1 * m + 2 * b + 1
            
            # Local indices for B (2x2)
            # e: j=2b, k=2c
            # f: j=2b, k=2c+1
            # g: j=2b+1, k=2c
            # h: j=2b+1, k=2c+1
            idx_e = 2 * b * p + 2 * c
            idx_f = 2 * b * p + 2 * c + 1
            idx_g = (2 * b + 1) * p + 2 * c
            idx_h = (2 * b + 1) * p + 2 * c + 1
            
            # Local indices for C (2x2 output)
            # C11: i=0, k=2c
            # C12: i=0, k=2c+1
            # C21: i=1, k=2c
            # C22: i=1, k=2c+1
            idx_c11 = 2 * c * n + 0
            idx_c12 = (2 * c + 1) * n + 0
            idx_c21 = 2 * c * n + 1
            idx_c22 = (2 * c + 1) * n + 1
            
            # Strassen terms
            # M1 = (a+d)(e+h)
            U[idx_a][col] += 1
            U[idx_d][col] += 1
            V[idx_e][col] += 1
            V[idx_h][col] += 1
            W[idx_c11][col] += 1
            W[idx_c22][col] += 1
            col += 1
            
            # M2 = (c+d)e
            U[idx_c][col] += 1
            U[idx_d][col] += 1
            V[idx_e][col] += 1
            W[idx_c21][col] += 1
            W[idx_c22][col] -= 1
            col += 1
            
            # M3 = a(f-h)
            U[idx_a][col] += 1
            V[idx_f][col] += 1
            V[idx_h][col] -= 1
            W[idx_c12][col] += 1
            W[idx_c22][col] += 1
            col += 1
            
            # M4 = d(g-e)
            U[idx_d][col] += 1
            V[idx_g][col] += 1
            V[idx_e][col] -= 1
            W[idx_c11][col] += 1
            W[idx_c21][col] += 1
            col += 1
            
            # M5 = (a+b)h
            U[idx_a][col] += 1
            U[idx_b][col] += 1
            V[idx_h][col] += 1
            W[idx_c11][col] -= 1
            W[idx_c12][col] += 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[idx_c][col] += 1
            U[idx_a][col] -= 1
            V[idx_e][col] += 1
            V[idx_f][col] += 1
            W[idx_c22][col] += 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[idx_b][col] += 1
            U[idx_d][col] -= 1
            V[idx_g][col] += 1
            V[idx_h][col] += 1
            W[idx_c11][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
