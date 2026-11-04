import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-55 decomposition for 2x3 * 3x10 matrix multiplication.
    
    Structure:
    A (2x3) = [A_left (2x2) | A_right (2x1)]
    B (3x10) = [B_top (2x10); B_bottom (1x10)]
    
    C = A_left @ B_top + A_right @ B_bottom
    
    1. A_left @ B_top (2x2 x 2x10):
       Split B_top into 5 blocks of 2x2 columns.
       Each 2x2 x 2x2 block uses Strassen's algorithm (rank 7).
       Total rank: 5 * 7 = 35.
       
    2. A_right @ B_bottom (2x1 x 1x10):
       Schoolbook multiplication.
       Total rank: 2 * 1 * 10 = 20.
       
    Total Rank: 55.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    # Indexing:
    # U row: i*m + j, i in [0,1], j in [0,1,2]
    # V row: j*p + k, j in [0,1,2], k in [0..9]
    # W row: k*n + i, k in [0..9], i in [0,1]
    
    r = 0  # current rank index
    
    # --- Part 1: A_left (2x2) @ B_top (2x10) via Strassen on 5 2x2 blocks ---
    # A_left entries: A[0,0], A[0,1], A[1,0], A[1,1]
    #   U indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    # B_top entries: B[0,k], B[1,k] for k=0..9
    #   V indices: 0*10+k = k, 1*10+k = 10+k
    
    # Strassen's 7 products for 2x2 x 2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5],
    #      [m2+m4,        m1-m2+m3+m6]]
    
    # For block b (b=0..4), columns 2b and 2b+1 of B_top:
    # A_left = [[A00, A01], [A10, A11]]
    # B_block = [[B0_{2b}, B0_{2b+1}], [B1_{2b}, B1_{2b+1}]]
    
    # Strassen coefficients for U (A_left part):
    # m1: A00 + A11  -> U[0] += 1, U[4] += 1
    # m2: A10 + A11  -> U[3] += 1, U[4] += 1
    # m3: A00        -> U[0] += 1
    # m4: A11        -> U[4] += 1
    # m5: A00 + A01  -> U[0] += 1, U[1] += 1
    # m6: A10 - A00  -> U[3] += 1, U[0] += -1
    # m7: A01 - A11  -> U[1] += 1, U[4] += -1
    
    # Strassen coefficients for V (B_block part):
    # m1: B0_{2b} + B1_{2b+1}  -> V[2b] += 1, V[10+2b+1] += 1
    # m2: B0_{2b}              -> V[2b] += 1
    # m3: B0_{2b+1} - B1_{2b+1} -> V[2b+1] += 1, V[10+2b+1] += -1
    # m4: B1_{2b} - B0_{2b}    -> V[10+2b] += 1, V[2b] += -1
    # m5: B1_{2b+1}            -> V[10+2b+1] += 1
    # m6: B0_{2b} + B0_{2b+1}  -> V[2b] += 1, V[2b+1] += 1
    # m7: B1_{2b} + B1_{2b+1}  -> V[10+2b] += 1, V[10+2b+1] += 1
    
    # W contributions:
    # C[0, 2b]   = m1 + m4 - m5 + m7
    # C[0, 2b+1] = m3 + m5
    # C[1, 2b]   = m2 + m4
    # C[1, 2b+1] = m1 - m2 + m3 + m6
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # m1
        U[0][r] += 1
        U[4][r] += 1
        V[k0][r] += 1
        V[10 + k1][r] += 1
        W[k0 * 2 + 0][r] += 1
        W[k1 * 2 + 1][r] += 1
        r += 1
        
        # m2
        U[3][r] += 1
        U[4][r] += 1
        V[k0][r] += 1
        W[k0 * 2 + 1][r] += 1
        W[k1 * 2 + 1][r] -= 1
        r += 1
        
        # m3
        U[0][r] += 1
        V[k1][r] += 1
        V[10 + k1][r] -= 1
        W[k1 * 2 + 0][r] += 1
        W[k1 * 2 + 1][r] += 1
        r += 1
        
        # m4
        U[4][r] += 1
        V[10 + k0][r] += 1
        V[k0][r] -= 1
        W[k0 * 2 + 0][r] += 1
        W[k0 * 2 + 1][r] += 1
        r += 1
        
        # m5
        U[0][r] += 1
        U[1][r] += 1
        V[10 + k1][r] += 1
        W[k0 * 2 + 0][r] -= 1
        W[k1 * 2 + 0][r] += 1
        r += 1
        
        # m6
        U[3][r] += 1
        U[0][r] -= 1
        V[k0][r] += 1
        V[k1][r] += 1
        W[k1 * 2 + 1][r] += 1
        r += 1
        
        # m7
        U[1][r] += 1
        U[4][r] -= 1
        V[10 + k0][r] += 1
        V[10 + k1][r] += 1
        W[k0 * 2 + 0][r] += 1
        r += 1
        
    # --- Part 2: A_right (2x1) @ B_bottom (1x10) ---
    # A_right entries: A[0,2], A[1,2]
    #   U indices: 0*3+2=2, 1*3+2=5
    # B_bottom entries: B[2,k] for k=0..9
    #   V indices: 2*10+k = 20+k
    
    for k in range(10):
        # Term for C[0, k]: A[0,2] * B[2, k]
        U[2][r] += 1
        V[20 + k][r] += 1
        W[k * 2 + 0][r] += 1
        r += 1
        
        # Term for C[1, k]: A[1,2] * B[2, k]
        U[5][r] += 1
        V[20 + k][r] += 1
        W[k * 2 + 1][r] += 1
        r += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
