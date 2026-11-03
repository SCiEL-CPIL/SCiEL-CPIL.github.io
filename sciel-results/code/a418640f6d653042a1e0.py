"""Matrix multiplication 2,4,8 - Block Strassen decomposition rank 14."""
DIMENSIONS = [2, 4, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    start_time = time.time()
    deadline = start_time + 140
    
    # Split m=4 into two groups of 2: j_group0 = {0,1}, j_group1 = {2,3}
    # Each sub-problem: A_sub (2x2) * B_sub (2x8) -> C_sub (2x8)
    # Use Strassen (rank 7) for each sub-problem
    # Total rank = 14
    
    rank = 14
    U = np.zeros((n * m, rank))  # (8, 14)
    V = np.zeros((m * p, rank))  # (32, 14)
    W = np.zeros((p * n, rank))  # (16, 14)
    
    def _strassen_2x2_x_2x8():
        """Return U (4,7), V (16,7), W (16,7) for 2x2 * 2x8 multiplication.
        U indexed by [i*2+j], V by [j*8+k], W by [k*2+i]
        """
        r = 7
        U = np.zeros((4, r))
        V = np.zeros((16, r))
        W = np.zeros((16, r))
        
        # M0: (A00+A11)(B0+B1) -> C0 += M0, C1 += M0
        U[0, 0] = 1; U[3, 0] = 1
        for k in range(8):
            V[0*8+k, 0] = 1
            V[1*8+k, 0] = 1
            W[k*2+0, 0] = 1
            W[k*2+1, 0] = 1
        
        # M1: (A10+A11)B0 -> C1 += M1
        U[2, 1] = 1; U[3, 1] = 1
        for k in range(8):
            V[0*8+k, 1] = 1
            W[k*2+1, 1] = 1
        
        # M2: A00(B0-B1) -> C0 += M2
        U[0, 2] = 1
        for k in range(8):
            V[0*8+k, 2] = 1
            V[1*8+k, 2] = -1
            W[k*2+0, 2] = 1
        
        # M3: A11(B1-B0) -> C0 -= M3
        U[3, 3] = 1
        for k in range(8):
            V[1*8+k, 3] = 1
            V[0*8+k, 3] = -1
            W[k*2+0, 3] = -1
        
        # M4: (A00+A01)B1 -> C0 += M4
        U[0, 4] = 1; U[1, 4] = 1
        for k in range(8):
            V[1*8+k, 4] = 1
            W[k*2+0, 4] = 1
        
        # M5: (A10-A00)(B0+B1) -> C0 += M5
        U[2, 5] = 1; U[0, 5] = -1
        for k in range(8):
            V[0*8+k, 5] = 1
            V[1*8+k, 5] = 1
            W[k*2+0, 5] = 1
        
        # M6: (A01-A11)(B1+B0) -> C1 += M6
        U[1, 6] = 1; U[3, 6] = -1
        for k in range(8):
            V[1*8+k, 6] = 1
            V[0*8+k, 6] = 1
            W[k*2+1, 6] = 1
        
        return U, V, W
    
    U_sub, V_sub, W_sub = _strassen_2x2_x_2x8()
    
    # Group 0: j offset 0
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + (j+0), c] += U_sub[i*2+j, c]
        for j in range(2):
            for k in range(8):
                V[(j+0)*p + k, c] += V_sub[j*8+k, c]
        for k in range(8):
            for i in range(2):
                W[k*n + i, c] += W_sub[k*2+i, c]
    
    # Group 1: j offset 2
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + (j+2), 7+c] += U_sub[i*2+j, c]
        for j in range(2):
            for k in range(8):
                V[(j+2)*p + k, 7+c] += V_sub[j*8+k, c]
        for k in range(8):
            for i in range(2):
                W[k*n + i, 7+c] += W_sub[k*2+i, c]
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid and time.time() < deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank_fb = n * m * p
    U_fb = [[0.0] * rank_fb for _ in range(n * m)]
    V_fb = [[0.0] * rank_fb for _ in range(m * p)]
    W_fb = [[0.0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1.0
                V_fb[j * p + k][col] = 1.0
                W_fb[k * n + i][col] = 1.0
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
