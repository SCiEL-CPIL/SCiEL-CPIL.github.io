"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Use 2x2 block decomposition with schoolbook for each 2x2x2x2 block
    # A (4x4) -> 2x2 blocks A[i_b, j_b] for i_b, j_b in {0,1}
    # B (4x8) -> 2x2 blocks B[j_b, k_b] for j_b in {0,1}, k_b in {0,1,2,3}
    # C (4x8) -> 2x2 blocks C[i_b, k_b] for i_b in {0,1}, k_b in {0,1,2,3}
    #
    # C[i_b, k_b] = A[i_b, 0] * B[0, k_b] + A[i_b, 1] * B[1, k_b]
    # Each 2x2 * 2x2 product uses 4 schoolbook terms
    # Total: 2 * 4 * 2 * 4 = 64
    
    rank = 64
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 64
    V = [[0] * rank for _ in range(m * p)]  # 32 x 64
    W = [[0] * rank for _ in range(p * n)]  # 32 x 64
    
    col = 0
    
    # For each (i_b, j_b, k_b) and each (i_l, k_l) in the 2x2 output block
    for i_b in range(2):
        for k_b in range(4):
            for j_b in range(2):
                for i_l in range(2):
                    for k_l in range(2):
                        # C[i_b*2+i_l, k_b*2+k_l] += A[i_b*2+i_l, j_b*2] * B[j_b*2, k_b*2+k_l]
                        # This is one schoolbook term for the (i_l, k_l) entry of the 2x2 product
                        
                        u_idx = (i_b * 2 + i_l) * m + (j_b * 2)
                        v_idx = (j_b * 2) * p + (k_b * 2 + k_l)
                        w_idx = (k_b * 2 + k_l) * n + (i_b * 2 + i_l)
                        
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                        col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-9:
        # Fallback to full schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
