import time
import sys

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # 150 seconds wall-clock limit
    
    n, m, p = DIMENSIONS
    
    # We use a rank 21 decomposition based on Strassen's algorithm for 2x2 matrices.
    # B is 2x6, so we split it into three 2x2 blocks: B1, B2, B3.
    # C = A * [B1 B2 B3] = [A*B1, A*B2, A*B3].
    # Each A*Br requires rank 7 (Strassen). Total rank = 7 * 3 = 21.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    # Strassen coefficients for A (2x2): U index = i*m + j = i*2 + j
    # Indices: 0=A00, 1=A01, 2=A10, 3=A11
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # Strassen coefficients for B (2x2 per block): V index = j*p + k
    # For block r (r=0,1,2), local columns are 2r and 2r+1
    # V_coeffs[k] = list of (j_local, k_local, coeff)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    # W mapping: for each Strassen product k, which C entries it affects
    # W index = global_k * n + i
    # W_map[k] = list of (i, j_local, coeff)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # M1 -> C00(+1), C11(+1)
        [(1, 0, 1), (1, 1, -1)],  # M2 -> C10(+1), C11(-1)
        [(0, 1, 1), (1, 1, 1)],   # M3 -> C01(+1), C11(+1)
        [(0, 0, 1), (1, 0, 1)],   # M4 -> C00(+1), C10(+1)
        [(0, 0, -1), (0, 1, 1)],  # M5 -> C00(-1), C01(+1)
        [(1, 1, 1)],              # M6 -> C11(+1)
        [(0, 0, 1)]               # M7 -> C00(+1)
    ]
    
    for r in range(3):
        # Check deadline
        if time.time() > deadline - 5:
            break
            
        for k in range(7):
            col = r * 7 + k
            
            # Fill U
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
                
            # Fill V
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                v_idx = j_local * p + global_k
                V[v_idx][col] = val
                
            # Fill W
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                w_idx = global_k * n + i
                W[w_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
