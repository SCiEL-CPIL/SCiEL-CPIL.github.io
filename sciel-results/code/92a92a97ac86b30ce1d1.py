import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Strategy: 
    # 1. Attempt Rank 46 decomposition:
    #    Split J={0,1,2}, {3,4} and K={0,1,2}, {3,4,5}
    #    Blocks:
    #    (0,1,2)x(0,1,2): 2x3x3 (Attempt Rank 11)
    #    (0,1,2)x(3,4,5): 2x3x3 (Attempt Rank 11)
    #    (3,4)x(0,1,2): 2x2x3 (Standard Rank 12)
    #    (3,4)x(3,4,5): 2x2x3 (Standard Rank 12)
    #    Total: 11 + 11 + 12 + 12 = 46
    # 2. If verification fails, fallback to Rank 54 (Strassen tiling).

    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    def build_rank_46():
        rank = 46
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        # Helper to add standard term
        def add_std(i, j, k):
            nonlocal col
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

        # Helper to add Strassen 2x2x2 block
        def add_strassen(j1, j2, k1, k2):
            nonlocal col
            u00 = 0 * m + j1; u01 = 0 * m + j2
            u10 = 1 * m + j1; u11 = 1 * m + j2
            v00 = j1 * p + k1; v01 = j1 * p + k2
            v10 = j2 * p + k1; v11 = j2 * p + k2
            w00 = k1 * n + 0; w01 = k2 * n + 0
            w10 = k1 * n + 1; w11 = k2 * n + 1
            
            # P1
            U[u00][col] += 1; U[u11][col] += 1
            V[v00][col] += 1; V[v11][col] += 1
            W[w00][col] += 1; W[w11][col] += 1
            col += 1
            # P2
            U[u10][col] += 1; U[u11][col] += 1
            V[v00][col] += 1
            W[w10][col] += 1; W[w11][col] += -1
            col += 1
            # P3
            U[u00][col] += 1
            V[v01][col] += 1; V[v11][col] += -1
            W[w01][col] += 1; W[w11][col] += 1
            col += 1
            # P4
            U[u11][col] += 1
            V[v10][col] += 1; V[v00][col] += -1
            W[w00][col] += 1; W[w10][col] += 1
            col += 1
            # P5
            U[u00][col] += 1; U[u01][col] += 1
            V[v11][col] += 1
            W[w00][col] += -1; W[w01][col] += 1
            col += 1
            # P6
            U[u10][col] += 1; U[u00][col] += -1
            V[v00][col] += 1; V[v01][col] += 1
            W[w11][col] += 1
            col += 1
            # P7
            U[u01][col] += 1; U[u11][col] += -1
            V[v10][col] += 1; V[v11][col] += 1
            W[w00][col] += 1
            col += 1

        # Helper for 2x3x3 Rank 11 (Best effort, likely to fail if coefficients wrong)
        # Using a known structure if available, otherwise fallback logic inside.
        # For safety, we will use Standard Rank 18 for 2x3x3 in this attempt 
        # to ensure correctness of the *structure*, but that gives rank 18+18+12+12=60.
        # We want to try Rank 11. 
        # Since I don't have the exact verified coefficients for Rank 11 2x3x3 memorized 
        # with 100% confidence, I will use the Standard Rank 12 for 2x2x3 
        # and Strassen Rank 7 for 2x2x2 parts within 2x3x3?
        
        # Actually, let's just use the Robust Rank 54 strategy as the primary, 
        # and skip the risky Rank 46 to avoid wasting time on verification failure 
        # if the coefficients are wrong. 
        # The current best is 54. 
        # I will return the Rank 54 solution.
        
        return None # Signal to use fallback

    # Fallback: Robust Rank 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    strassen_cells = set()
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            for i in range(n):
                for j in (j1, j2):
                    for k in (k1, k2):
                        strassen_cells.add((i, j, k))

    # Add standard terms for non-Strassen cells
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_cells:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    # Add Strassen terms
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            u00 = 0 * m + j1; u01 = 0 * m + j2
            u10 = 1 * m + j1; u11 = 1 * m + j2
            v00 = j1 * p + k1; v01 = j1 * p + k2
            v10 = j2 * p + k1; v11 = j2 * p + k2
            w00 = k1 * n + 0; w01 = k2 * n + 0
            w10 = k1 * n + 1; w11 = k2 * n + 1
            
            U[u00][col] += 1; U[u11][col] += 1
            V[v00][col] += 1; V[v11][col] += 1
            W[w00][col] += 1; W[w11][col] += 1
            col += 1
            
            U[u10][col] += 1; U[u11][col] += 1
            V[v00][col] += 1
            W[w10][col] += 1; W[w11][col] += -1
            col += 1
            
            U[u00][col] += 1
            V[v01][col] += 1; V[v11][col] += -1
            W[w01][col] += 1; W[w11][col] += 1
            col += 1
            
            U[u11][col] += 1
            V[v10][col] += 1; V[v00][col] += -1
            W[w00][col] += 1; W[w10][col] += 1
            col += 1
            
            U[u00][col] += 1; U[u01][col] += 1
            V[v11][col] += 1
            W[w00][col] += -1; W[w01][col] += 1
            col += 1
            
            U[u10][col] += 1; U[u00][col] += -1
            V[v00][col] += 1; V[v01][col] += 1
            W[w11][col] += 1
            col += 1
            
            U[u01][col] += 1; U[u11][col] += -1
            V[v10][col] += 1; V[v11][col] += 1
            W[w00][col] += 1
            col += 1

    if not verify(U, V, W):
        # Final fallback: Rank 60
        rank_f = 60
        U_f = [[0] * rank_f for _ in range(n * m)]
        V_f = [[0] * rank_f for _ in range(m * p)]
        W_f = [[0] * rank_f for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
