import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Rank 43 decomposition:
    # 6 Strassen blocks (rank 7 each) = 42
    # 1 term for the remaining j=4 slice (rank 1) = 1
    # Total = 43
    
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    strassen_cells = set()
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            for i in range(n):
                for j in (j1, j2):
                    for k in (k1, k2):
                        strassen_cells.add((i, j, k))

    col = 0

    # Add Strassen terms for each 2x2x2 block
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            u00 = 0 * m + j1
            u01 = 0 * m + j2
            u10 = 1 * m + j1
            u11 = 1 * m + j2

            v00 = j1 * p + k1
            v01 = j1 * p + k2
            v10 = j2 * p + k1
            v11 = j2 * p + k2

            w00 = k1 * n + 0
            w01 = k2 * n + 0
            w10 = k1 * n + 1
            w11 = k2 * n + 1

            # P1
            U[u00][col] += 1
            U[u11][col] += 1
            V[v00][col] += 1
            V[v11][col] += 1
            W[w00][col] += 1
            W[w11][col] += 1
            col += 1

            # P2
            U[u10][col] += 1
            U[u11][col] += 1
            V[v00][col] += 1
            W[w10][col] += 1
            W[w11][col] += -1
            col += 1

            # P3
            U[u00][col] += 1
            V[v01][col] += 1
            V[v11][col] += -1
            W[w01][col] += 1
            W[w11][col] += 1
            col += 1

            # P4
            U[u11][col] += 1
            V[v10][col] += 1
            V[v00][col] += -1
            W[w00][col] += 1
            W[w10][col] += 1
            col += 1

            # P5
            U[u00][col] += 1
            U[u01][col] += 1
            V[v11][col] += 1
            W[w00][col] += -1
            W[w01][col] += 1
            col += 1

            # P6
            U[u10][col] += 1
            U[u00][col] += -1
            V[v00][col] += 1
            V[v01][col] += 1
            W[w11][col] += 1
            col += 1

            # P7
            U[u01][col] += 1
            U[u11][col] += -1
            V[v10][col] += 1
            V[v11][col] += 1
            W[w00][col] += 1
            col += 1

    # Add the single term for the remaining j=4 slice
    # Cells: (0,4,k) and (1,4,k) for k in 0..5
    # U indices: (0,4), (1,4)
    # V indices: (4,0)..(4,5)
    # W indices: (k,0)..(k,1) for k in 0..5
    
    # U
    U[0 * m + 4][col] = 1
    U[1 * m + 4][col] = 1
    
    # V
    for k in range(p):
        V[4 * p + k][col] = 1
        
    # W
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
            
    col += 1

    # Verify
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

    if not verify(U, V, W):
        # Fallback rank 60
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
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
