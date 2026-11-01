import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Max possible rank
    rank_max = n * m * p
    U = np.zeros((n * m, rank_max))
    V = np.zeros((m * p, rank_max))
    W = np.zeros((p * n, rank_max))
    
    col = 0
    
    # Partitions
    J_parts = [[0, 1], [2, 3], [4]]
    K_parts = [[0, 1], [2, 3], [4, 5]]
    
    for J in J_parts:
        for K in K_parts:
            A_rows = [0, 1]
            A_cols = J
            B_rows = J
            B_cols = K
            C_rows = [0, 1]
            C_cols = K
            
            if len(A_cols) == 2 and len(B_cols) == 2:
                # Strassen
                # 1
                U[A_rows[0] * m + A_cols[0], col] = 1
                U[A_rows[1] * m + A_cols[1], col] = 1
                V[B_rows[0] * p + B_cols[0], col] = 1
                V[B_rows[1] * p + B_cols[1], col] = 1
                W[C_cols[0] * n + C_rows[0], col] = 1
                W[C_cols[1] * n + C_rows[1], col] = 1
                col += 1
                # 2
                U[A_rows[1] * m + A_cols[0], col] = 1
                U[A_rows[1] * m + A_cols[1], col] = 1
                V[B_rows[0] * p + B_cols[0], col] = 1
                W[C_cols[0] * n + C_rows[1], col] = 1
                col += 1
                # 3
                U[A_rows[0] * m + A_cols[0], col] = 1
                V[B_rows[0] * p + B_cols[1], col] = 1
                V[B_rows[1] * p + B_cols[1], col] = -1
                W[C_cols[1] * n + C_rows[0], col] = 1
                col += 1
                # 4
                U[A_rows[1] * m + A_cols[1], col] = 1
                V[B_rows[1] * p + B_cols[0], col] = 1
                V[B_rows[0] * p + B_cols[0], col] = -1
                W[C_cols[0] * n + C_rows[1], col] = 1
                col += 1
                # 5
                U[A_rows[0] * m + A_cols[0], col] = 1
                U[A_rows[0] * m + A_cols[1], col] = 1
                V[B_rows[1] * p + B_cols[1], col] = 1
                W[C_cols[1] * n + C_rows[0], col] = 1
                col += 1
                # 6
                U[A_rows[1] * m + A_cols[0], col] = 1
                U[A_rows[0] * m + A_cols[0], col] = -1
                V[B_rows[0] * p + B_cols[0], col] = 1
                V[B_rows[0] * p + B_cols[1], col] = 1
                W[C_cols[1] * n + C_rows[1], col] = 1
                col += 1
                # 7
                U[A_rows[0] * m + A_cols[1], col] = 1
                U[A_rows[1] * m + A_cols[1], col] = -1
                V[B_rows[1] * p + B_cols[0], col] = 1
                V[B_rows[1] * p + B_cols[1], col] = 1
                W[C_cols[0] * n + C_rows[0], col] = 1
                col += 1
            elif len(A_cols) == 1 and len(B_cols) == 2:
                # Schoolbook 2x1x2
                # 1
                U[A_rows[0] * m + A_cols[0], col] = 1
                V[B_rows[0] * p + B_cols[0], col] = 1
                W[C_cols[0] * n + C_rows[0], col] = 1
                col += 1
                # 2
                U[A_rows[0] * m + A_cols[0], col] = 1
                V[B_rows[0] * p + B_cols[1], col] = 1
                W[C_cols[1] * n + C_rows[0], col] = 1
                col += 1
                # 3
                U[A_rows[1] * m + A_cols[0], col] = 1
                V[B_rows[0] * p + B_cols[0], col] = 1
                W[C_cols[0] * n + C_rows[1], col] = 1
                col += 1
                # 4
                U[A_rows[1] * m + A_cols[0], col] = 1
                V[B_rows[0] * p + B_cols[1], col] = 1
                W[C_cols[1] * n + C_rows[1], col] = 1
                col += 1
            else:
                # Fallback
                for i in range(n):
                    for j in A_cols:
                        for k in B_cols:
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
                            
    # Check time
    if time.time() - start_time > 145:
         # Return best found so far (which is the full construction)
         pass

    U = U[:, :col].tolist()
    V = V[:, :col].tolist()
    W = W[:, :col].tolist()
    
    return {"U": U, "V": V, "W": W}
