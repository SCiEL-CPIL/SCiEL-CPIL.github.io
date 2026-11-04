import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    rank = 44
    
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    
    def add_strassen(X_u, Y_v, Z_w, col):
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col
    
    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
    
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 0][col] += 1; W[2][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 1][col] += 1; W[5][col] += 1; col += 1
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 2][col] += 1; W[8][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 3][col] += 1; W[11][col] += 1; col += 1
    
    # Attempt rank reduction by checking for linear dependencies
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check if any column can be eliminated by expressing it as a linear combination
    # of other columns (in the tensor product space)
    removed = []
    for c in range(rank - 1, -1, -1):
        if c in removed:
            continue
        # Try to see if column c is linearly dependent on remaining columns
        # Build the "outer product" matrix for each column: U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # A column is redundant if its tensor can be expressed as a combination of others
        # This is a complex check, so we do a simpler heuristic:
        # Check if U[:,c], V[:,c], W[:,c] are all zero (shouldn't happen) or
        # if the column is a scalar multiple of another
        
        # Simple check: see if this column's U,V,W vectors are proportional to another
        dependent = False
        for c2 in range(rank):
            if c2 == c or c2 in removed:
                continue
            # Check if U[:,c] = alpha * U[:,c2]
            if np.allclose(U_arr[:, c], 0) and np.allclose(U_arr[:, c2], 0):
                continue
            # Find alpha from non-zero entries
            mask_u = U_arr[:, c] != 0
            if np.any(mask_u):
                alpha_u = U_arr[mask_u, c] / U_arr[mask_u, c2]
                if len(set(alpha_u)) == 1:
                    alpha = alpha_u[0]
                    if np.allclose(U_arr[:, c], alpha * U_arr[:, c2]) and \
                       np.allclose(V_arr[:, c], alpha * V_arr[:, c2]) and \
                       np.allclose(W_arr[:, c], alpha * W_arr[:, c2]):
                        dependent = True
                        break
        
        if dependent:
            removed.append(c)
    
    if removed:
        # Rebuild without removed columns
        keep = [c for c in range(rank) if c not in removed]
        new_rank = len(keep)
        U_new = np.array(U)[:, keep].tolist()
        V_new = np.array(V)[:, keep].tolist()
        W_new = np.array(W)[:, keep].tolist()
        return {"U": U_new, "V": V_new, "W": W_new}
    
    return {"U": U, "V": V, "W": W}
