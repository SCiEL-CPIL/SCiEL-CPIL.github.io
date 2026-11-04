import numpy as np
import time
import random

def construct():
    n, m, p = 2, 4, 4
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1
    T_flat = T.flatten()

    rank_known = 28
    U_known = np.zeros((8, rank_known))
    V_known = np.zeros((16, rank_known))
    W_known = np.zeros((8, rank_known))
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U_known[X_rows[0]][col_offset + 0] += 1
        U_known[X_rows[3]][col_offset + 0] += 1
        V_known[Y_rows[0]][col_offset + 0] += 1
        V_known[Y_rows[3]][col_offset + 0] += 1
        W_known[Z_rows[0]][col_offset + 0] += 1
        W_known[Z_rows[3]][col_offset + 0] += 1
        U_known[X_rows[2]][col_offset + 1] += 1
        U_known[X_rows[3]][col_offset + 1] += 1
        V_known[Y_rows[0]][col_offset + 1] += 1
        W_known[Z_rows[2]][col_offset + 1] += 1
        W_known[Z_rows[3]][col_offset + 1] -= 1
        U_known[X_rows[0]][col_offset + 2] += 1
        V_known[Y_rows[1]][col_offset + 2] += 1
        V_known[Y_rows[3]][col_offset + 2] -= 1
        W_known[Z_rows[1]][col_offset + 2] += 1
        W_known[Z_rows[3]][col_offset + 2] += 1
        U_known[X_rows[3]][col_offset + 3] += 1
        V_known[Y_rows[2]][col_offset + 3] += 1
        V_known[Y_rows[0]][col_offset + 3] -= 1
        W_known[Z_rows[0]][col_offset + 3] += 1
        W_known[Z_rows[2]][col_offset + 3] += 1
        U_known[X_rows[0]][col_offset + 4] += 1
        U_known[X_rows[1]][col_offset + 4] += 1
        V_known[Y_rows[3]][col_offset + 4] += 1
        W_known[Z_rows[0]][col_offset + 4] -= 1
        W_known[Z_rows[1]][col_offset + 4] += 1
        U_known[X_rows[2]][col_offset + 5] += 1
        U_known[X_rows[0]][col_offset + 5] -= 1
        V_known[Y_rows[0]][col_offset + 5] += 1
        V_known[Y_rows[1]][col_offset + 5] += 1
        W_known[Z_rows[3]][col_offset + 5] += 1
        U_known[X_rows[1]][col_offset + 6] += 1
        U_known[X_rows[3]][col_offset + 6] -= 1
        V_known[Y_rows[2]][col_offset + 6] += 1
        V_known[Y_rows[3]][col_offset + 6] += 1
        W_known[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    X2 = [2, 3, 6, 7]
    Y11 = [0, 1, 4, 5]
    Y12 = [2, 3, 6, 7]
    Y21 = [8, 9, 12, 13]
    Y22 = [10, 11, 14, 15]
    Z11 = [0, 2, 1, 3]
    Z12 = [4, 6, 5, 7]
    
    add_block(0, X1, Y11, Z11)
    add_block(7, X2, Y21, Z11)
    add_block(14, X1, Y12, Z12)
    add_block(21, X2, Y22, Z12)

    def check_solution(U, V, W, rank):
        T_est = np.einsum('ir,sr,tr->ist', U[:, :rank], V[:, :rank], W[:, :rank])
        return np.allclose(T_est, T, atol=1e-9)

    def residual_val(U, V, W, rank):
        T_est = np.einsum('ir,sr,tr->ist', U[:, :rank], V[:, :rank], W[:, :rank])
        return np.sum((T_est.flatten() - T_flat) ** 2)

    best_rank = 28
    best_U = U_known.tolist()
    best_V = V_known.tolist()
    best_W = W_known.tolist()
    
    start_time = time.time()
    deadline = start_time + 110

    # Try to find rank 27 by zeroing each term and doing local search
    for zero_idx in range(28):
        if time.time() > deadline:
            break
        
        U_np = U_known.copy()
        V_np = V_known.copy()
        W_np = W_known.copy()
        
        U_np[:, zero_idx] = 0
        V_np[:, zero_idx] = 0
        W_np[:, zero_idx] = 0
        
        rank = 27
        current_residual = residual_val(U_np, V_np, W_np, rank)
        
        for it in range(3000):
            if time.time() > deadline:
                break
            
            r = random.randint(0, rank - 1)
            mat = random.choice(['U', 'V', 'W'])
            delta = random.choice([-2, -1, 1, 2])
            
            if mat == 'U':
                row = random.randint(0, 7)
                old_val = U_np[row, r]
                U_np[row, r] += delta
            elif mat == 'V':
                row = random.randint(0, 15)
                old_val = V_np[row, r]
                V_np[row, r] += delta
            else:
                row = random.randint(0, 7)
                old_val = W_np[row, r]
                W_np[row, r] += delta
            
            new_residual = residual_val(U_np, V_np, W_np, rank)
            
            if new_residual < current_residual:
                current_residual = new_residual
                if current_residual < 1e-9:
                    if check_solution(U_np, V_np, W_np, rank):
                        best_rank = 27
                        best_U = U_np.tolist()
                        best_V = V_np.tolist()
                        best_W = W_np.tolist()
                        break
            else:
                if mat == 'U':
                    U_np[row, r] = old_val
                elif mat == 'V':
                    V_np[row, r] = old_val
                else:
                    W_np[row, r] = old_val
        
        if best_rank == 27:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
