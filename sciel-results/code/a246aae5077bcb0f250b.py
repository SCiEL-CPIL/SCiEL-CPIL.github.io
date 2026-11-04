import numpy as np
import time
import random

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = i + j * n
                T[x, y, z] = 1

    # Current known solution rank 28
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
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            u = U_np[:, r]
            v = V_np[:, r]
            w = W_np[:, r]
            T_est += np.einsum('r,s,t->rst', u, v, w)
            
        if not np.allclose(T_est, T, atol=1e-9):
            return False
        return True

    best_rank = 28
    best_U = U_known.tolist()
    best_V = V_known.tolist()
    best_W = W_known.tolist()
    
    start_time = time.time()
    deadline = start_time + 110

    # Fast incremental residual computation
    # When we change U[row][r] by delta, the change in T is delta * outer(V[:,r], W[:,r])
    # We can maintain the residual incrementally
    
    def compute_residual(U, V, W, rank):
        """Compute sum of squared differences between estimated and target tensor."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r])
        
        return np.sum((T_est - T) ** 2)

    # Try to find rank 27 by zeroing one term and adjusting others
    # Key insight: use incremental residual updates for speed
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        U = [row[:] for row in best_U]
        V = [row[:] for row in best_V]
        W = [row[:] for row in best_W]
        
        # Zero out the last term (index 27)
        for i in range(8):
            U[i][27] = 0
        for i in range(16):
            V[i][27] = 0
        for i in range(8):
            W[i][27] = 0
            
        rank = 27
        current_residual = compute_residual(U, V, W, rank)
        best_residual = current_residual
        best_U_27 = [row[:] for row in U]
        best_V_27 = [row[:] for row in V]
        best_W_27 = [row[:] for row in W]
        
        # Convert to numpy for faster incremental updates
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Precompute outer products of V and W columns for fast residual updates
        # outer_vw[r] = outer(V[:,r], W[:,r]) shape (16, 8)
        outer_vw = np.zeros((rank, 16, 8), dtype=float)
        for r in range(rank):
            outer_vw[r] = np.outer(V_np[:, r], W_np[:, r])
        
        # Compute current T_est
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += U_np[:, r][:, None, None] * outer_vw[r][None, :, :]
        
        current_res = np.sum((T_est - T) ** 2)
        
        for it in range(20000):
            if time.time() > deadline:
                break
                
            # Try all single moves and pick the best
            best_move = None
            best_new_res = current_res
            
            # Try changing U entries
            for r in range(rank):
                for row in range(8):
                    if U_np[row, r] == 0 and random.random() > 0.1:
                        continue  # Skip zeros most of the time
                    for delta in [-1, 1]:
                        # Change: T_est += delta * outer(V[:,r], W[:,r])
                        # New residual = sum((T_est + delta*outer_vw[r] - T)^2)
                        # = sum((T_est - T)^2) + 2*delta*sum((T_est-T)*outer_vw[r]) + delta^2*sum(outer_vw[r]^2)
                        diff = T_est - T
                        cross = np.sum(diff * outer_vw[r])
                        quad = np.sum(outer_vw[r] ** 2)
                        new_res = current_res + 2 * delta * cross + quad
                        if new_res < best_new_res:
                            best_new_res = new_res
                            best_move = ('U', row, r, delta)
            
            # Try changing V entries
            for r in range(rank):
                for row in range(16):
                    if V_np[row, r] == 0 and random.random() > 0.1:
                        continue
                    for delta in [-1, 1]:
                        # Change: T_est[:, row, :] += delta * U_np[:, r] * W_np[:, r]
                        # This is more complex - need to compute properly
                        # T_est[i, row, k] changes by delta * U_np[i, r] * W_np[k, r]
                        # diff[i, row, k] = T_est[i, row, k] - T[i, row, k]
                        # New term: 2*delta * sum_i,k diff[i,row,k]*U_np[i,r]*W_np[k,r] + delta^2 * sum_i,k (U_np[i,r]*W_np[k,r])^2
                        diff = T_est - T
                        u_col = U_np[:, r]
                        w_col = W_np[:, r]
                        # Sum over i,k of diff[i, row, k] * u_col[i] * w_col[k]
                        cross = np.sum(diff[:, row, :] * u_col[:, None] * w_col[None, :])
                        quad = np.sum((u_col[:, None] * w_col[None, :]) ** 2)
                        new_res = current_res + 2 * delta * cross + quad
                        if new_res < best_new_res:
                            best_new_res = new_res
                            best_move = ('V', row, r, delta)
            
            # Try changing W entries
            for r in range(rank):
                for row in range(8):
                    if W_np[row, r] == 0 and random.random() > 0.1:
                        continue
                    for delta in [-1, 1]:
                        # T_est[i, j, row] changes by delta * U_np[i, r] * V_np[j, r]
                        diff = T_est - T
                        u_col = U_np[:, r]
                        v_col = V_np[:, r]
                        cross = np.sum(diff[:, :, row] * u_col[:, None] * v_col[None, :])
                        quad = np.sum((u_col[:, None] * v_col[None, :]) ** 2)
                        new_res = current_res + 2 * delta * cross + quad
                        if new_res < best_new_res:
                            best_new_res = new_res
                            best_move = ('W', row, r, delta)
            
            if best_move is None:
                break
            
            mat, row, r, delta = best_move
            # Apply the move
            if mat == 'U':
                U_np[row, r] += delta
                U[row][r] = U_np[row, r]
                T_est += delta * outer_vw[r]
            elif mat == 'V':
                V_np[row, r] += delta
                V[row][r] = V_np[row, r]
                # Update T_est: T_est[:, row, :] += delta * outer(U[:,r], W[:,r])
                T_est[:, row, :] += delta * np.outer(U_np[:, r], W_np[:, r])
                # Update outer_vw for all r' where V[row, r'] might be used... 
                # Actually outer_vw[r'] = outer(V[:,r'], W[:,r']), so changing V[row, r] affects outer_vw[r]
                outer_vw[r, row, :] = V_np[row, r] * W_np[:, r]
            else:  # W
                W_np[row, r] += delta
                W[row][r] = W_np[row, r]
                T_est[:, :, row] += delta * np.outer(U_np[:, r], V_np[:, r])
                outer_vw[r, :, row] = V_np[:, r] * W_np[row, r]
            
            current_res = best_new_res
            if current_res < best_residual:
                best_residual = current_res
                best_U_27 = [row[:] for row in U]
                best_V_27 = [row[:] for row in V]
                best_W_27 = [row[:] for row in W]
            
            if best_residual < 1e-9:
                if check_solution(best_U_27, best_V_27, best_W_27, rank):
                    best_rank = 27
                    best_U = best_U_27
                    best_V = best_V_27
                    best_W = best_W_27
                    break
        
        if best_rank == 27:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
