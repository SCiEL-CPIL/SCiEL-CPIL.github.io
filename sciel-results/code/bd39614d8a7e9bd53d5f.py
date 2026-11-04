<<<<<<< SEARCH
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
=======
    # Try to find rank 27 by zeroing one term and adjusting others
    # Key insight: use precomputed matrices for fast incremental residual updates
    
    for restart in range(50):
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
        
        # Precompute outer products for fast residual updates
        # outer_vw[r] = outer(V[:,r], W[:,r]) shape (16, 8)
        outer_vw = np.zeros((rank, 16, 8), dtype=float)
        # outer_uw[r] = outer(U[:,r], W[:,r]) shape (8, 8) for V changes
        outer_uw = np.zeros((rank, 8, 8), dtype=float)
        # outer_uv[r] = outer(U[:,r], V[:,r]) shape (8, 16) for W changes
        outer_uv = np.zeros((rank, 8, 16), dtype=float)
        
        for r in range(rank):
            outer_vw[r] = np.outer(V_np[:, r], W_np[:, r])
            outer_uw[r] = np.outer(U_np[:, r], W_np[:, r])
            outer_uv[r] = np.outer(U_np[:, r], V_np[:, r])
        
        # Precompute quad terms (constant for each r)
        quad_vw = np.sum(outer_vw ** 2, axis=(1, 2))  # shape (rank,)
        quad_uw = np.sum(outer_uw ** 2, axis=(1, 2))  # shape (rank,)
        quad_uv = np.sum(outer_uv ** 2, axis=(1, 2))  # shape (rank,)
        
        # Compute current T_est
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += U_np[:, r][:, None, None] * outer_vw[r][None, :, :]
        
        current_res = np.sum((T_est - T) ** 2)
        
        for it in range(30000):
            if time.time() > deadline:
                break
                
            # Try all single moves and pick the best
            best_move = None
            best_new_res = current_res
            
            diff = T_est - T
            
            # Try changing U entries
            for r in range(rank):
                cross = np.sum(diff * outer_vw[r])
                q = quad_vw[r]
                for delta in [-1, 1]:
                    new_res = current_res + 2 * delta * cross + q
                    if new_res < best_new_res:
                        best_new_res = new_res
                        best_move = ('U', None, r, delta)
            
            # Try changing V entries
            for r in range(rank):
                u_col = U_np[:, r]
                w_col = W_np[:, r]
                # cross_v[row] = sum_i,k diff[i,row,k] * u_col[i] * w_col[k]
                # = sum_i (diff[:, row, :] @ w_col) * u_col[i] ... let's vectorize
                # diff[:, row, :] shape (8, 8), u_col shape (8,), w_col shape (8,)
                # cross_v[row] = sum_i sum_k diff[i,row,k] * u_col[i] * w_col[k]
                # = (diff @ np.outer(u_col, w_col)) [row] ... no
                # = sum_i u_col[i] * sum_k diff[i,row,k] * w_col[k]
                # Let temp[row, i] = sum_k diff[i, row, k] * w_col[k]
                # Then cross_v[row] = sum_i temp[row, i] * u_col[i]
                
                # Vectorized: for each row j in 16, cross_v[j] = sum_i,k diff[i,j,k]*u[i]*w[k]
                # = sum_i u[i] * (diff[:, j, :] @ w)
                # diff shape (8,16,8), w shape (8,)
                # diff @ w along last axis: shape (8, 16)
                diff_w = np.einsum('ijk,k->ij', diff, w_col)  # shape (8, 16)
                cross_v = np.einsum('ij,i->j', diff_w, u_col)  # shape (16,)
                
                q = quad_uw[r]
                for delta in [-1, 1]:
                    # Find best row to change
                    for delta_sign in [delta]:
                        # We want to minimize current_res + 2*delta*cross_v[j] + q
                        # So pick j that minimizes cross_v[j] if delta=1, or maximizes if delta=-1
                        pass
                # Simpler: just find the best (row, delta) pair
                if delta == 1:
                    best_j = np.argmin(cross_v)
                    new_res = current_res + 2 * cross_v[best_j] + q
                    if new_res < best_new_res:
                        best_new_res = new_res
                        best_move = ('V', best_j, r, 1)
                else:
                    best_j = np.argmax(cross_v)
                    new_res = current_res - 2 * cross_v[best_j] + q
                    if new_res < best_new_res:
                        best_new_res = new_res
                        best_move = ('V', best_j, r, -1)
            
            # Try changing W entries
            for r in range(rank):
                u_col = U_np[:, r]
                v_col = V_np[:, r]
                # cross_w[row] = sum_i,j diff[i,j,row] * u_col[i] * v_col[j]
                diff_u = np.einsum('ijk,i->jk', diff, u_col)  # shape (16, 8)
                cross_w = np.einsum('jk,k->j', diff_u, v_col)  # shape (8,)
                
                q = quad_uv[r]
                if delta == 1:
                    best_j = np.argmin(cross_w)
                    new_res = current_res + 2 * cross_w[best_j] + q
                    if new_res < best_new_res:
                        best_new_res = new_res
                        best_move = ('W', best_j, r, 1)
                else:
                    best_j = np.argmax(cross_w)
                    new_res = current_res - 2 * cross_w[best_j] + q
                    if new_res < best_new_res:
                        best_new_res = new_res
                        best_move = ('W', best_j, r, -1)
            
            if best_move is None:
                break
            
            mat, row, r, delta = best_move
            # Apply the move
            if mat == 'U':
                # For U, we need to find which row to change
                # Recompute: find best row for this r and delta
                cross_u = np.sum(diff * outer_vw[r], axis=(1, 2))  # shape (8,)
                if delta == 1:
                    row = np.argmin(cross_u)
                else:
                    row = np.argmax(cross_u)
                U_np[row, r] += delta
                U[row][r] = U_np[row, r]
                T_est += delta * outer_vw[r]
                # Update outer_uw and outer_uv for this r
                outer_uw[r] = np.outer(U_np[:, r], W_np[:, r])
                outer_uv[r] = np.outer(U_np[:, r], V_np[:, r])
                quad_uw[r] = np.sum(outer_uw[r] ** 2)
                quad_uv[r] = np.sum(outer_uv[r] ** 2)
            elif mat == 'V':
                V_np[row, r] += delta
                V[row][r] = V_np[row, r]
                T_est[:, row, :] += delta * outer_uw[r]
                outer_vw[r, row, :] = V_np[row, r] * W_np[:, r]
                outer_uv[r, :, row] = U_np[:, r] * V_np[row, r]
                quad_vw[r] = np.sum(outer_vw[r] ** 2)
                quad_uv[r] = np.sum(outer_uv[r] ** 2)
            else:  # W
                W_np[row, r] += delta
                W[row][r] = W_np[row, r]
                T_est[:, :, row] += delta * outer_uv[r]
                outer_vw[r, :, row] = V_np[:, r] * W_np[row, r]
                outer_uw[r, :, row] = U_np[:, r] * W_np[row, r]
                quad_vw[r] = np.sum(outer_vw[r] ** 2)
                quad_uw[r] = np.sum(outer_uw[r] ** 2)
            
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
>>>>>>> REPLACE
