<<<<<<< SEARCH
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 0, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    if not add_schoolbook_221(0, 0, 6):
        pass
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 2, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    if not add_schoolbook_221(0, 2, 6):
        pass
=======
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 0, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    if not add_schoolbook_221(0, 0, 6):
        pass
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 2, k0):
            break
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    if not add_schoolbook_221(0, 2, 6):
        pass
        
    # After building the full decomposition, attempt rank reduction
    actual_rank = col
    
    if actual_rank > 49 and time.time() < deadline - 5:
        # Build the tensor contributions matrix
        # Each term t contributes u_t[i] * v_t[j] * w_t[k] to T[i,j,k]
        # Flatten: index = i*(m*p) + j*n + k
        n_dim = n * m * p  # 56
        T = np.zeros((n_dim, actual_rank))
        
        for t in range(actual_rank):
            u_t = [U[i][t] for i in range(n * m)]
            v_t = [V[j][t] for j in range(m * p)]
            w_t = [W[k][t] for k in range(p * n)]
            
            for i in range(n * m):
                if u_t[i] != 0:
                    for j in range(m * p):
                        if v_t[j] != 0:
                            for k in range(p * n):
                                if w_t[k] != 0:
                                    T[i * (m * p) + j * n + k, t] += u_t[i] * v_t[j] * w_t[k]
        
        # Check for dependent columns
        removed_cols = set()
        for c in range(actual_rank):
            if time.time() > deadline - 3:
                break
            if c in removed_cols:
                continue
            others = [t for t in range(actual_rank) if t != c and t not in removed_cols]
            if len(others) < actual_rank - 1:
                continue
            
            sub = T[:, others]
            col_c = T[:, c]
            aug = np.column_stack([sub, col_c])
            
            r_sub = np.linalg.matrix_rank(sub)
            r_aug = np.linalg.matrix_rank(aug)
            
            if r_sub == r_aug:
                removed_cols.add(c)
        
        if removed_cols:
            new_rank = actual_rank - len(removed_cols)
            U_new = [[0] * new_rank for _ in range(n * m)]
            V_new = [[0] * new_rank for _ in range(m * p)]
            W_new = [[0] * new_rank for _ in range(p * n)]
            
            new_col = 0
            for t in range(actual_rank):
                if t not in removed_cols:
                    for i in range(n * m):
                        U_new[i][new_col] = U[i][t]
                    for j in range(m * p):
                        V_new[j][new_col] = V[j][t]
                    for k in range(p * n):
                        W_new[k][new_col] = W[k][t]
                    new_col += 1
            
            U = U_new
            V = V_new
            W = W_new
            col = new_rank
>>>>>>> REPLACE
