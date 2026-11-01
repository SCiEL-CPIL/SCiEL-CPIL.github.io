import numpy as np
import time

def construct():
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Try to find a rank 20 decomposition
    rank_target = 20
    deadline = time.time() + 140.0  # Safety margin
    
    # Initialize with rank 21 Strassen and drop one column?
    # Or random init?
    # Random init is risky.
    # Let's try to initialize by taking the rank 21 solution and projecting?
    # Actually, let's just use a simple ALS loop.
    
    # We need a function to compute the residual
    def residual(U, V, W):
        # T_approx = sum U[:,r] outer V[:,r] outer W[:,r]
        # Residual = T - T_approx
        # For ALS, we fix U, V and solve for W, etc.
        pass
        
    # Let's implement ALS for Tensor Decomposition
    # T_{abc} = sum_r U_{ar} V_{br} W_{cr}
    # Fix U, V. Then for each c, W_{cr} is part of the sum.
    # Actually, W is a matrix.
    # T_{ab,:} = sum_r U_{ar} V_{br} W_{r,:}
    # Let M_{ab} = T_{ab,:} (shape pn)
    # M_{ab} = sum_r (U_{ar} V_{br}) W_{r,:}
    # This is a linear system for W?
    # No, W is shared for all a,b.
    # We can vectorize.
    # Let U_flat (nm x rank), V_flat (mp x rank).
    # The term for r is U[:,r] outer V[:,r].
    # This is a matrix of size nm x mp.
    # We want to approximate T (nm x mp x pn) by sum_r (U[:,r] outer V[:,r]) outer W[:,r].
    # This is equivalent to:
    # Reshape T to (nm*mp, pn).
    # Let T_2d = T.reshape(nm*mp, pn).
    # Let A_r = (U[:,r] outer V[:,r]).reshape(nm*mp).
    # Then T_2d = sum_r A_r outer W[:,r].
    # This is a standard matrix factorization problem: T_2d = A W^T?
    # No. T_2d[i,j] = sum_r A_r[i] W[r,j].
    # So T_2d = A_r_matrix @ W^T.
    # Where A_r_matrix is (nm*mp) x rank.
    # So if we fix U and V, we can solve for W exactly by least squares!
    # Similarly for U and V.
    
    def solve_for_W(U, V):
        # Construct A_r_matrix
        # U: (nm, rank), V: (mp, rank)
        # A_r[i,j] = U[i,r] * V[j,r]
        # We want W such that T_2d = A @ W.T
        # A is (nm*mp, rank)
        # W is (rank, pn)
        # T_2d is (nm*mp, pn)
        
        # Efficient construction of A
        # A_{(i,j), r} = U_{i,r} V_{j,r}
        # This is the outer product of U and V for each r?
        # No, it's a 3D tensor U outer V.
        # We can use einsum or broadcasting.
        # A_flat = np.einsum('ir,jr->ijr', U, V).reshape(nm*mp, rank)
        
        A_flat = np.einsum('ir,jr->ijr', U, V).reshape(nm * mp, rank_target)
        # Solve T_2d = A_flat @ W.T
        # W.T = pinv(A_flat) @ T_2d
        # Or least squares
        W_T, residuals, rank, s = np.linalg.lstsq(A_flat, T.reshape(nm * mp, pn), rcond=None)
        return W_T.T # (rank, pn)

    def solve_for_U(W, V):
        # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # Fix W, V.
        # For each c, T_{a,b,c} = sum_r U_{a,r} (V_{b,r} W_{c,r})
        # Let M_c = T[:,:,c] (nm x mp)
        # M_c = U @ (V * W_c).T ?
        # V * W_c: V is (mp, rank), W_c is (rank,).
        # Let Z_c = V * W_c (mp x rank) where Z_c[b,r] = V[b,r] W[c,r].
        # Then M_c = U @ Z_c.T
        # U (nm x rank), Z_c.T (rank x mp).
        # We need to solve for U for all c simultaneously.
        # Stack M_c into a big matrix?
        # Or solve for each c and average?
        # Better: 
        # T_reshaped = T.transpose(2, 0, 1).reshape(pn, nm*mp)
        # We want U such that for all c, T_reshaped[c] = (U @ Z_c.T).flatten()
        # This is a linear system for U.
        # Let's construct the big system.
        # For each r, U[:,r] is a variable.
        # Actually, we can just do least squares on the stacked system.
        # Let's construct a matrix B of size (pn * nm * mp, rank * nm)? No.
        
        # Simpler: 
        # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # Let's fix r. The contribution is U[:,r] outer (V[:,r] * W[:,r]).
        # This is getting complicated to vectorize for U solve.
        
        # Alternative: 
        # We can solve for U column by column? No, they are coupled.
        # Let's use the property that we can solve for U by treating V*W as the "B" matrix.
        # Let M = V * W (mp x pn x rank)? No.
        # Let's just iterate c.
        # For each c, we have M_c = U @ Z_c.T
        # This is a linear system for U.
        # We can stack all c.
        # Let A_big be (pn * mp, nm * rank)? No.
        
        # Let's just use np.linalg.lstsq on the full tensor.
        # We want to minimize || T - sum_r U[:,r] outer V[:,r] outer W[:,r] ||
        # Fix V, W.
        # This is linear in U.
        # T_{a,b,c} = sum_r U_{a,r} (V_{b,r} W_{c,r})
        # Let Y_{a, b, c} = V_{b,r} W_{c,r} ? No, r is summed.
        # Let's define a matrix A_U of size (nm, rank * mp * pn)?
        # A_U[a, (r, b, c)] = V[b,r] * W[c,r]
        # Then T.flatten() = A_U @ U.flatten()
        
        A_U = np.einsum('br,cr->bcr', V, W) # (mp, pn, rank)
        # We need to map indices correctly.
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Vectorize T to (nm*mp*pn,).
        # Vectorize U to (nm*rank,).
        # We need a matrix M of size (nm*mp*pn, nm*rank) such that T_flat = M @ U_flat.
        # M[(a,b,c), (a', r)] = delta_{a,a'} * V[b,r] * W[c,r]
        # This is a huge sparse matrix.
        
        # Instead, let's just solve for U using the fact that for each a,
        # T[a,:,:] = sum_r U[a,r] (V[:,r] outer W[:,r])
        # Let B_r = V[:,r] outer W[:,r] (mp x pn)
        # T[a,:,:] = sum_r U[a,r] B_r
        # This is a linear system for U[a,:] for each a?
        # No, U[a,:] is the same for all a? No, U is a matrix.
        # U[a,r] is the variable.
        # For a fixed a, we have T[a,:,:] = sum_r U[a,r] B_r.
        # This determines U[a,:] uniquely (if B_r are independent).
        # So we can solve for each row of U independently!
        
        U_new = np.zeros((nm, rank_target))
        for a in range(nm):
            # T[a, :, :] is (mp, pn)
            # We want to express it as sum_r U[a,r] B_r
            # Where B_r = V[:,r] outer W[:,r]
            # Let's stack B_r into a matrix B_mat of size (mp*pn, rank)
            B_mat = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, rank_target)
            T_a_flat = T[a, :, :].flatten()
            U_a, residuals, rank, s = np.linalg.lstsq(B_mat, T_a_flat, rcond=None)
            U_new[a, :] = U_a
        return U_new

    def solve_for_V(U, W):
        # Similar to U
        V_new = np.zeros((mp, rank_target))
        for b in range(mp):
            # T[:, b, :] is (nm, pn)
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed b, T[:,b,:] = sum_r V[b,r] (U[:,r] outer W[:,r])
            # Let C_r = U[:,r] outer W[:,r] (nm x pn)
            C_mat = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, rank_target)
            T_b_flat = T[:, b, :].flatten()
            V_b, residuals, rank, s = np.linalg.lstsq(C_mat, T_b_flat, rcond=None)
            V_new[b, :] = V_b
        return V_new

    # Initialize
    # Use random init
    np.random.seed(42)
    U = np.random.randn(nm, rank_target)
    V = np.random.randn(mp, rank_target)
    W = np.random.randn(pn, rank_target)
    
    # ALS Loop
    for i in range(100):
        if time.time() > deadline:
            break
        U = solve_for_U(W, V)
        V = solve_for_V(U, W)
        W = solve_for_W(U, V)
        
        # Check convergence / exactness
        # Compute reconstruction error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T - T_approx)
        if error < 1e-10:
            # Verify exactness with higher precision or rational check
            # For now, if error is tiny, it's likely exact
            # Check if entries are close to integers/half-integers?
            # The problem asks for integer/half-integer entries.
            # If the solution is exact, it should be close to such values.
            # Let's snap to nearest half-integer?
            # U_snapped = np.round(2*U)/2
            # Check if snapped version is still exact.
            
            U_s = np.round(2*U)/2
            V_s = np.round(2*V)/2
            W_s = np.round(2*W)/2
            
            T_check = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
            if np.allclose(T_check, T, atol=1e-9):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
    # If rank 20 not found, return rank 21 Strassen
    # (Reuse the logic from the original program)
    
    rank = 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    def map_a(local_idx):
        return local_idx
        
    def map_b(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return 0 * 6 + k0
        elif local_idx == 1:
            return 0 * 6 + k1
        elif local_idx == 2:
            return 1 * 6 + k0
        else:
            return 1 * 6 + k1

    def map_c(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return k0 * 2 + 0
        elif local_idx == 1:
            return k1 * 2 + 0
        elif local_idx == 2:
            return k0 * 2 + 1
        else:
            return k1 * 2 + 1

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    a_global = map_a(local_idx)
                    U[a_global, r] = u_val
                    
                v_val = step['v'][local_idx]
                if v_val != 0:
                    b_global = map_b(local_idx, g)
                    V[b_global, r] = v_val
                    
                w_val = step['w'][local_idx]
                if w_val != 0:
                    c_global = map_c(local_idx, g)
                    W[c_global, r] = w_val

    # Verify rank 21
    T_check = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_check, T, atol=1e-10):
        # Fallback to schoolbook
        rank_school = 24
        U_s = np.zeros((nm, rank_school))
        V_s = np.zeros((mp, rank_school))
        W_s = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
