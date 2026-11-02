    def try_rank(rank, max_restarts=5, max_iters=100):
        """Try to find exact rank decomposition via alternating minimization."""
        if time.time() > deadline - 15:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 8:
                return None
            
            np.random.seed(rank * 7919 + restart * 104729 + 17)
            
            # Random initialization with small values
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            best_err = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    return None
                
                # Vectorized Update U
                # U[i*m+j] depends on V and W
                # For each i, j:
                # B[c, k] = V[j*p+k, c] * W[k*n+i, c]
                # We want U[i*m+j, c] such that sum_c U[c] * B[c, k] = 1 for all k
                # This is a linear system for each i, j.
                
                # We can batch this.
                # Shape of V: (m*p, rank) -> (m, p, rank)
                # Shape of W: (p*n, rank) -> (p, n, rank)
                
                V_reshaped = V.reshape(m, p, rank)
                W_reshaped = W.reshape(p, n, rank)
                
                # B[i, j, c, k] = V[j, k, c] * W[k, i, c]
                # V[j, k, c] needs to be broadcast to [i, j, c, k]
                # W[k, i, c] needs to be broadcast to [i, j, c, k]
                
                # V[j, k, c] -> [1, m, rank, p]
                V_b = V_reshaped[np.newaxis, :, :, :] # (1, m, rank, p)
                # W[k, i, c] -> (p, n, rank) -> (n, 1, rank, p) ? No.
                # We need W[k, i, c] for fixed i, k.
                # W is indexed by k, i.
                # W_reshaped[k, i, c].
                # We want B[i, j, c, k] = V[j, k, c] * W[k, i, c].
                
                # V[j, k, c] is at V_b[0, j, c, k].
                # W[k, i, c] is at W_reshaped[k, i, c].
                # Let's permute W to [n, p, rank] -> W_p[i, k, c]
                W_perm = W_reshaped.transpose(1, 0, 2) # (n, p, rank)
                W_b = W_perm[:, np.newaxis, :, :] # (n, 1, rank, p)
                
                # B[i, j, c, k] = V_b[0, j, c, k] * W_b[i, 0, c, k]
                # V_b is (1, m, rank, p). W_b is (n, 1, rank, p).
                # Broadcast: (n, m, rank, p).
                B = V_b * W_b # (n, m, rank, p)
                
                # Solve U[i, j, c] for B[i, j, c, k] * U[i, j, c] = 1 (vector of 1s of length p)
                # B is (n, m, rank, p).
                # We need to solve X^T y = 1 for each i, j.
                # B[i, j] is (rank, p).
                # We want U[i, j] (rank,) such that B[i, j].T @ U[i, j] = ones(p).
                # This is a least squares problem.
                
                # Reshape B to (n*m, rank, p)
                B_flat = B.reshape(n*m, rank, p)
                
                # Use lstsq. 
                # np.linalg.lstsq expects (M, N) and (M,).
                # Here M=rank, N=p.
                # We want U (rank,) such that B.T (p, rank) @ U (rank,) = ones(p).
                # So A = B.T (p, rank), b = ones(p).
                # U = lstsq(A, b).
                
                # B_flat is (n*m, rank, p).
                # Transpose to (n*m, p, rank).
                B_T = B_flat.transpose(0, 2, 1)
                
                # Solve for each of the n*m systems.
                # U_new = (n*m, rank)
                try:
                    # np.linalg.lstsq doesn't batch directly in older numpy, but we can use einsum or loop.
                    # Given 150s, a loop over 25 is fine.
                    U_new = np.zeros((n*m, rank))
                    for idx in range(n*m):
                        A_mat = B_T[idx] # (p, rank)
                        b_vec = np.ones(p)
                        U_new[idx], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U = U_new
                except:
                    pass

                # Similarly for V and W.
                # V[j, k, c] depends on U and W.
                # A[c, i] = U[i, j, c] * W[k, i, c]
                # V[j, k, c] such that sum_c A[c, i] V[j, k, c] = 1 for all i.
                
                U_reshaped = U.reshape(n, m, rank)
                W_reshaped = W.reshape(p, n, rank)
                
                # A[j, k, c, i] = U[i, j, c] * W[k, i, c]
                # U[i, j, c] -> (n, m, rank) -> (1, m, rank, n)
                U_b = U_reshaped[np.newaxis, :, :, :] # (1, m, rank, n)
                # W[k, i, c] -> (p, n, rank) -> (p, 1, rank, n)
                W_b = W_reshaped[:, np.newaxis, :, :] # (p, 1, rank, n)
                
                A = U_b * W_b # (p, m, rank, n)
                # A is indexed by [k, j, c, i].
                # We want V[j, k, c] such that sum_c A[k, j, c, i] V[j, k, c] = 1 for all i.
                # A_flat: (p*m, rank, n).
                A_flat = A.reshape(p*m, rank, n)
                A_T = A_flat.transpose(0, 2, 1) # (p*m, n, rank)
                
                V_new = np.zeros((m*p, rank))
                for idx in range(m*p):
                    # idx corresponds to j, k?
                    # V is (m*p, rank). Index j*p + k.
                    # A is (p, m, ...). Index k, j.
                    # So A_flat index is k*m + j.
                    # We need to map V index j*p + k to A index k*m + j.
                    j = idx // p
                    k = idx % p
                    a_idx = k * m + j
                    A_mat = A_T[a_idx] # (n, rank)
                    b_vec = np.ones(n)
                    V_new[idx], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V = V_new

                # W[k, i, c] depends on U and V.
                # C[c, j] = U[i, j, c] * V[j, k, c]
                # W[k, i, c] such that sum_c C[c, j] W[k, i, c] = 1 for all j.
                
                U_reshaped = U.reshape(n, m, rank)
                V_reshaped = V.reshape(m, p, rank)
                
                # C[i, k, c, j] = U[i, j, c] * V[j, k, c]
                # U[i, j, c] -> (n, m, rank) -> (n, 1, rank, m)
                U_b = U_reshaped[:, np.newaxis, :, :] # (n, 1, rank, m)
                # V[j, k, c] -> (m, p, rank) -> (1, p, rank, m)
                V_b = V_reshaped[np.newaxis, :, :, :] # (1, p, rank, m)
                
                C = U_b * V_b # (n, p, rank, m)
                # C is indexed by [i, k, c, j].
                # We want W[k, i, c] such that sum_c C[i, k, c, j] W[k, i, c] = 1 for all j.
                # C_flat: (n*p, rank, m).
                C_flat = C.reshape(n*p, rank, m)
                C_T = C_flat.transpose(0, 2, 1) # (n*p, m, rank)
                
                W_new = np.zeros((p*n, rank))
                for idx in range(p*n):
                    # W index k*n + i.
                    # C index i*p + k.
                    k = idx // n
                    i = idx % n
                    c_idx = i * p + k
                    C_mat = C_T[c_idx] # (m, rank)
                    b_vec = np.ones(m)
                    W_new[idx], _, _, _ = np.linalg.lstsq(C_mat, b_vec, rcond=None)
                W = W_new

                # Check convergence
                if it % 3 == 2 or it == max_iters - 1:
                    max_err = 0
                    # Vectorized check
                    # Sum_c U[i, j, c] V[j, k, c] W[k, i, c]
                    # U: (n, m, rank)
                    # V: (m, p, rank)
                    # W: (p, n, rank)
                    
                    # Product P[i, j, k, c] = U[i, j, c] V[j, k, c] W[k, i, c]
                    # U_b: (n, m, 1, rank)
                    # V_b: (1, m, p, rank)
                    # W_b: (p, 1, n, rank)
                    
                    U_b = U_reshaped[:, :, np.newaxis, :]
                    V_b = V_reshaped[np.newaxis, :, :, :]
                    W_b = W_reshaped[:, np.newaxis, :, :]
                    
                    # W_b is (p, 1, n, rank). We need W[k, i, c] at [i, j, k, c].
                    # W_b[i, 0, k, c]? No.
                    # W_reshaped is (p, n, rank).
                    # W_b = W_reshaped[np.newaxis, :, np.newaxis, :]? No.
                    # We need W[k, i, c] to align with i, j, k.
                    # W_reshaped[k, i, c].
                    # We want P[i, j, k, c] = U[i, j, c] V[j, k, c] W[k, i, c].
                    
                    # U: (n, m, rank) -> (n, m, 1, rank)
                    # V: (m, p, rank) -> (1, m, p, rank)
                    # W: (p, n, rank) -> (1, 1, p, n) ? No.
                    # W[k, i, c].
                    # We need to broadcast W to (n, m, p, rank).
                    # W is (p, n, rank).
                    # Permute to (n, p, rank).
                    W_perm = W_reshaped.transpose(1, 0, 2) # (n, p, rank)
                    W_b = W_perm[:, np.newaxis, :, :] # (n, 1, p, rank)
                    
                    P = U_b * V_b * W_b # (n, m, p, rank)
                    Sum = np.sum(P, axis=3) # (n, m, p)
                    Err = np.abs(Sum - 1.0)
                    max_err = np.max(Err)
                    
                    if max_err < best_err:
                        best_err = max_err
                        best_state = (U.copy(), V.copy(), W.copy())
                    
                    if max_err < 1e-8:
                        # Round and verify
                        U_r = np.array([[round(x * 2) / 2.0 for x in U[i]] for i in range(n*m)])
                        V_r = np.array([[round(x * 2) / 2.0 for x in V[j]] for j in range(m*p)])
                        W_r = np.array([[round(x * 2) / 2.0 for x in W[k]] for k in range(p*n)])
                        
                        # Verify rounded
                        U_r_sh = U_r.reshape(n, m, rank)
                        V_r_sh = V_r.reshape(m, p, rank)
                        W_r_sh = W_r.reshape(p, n, rank)
                        W_r_perm = W_r_sh.transpose(1, 0, 2)
                        
                        U_b_r = U_r_sh[:, :, np.newaxis, :]
                        V_b_r = V_r_sh[np.newaxis, :, :, :]
                        W_b_r = W_r_perm[:, np.newaxis, :, :]
                        
                        P_r = U_b_r * V_b_r * W_b_r
                        Sum_r = np.sum(P_r, axis=3)
                        Err_r = np.max(np.abs(Sum_r - 1.0))
                        
                        if Err_r < 1e-8:
                            return {
                                "U": U_r.tolist(),
                                "V": V_r.tolist(),
                                "W": W_r.tolist()
                            }
                    
                    if max_err < 1e-8:
                        break

            if best_err < 1e-6:
                U, V, W = best_state
                U_r = np.array([[round(x * 2) / 2.0 for x in U[i]] for i in range(n*m)])
                V_r = np.array([[round(x * 2) / 2.0 for x in V[j]] for j in range(m*p)])
                W_r = np.array([[round(x * 2) / 2.0 for x in W[k]] for k in range(p*n)])
                
                U_r_sh = U_r.reshape(n, m, rank)
                V_r_sh = V_r.reshape(m, p, rank)
                W_r_sh = W_r.reshape(p, n, rank)
                W_r_perm = W_r_sh.transpose(1, 0, 2)
                
                U_b_r = U_r_sh[:, :, np.newaxis, :]
                V_b_r = V_r_sh[np.newaxis, :, :, :]
                W_b_r = W_r_perm[:, np.newaxis, :, :]
                
                P_r = U_b_r * V_b_r * W_b_r
                Sum_r = np.sum(P_r, axis=3)
                Err_r = np.max(np.abs(Sum_r - 1.0))
                
                if Err_r < 1e-8:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        return None
