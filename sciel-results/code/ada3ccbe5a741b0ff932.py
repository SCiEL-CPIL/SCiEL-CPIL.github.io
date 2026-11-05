            import time
            import numpy as np
            from scipy.optimize import minimize

            DIMENSIONS = [3, 3, 7]

            def construct():
                n, m, p = DIMENSIONS
                start_time = time.time()
                deadline = start_time + 140

                # Construct the matrix multiplication tensor T of shape (n*m, m*p, p*n)
                # T[a, b, c] = 1 if a=(i,j), b=(j',k), c=(i',k') and i==i', j==j', k==k'
                nm = n * m
                mp = m * p
                pn = p * n
                T = np.zeros((nm, mp, pn))
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = i * p + k
                            T[a, b, c] = 1.0
                
                T_flat = T.flatten()

                def compute_error(params, rank):
                    U = params[:nm * rank].reshape(nm, rank)
                    V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                    W = params[nm * rank + mp * rank:].reshape(pn, rank)
                    
                    # D[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                    D = np.einsum('ar,br,cr->abc', U, V, W)
                    err = D.flatten() - T_flat
                    return np.sum(err**2)

                def get_matrices(params, rank):
                    U = params[:nm * rank].reshape(nm, rank)
                    V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                    W = params[nm * rank + mp * rank:].reshape(pn, rank)
                    return U, V, W

                # Initialize with schoolbook rank
                best_rank = nm * mp # This is wrong. Schoolbook rank is n*m*p? No.
                # Schoolbook rank for n x m by m x p is n*m*p?
                # C[i,k] = sum_j A[i,j] B[j,k].
                # Each term A[i,j] B[j,k] is a rank 1 tensor?
                # T[i,j, j',k, i',k'] = delta...
                # Each (i,j,k) contributes 1 to T[i*m+j, j*p+k, i*p+k].
                # There are n*m*p such terms.
                # So rank is n*m*p = 3*3*7 = 63.
                
                best_rank = n * m * p
                best_U = np.zeros((nm, best_rank))
                best_V = np.zeros((mp, best_rank))
                best_W = np.zeros((pn, best_rank))
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = i * p + k
                            best_U[a, col] = 1
                            best_V[b, col] = 1
                            best_W[c, col] = 1
                            col += 1

                search_deadline = start_time + 120
                
                # Search for lower rank
                # We know rank 63 is valid.
                # Try to find lower.
                for rank in range(best_rank - 1, 10, -1):
                    if time.time() > search_deadline:
                        break
                    
                    found = False
                    # Fewer attempts
                    for attempt in range(3):
                        if time.time() > search_deadline:
                            break
                        
                        # Random initialization
                        U = np.random.randn(nm, rank)
                        V = np.random.randn(mp, rank)
                        W = np.random.randn(pn, rank)
                        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                        
                        try:
                            res = minimize(compute_error, x0, args=(rank,), method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
                            if res.fun < 1e-6:
                                U_opt, V_opt, W_opt = get_matrices(res.x, rank)
                                
                                # Round to half-integers
                                U_rounded = np.round(U_opt * 2) / 2
                                V_rounded = np.round(V_opt * 2) / 2
                                W_rounded = np.round(W_opt * 2) / 2
                                
                                params_rounded = np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()])
                                err_rounded = compute_error(params_rounded, rank)
                                
                                if err_rounded < 1e-6:
                                    best_rank = rank
                                    best_U = U_rounded
                                    best_V = V_rounded
                                    best_W = W_rounded
                                    found = True
                                    break
                                else:
                                    # If not half-integer, but very low error, accept?
                                    # Task requires integer/half-integer.
                                    # If we can't round, we might need to search harder or accept non-integer if allowed?
                                    # "integer/half-integer entries"
                                    # So we MUST round.
                                    pass
                        except Exception:
                            continue
                    
                    if found:
                        continue
                    else:
                        # If not found at this rank, continue to lower rank?
                        # Usually if rank R is not found, R-1 is harder.
                        # But maybe random init failed.
                        pass

                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            