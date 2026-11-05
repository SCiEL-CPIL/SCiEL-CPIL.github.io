# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Build target tensor T[i,j,k] = 1 if (i,j,k) is valid else 0
    # T has shape (n, m, p)
    T = np.ones((n, m, p))
    
    NM, MP, PN = n*m, m*p, p*n
    
    def search_binary_decomposition_fast(n, m, p, rank, max_iters=20000, restarts=50):
        NM, MP, PN = n*m, m*p, p*n
        best_err = np.inf
        best_sol = None
        
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
            U = np.random.randint(0, 2, (n, m, rank))
            V = np.random.randint(0, 2, (m, p, rank))
            W = np.random.randint(0, 2, (p, n, rank))
            
            S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            curr_err = np.sum((S - 1.0)**2)
            
            if curr_err < best_err:
                best_err = curr_err
                best_sol = (U.copy(), V.copy(), W.copy(), S.copy())
            if curr_err == 0:
                break
                
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                if curr_err == 0:
                    break
                    
                var_type = np.random.choice(3)
                if var_type == 0:
                    i, j, c = np.random.randint(n), np.random.randint(m), np.random.randint(rank)
                    old_u = U[i, j, c]
                    new_u = 1 - old_u
                    delta_u = new_u - old_u
                    
                    delta_S = delta_u * V[j, :, c] * W[:, i, c]
                    
                    old_S_slice = S[i, j, :].copy()
                    new_S_slice = old_S_slice + delta_S
                    
                    old_err_slice = np.sum((old_S_slice - 1.0)**2)
                    new_err_slice = np.sum((new_S_slice - 1.0)**2)
                    
                    delta_err = new_err_slice - old_err_slice
                    
                    if delta_err <= 0:
                        S[i, j, :] = new_S_slice
                        U[i, j, c] = new_u
                        curr_err += delta_err
                        if curr_err < best_err:
                            best_err = curr_err
                            best_sol = (U.copy(), V.copy(), W.copy(), S.copy())
                        
                elif var_type == 1:
                    j, k, c = np.random.randint(m), np.random.randint(p), np.random.randint(rank)
                    old_v = V[j, k, c]
                    new_v = 1 - old_v
                    delta_v = new_v - old_v
                    
                    delta_S = delta_v * U[:, j, c] * W[k, :, c]
                    
                    old_S_slice = S[:, j, k].copy()
                    new_S_slice = old_S_slice + delta_S
                    
                    old_err_slice = np.sum((old_S_slice - 1.0)**2)
                    new_err_slice = np.sum((new_S_slice - 1.0)**2)
                    
                    delta_err = new_err_slice - old_err_slice
                    
                    if delta_err <= 0:
                        S[:, j, k] = new_S_slice
                        V[j, k, c] = new_v
                        curr_err += delta_err
                        if curr_err < best_err:
                            best_err = curr_err
                            best_sol = (U.copy(), V.copy(), W.copy(), S.copy())
                            
                else:
                    k, i, c = np.random.randint(p), np.random.randint(n), np.random.randint(rank)
                    old_w = W[k, i, c]
                    new_w = 1 - old_w
                    delta_w = new_w - old_w
                    
                    delta_S = delta_w * U[i, :, c] * V[:, k, c]
                    
                    old_S_slice = S[i, :, k].copy()
                    new_S_slice = old_S_slice + delta_S
                    
                    old_err_slice = np.sum((old_S_slice - 1.0)**2)
                    new_err_slice = np.sum((new_S_slice - 1.0)**2)
                    
                    delta_err = new_err_slice - old_err_slice
                    
                    if delta_err <= 0:
                        S[i, :, k] = new_S_slice
                        W[k, i, c] = new_w
                        curr_err += delta_err
                        if curr_err < best_err:
                            best_err = curr_err
                            best_sol = (U.copy(), V.copy(), W.copy(), S.copy())
                            
            if curr_err == 0:
                break

        if best_err < 1e-8:
            U, V, W, _ = best_sol
            return (U.reshape(NM, rank).tolist(), V.reshape(MP, rank).tolist(), W.reshape(PN, rank).tolist())
        return None

    # Try to find a binary decomposition for small ranks
    for rank in range(9, 21):
        if time.time() > deadline - 20:
            break
        result = search_binary_decomposition_fast(n, m, p, rank)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}

    # Fallback to schoolbook
    U = [[0] * rank_schoolbook for _ in range(NM)]
    V = [[0] * rank_schoolbook for _ in range(MP)]
    W = [[0] * rank_schoolbook for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
