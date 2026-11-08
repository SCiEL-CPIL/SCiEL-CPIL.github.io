# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Schoolbook solution
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Construct the target tensor T of size (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    # Try to find lower rank decompositions
    # We search for ranks from 20 to 35
    for rank in range(20, 36):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Use multiple seeds
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(seed * 1000 + rank * 100)
            
            # Initialize
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # ALS with full tensor error
            for it in range(50):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                # For each a, minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
                # This is a linear least squares problem for U[a, :]
                for a in range(n*m):
                    # Construct matrix M such that U[a, :] = lstsq(M, T[a, :, :])
                    # M[b*c, r] = V[b, r] * W[c, r]
                    M = np.zeros((m*p * p*n, rank))
                    for b in range(m*p):
                        for c in range(p*n):
                            M[b * p*n + c, :] = V[b, :] * W[c, :]
                    t_vec = T[a, :, :].flatten()
                    try:
                        U[a, :] = np.linalg.lstsq(M, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(m*p):
                    M = np.zeros((n*m * p*n, rank))
                    for a in range(n*m):
                        for c in range(p*n):
                            M[a * p*n + c, :] = U[a, :] * W[c, :]
                    t_vec = T[:, b, :].flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(M, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c in range(p*n):
                    M = np.zeros((n*m * m*p, rank))
                    for a in range(n*m):
                        for b in range(m*p):
                            M[a * m*p + b, :] = U[a, :] * V[b, :]
                    t_vec = T[:, :, c].flatten()
                    try:
                        W[c, :] = np.linalg.lstsq(M, t_vec, rcond=None)[0]
                    except:
                        pass
            
            # Round to half-integers and verify
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify against full tensor T
            valid = True
            # Reconstruct tensor
            T_rec = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T_rec += np.einsum('ar,br,cr->abc', U_r[:, r], V_r[:, r], W_r[:, r])
            
            # Check if T_rec matches T
            if not np.allclose(T_rec, T, atol=1e-9):
                valid = False
            
            if valid:
                best_rank = rank
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
