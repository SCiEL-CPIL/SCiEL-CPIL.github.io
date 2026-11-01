# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    U_dim = n * m  # 6
    V_dim = m * p  # 9
    W_dim = p * n  # 6
    
    # Build the target tensor T as a dictionary of non-zero entries
    # T[a, b, c] = 1 if indices are consistent
    # a = i*m + j, b = j*p + k, c = k*n + i
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1.0

    def verify(U, V, W, tol=1e-6):
        rank = U.shape[1]
        # Check all target entries
        for (a, b, c), val in target.items():
            s = 0.0
            for r in range(rank):
                s += U[a, r] * V[b, r] * W[c, r]
            if abs(s - val) > tol:
                return False
        
        # Check that non-target entries are ~0
        for a in range(U_dim):
            for b in range(V_dim):
                for c in range(W_dim):
                    if (a, b, c) in target:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
        return True

    # Schoolbook decomposition (Rank 18)
    rank_school = n * m * p
    U_school = np.zeros((U_dim, rank_school))
    V_school = np.zeros((V_dim, rank_school))
    W_school = np.zeros((W_dim, rank_school))
    for idx, ((a, b, c), val) in enumerate(target.items()):
        U_school[a, idx] = val
        V_school[b, idx] = val
        W_school[c, idx] = val

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def als_decompose(target_rank, iterations=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(U_dim, target_rank)
        V = np.random.randn(V_dim, target_rank)
        W = np.random.randn(W_dim, target_rank)
        
        # Pre-compute target tensors as matrices for ALS
        # T_a[a, b*W_dim + c] = T[a, b, c]
        T_a = np.zeros((U_dim, V_dim * W_dim))
        # T_b[b, a*W_dim + c] = T[a, b, c]
        T_b = np.zeros((V_dim, U_dim * W_dim))
        # T_c[c, a*V_dim + b] = T[a, b, c]
        T_c = np.zeros((W_dim, U_dim * V_dim))
        
        for (a, b, c), val in target.items():
            T_a[a, b * W_dim + c] = val
            T_b[b, a * W_dim + c] = val
            T_c[c, a * V_dim + b] = val

        for _ in range(iterations):
            if time.time() > deadline:
                break
            
            # Update U
            # VW[b*W_dim+c, r] = V[b, r] * W[c, r]
            VW = np.zeros((V_dim * W_dim, target_rank))
            for r in range(target_rank):
                VW[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            try:
                U = T_a @ np.linalg.pinv(VW.T)
            except:
                continue
            
            # Update V
            # UW[a*W_dim+c, r] = U[a, r] * W[c, r]
            UW = np.zeros((U_dim * W_dim, target_rank))
            for r in range(target_rank):
                UW[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            try:
                V = T_b @ np.linalg.pinv(UW.T)
            except:
                continue
            
            # Update W
            # UV[a*V_dim+b, r] = U[a, r] * V[b, r]
            UV = np.zeros((U_dim * V_dim, target_rank))
            for r in range(target_rank):
                UV[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            try:
                W = T_c @ np.linalg.pinv(UV.T)
            except:
                continue

        return U, V, W

    def snap_and_verify(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if verify(U_s, V_s, W_s):
            return U_s, V_s, W_s, True
        return None, None, None, False

    # Search for lower ranks
    # Try 17, 16, 15...
    for target_rank in range(rank_school - 1, 10, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random restarts
        for seed in range(50):
            if time.time() > deadline:
                break
            
            U, V, W = als_decompose(target_rank, iterations=50, seed=seed)
            
            # Check if real solution is good
            # Compute error
            err = 0.0
            for (a, b, c), val in target.items():
                s = np.dot(U[a, :], V[b, :] * W[c, :])
                err += (s - val)**2
            for a in range(U_dim):
                for b in range(V_dim):
                    for c in range(W_dim):
                        if (a, b, c) in target:
                            continue
                        s = np.dot(U[a, :], V[b, :] * W[c, :])
                        err += s**2
                        
            if err < 1e-4:
                # Close to exact, try snapping
                Us, Vs, Ws, success = snap_and_verify(U, V, W)
                if success:
                    best_U, best_V, best_W = Us, Vs, Ws
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    def to_list(M):
        return M.tolist()
        
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
