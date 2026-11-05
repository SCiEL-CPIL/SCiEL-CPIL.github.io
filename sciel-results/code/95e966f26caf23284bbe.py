<<<<<<< SEARCH
    def try_als(rank, max_iter=300, tol=1e-12, seed=42):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = np.zeros((p * n, rank))
        
        for iteration in range(max_iter):
            if time.time() - start_time > max_time - 5:
                return None
            
            # Solve for W: fix U, V
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U: fix V, W
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V: fix U, W
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                U_list = [[round(float(x), 10) for x in row] for row in U]
                V_list = [[round(float(x), 10) for x in row] for row in V]
                W_list = [[round(float(x), 10) for x in row] for row in W]
                if verify(U_list, V_list, W_list, rank):
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    U_list = [[round(float(x), 10) for x in row] for row in U_snap]
                    V_list = [[round(float(x), 10) for x in row] for row in V_snap]
                    W_list = [[round(float(x), 10) for x in row] for row in W_snap]
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list
                    else:
                        return U_list, V_list, W_list
        return None
=======
    def try_als(rank, max_iter=500, tol=1e-14, seed=42):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        # Try multiple initialization strategies
        init_scales = [0.1, 0.5, 1.0, 2.0]
        for scale in init_scales:
            if time.time() - start_time > max_time - 5:
                return None
            
            U = rng.randn(n * m, rank) * scale
            V = rng.randn(m * p, rank) * scale
            W = np.zeros((p * n, rank))
            
            for iteration in range(max_iter):
                if time.time() - start_time > max_time - 5:
                    break
                
                # Solve for W: fix U, V
                W_new = np.zeros((p * n, rank))
                for k in range(p):
                    for i in range(n):
                        w = k * n + i
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b = np.ones(m)
                        try:
                            W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        except np.linalg.LinAlgError:
                            W_new[w, :] = rng.randn(rank) * 0.1
                
                # Solve for U: fix V, W
                U_new = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        u = i * m + j
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b = np.ones(p)
                        try:
                            U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        except np.linalg.LinAlgError:
                            U_new[u, :] = rng.randn(rank) * 0.1
                
                # Solve for V: fix U, W
                V_new = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        v = j * p + k
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b = np.ones(n)
                        try:
                            V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        except np.linalg.LinAlgError:
                            V_new[v, :] = rng.randn(rank) * 0.1
                
                diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
                W, U, V = W_new, U_new, V_new
                
                if diff < tol:
                    # High-precision polish: run additional iterations with tighter tolerance
                    for polish_iter in range(100):
                        if time.time() - start_time > max_time - 5:
                            break
                        W_new = np.zeros((p * n, rank))
                        for k in range(p):
                            for i in range(n):
                                w = k * n + i
                                A = np.zeros((m, rank))
                                for j in range(m):
                                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                                b = np.ones(m)
                                try:
                                    W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                                except np.linalg.LinAlgError:
                                    W_new[w, :] = W[w, :]
                        
                        U_new = np.zeros((n * m, rank))
                        for i in range(n):
                            for j in range(m):
                                u = i * m + j
                                A = np.zeros((p, rank))
                                for k in range(p):
                                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                                b = np.ones(p)
                                try:
                                    U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                                except np.linalg.LinAlgError:
                                    U_new[u, :] = U[u, :]
                        
                        V_new = np.zeros((m * p, rank))
                        for j in range(m):
                            for k in range(p):
                                v = j * p + k
                                A = np.zeros((n, rank))
                                for i in range(n):
                                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                                b = np.ones(n)
                                try:
                                    V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                                except np.linalg.LinAlgError:
                                    V_new[v, :] = V[v, :]
                        
                        diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
                        W, U, V = W_new, U_new, V_new
                        if diff < 1e-16:
                            break
                    
                    U_list = [[round(float(x), 12) for x in row] for row in U]
                    V_list = [[round(float(x), 12) for x in row] for row in V]
                    W_list = [[round(float(x), 12) for x in row] for row in W]
                    if verify(U_list, V_list, W_list, rank):
                        U_snap = snap_to_half_integer(U)
                        V_snap = snap_to_half_integer(V)
                        W_snap = snap_to_half_integer(W)
                        U_list = [[round(float(x), 12) for x in row] for row in U_snap]
                        V_list = [[round(float(x), 12) for x in row] for row in V_snap]
                        W_list = [[round(float(x), 12) for x in row] for row in W_snap]
                        if verify(U_list, V_list, W_list, rank):
                            return U_list, V_list, W_list
                        else:
                            return U_list, V_list, W_list
        return None
>>>>>>> REPLACE

