# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build the target tensor T of shape (dim_A, dim_B, dim_C)
    T = np.zeros((dim_A, dim_B, dim_C))
    for i_n in range(n):
        for i_m in range(m):
            for k_p in range(p):
                i = i_n * m + i_m
                j = i_m * p + k_p
                k = i_n * p + k_p
                T[i, j, k] = 1.0
                
    def verify_decomposition(U, V, W, rank):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        return err
        
    def search_rank(rank, time_limit=100):
        start_time = time.time()
        rng = np.random.RandomState(42)
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Strategy: Random initialization with sparse structure or random signs
            # Often tensor decompositions for matrix multiplication have small integer entries
            if attempt % 2 == 0:
                # Random integers in {-1, 0, 1}
                U = rng.choice([-1, 0, 1], size=(dim_A, rank))
                V = rng.choice([-1, 0, 1], size=(dim_B, rank))
                W = rng.choice([-1, 0, 1], size=(dim_C, rank))
            else:
                # Random Gaussian
                U = rng.randn(dim_A, rank)
                V = rng.randn(dim_B, rank)
                W = rng.randn(dim_C, rank)
            
            # Normalize to avoid scaling issues
            if np.linalg.norm(U) > 0: U /= np.linalg.norm(U)
            if np.linalg.norm(V) > 0: V /= np.linalg.norm(V)
            if np.linalg.norm(W) > 0: W /= np.linalg.norm(W)
            
            prev_err = verify_decomposition(U, V, W, rank)
            
            # Alternating Least Squares
            for step in range(200):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U
                M = np.einsum('jr,kr->jkr', V, W).reshape(dim_B * dim_C, rank)
                T_flat = T.reshape(dim_A, dim_B * dim_C)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                    
                # Update V
                N = np.einsum('ir,kr->ikr', U, W).reshape(dim_A * dim_C, rank)
                T_flat_V = T.transpose(1, 0, 2).reshape(dim_B, dim_A * dim_C)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(N, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                    
                # Update W
                M = np.einsum('ir,jr->ijr', U, V).reshape(dim_A * dim_B, rank)
                T_flat_W = T.transpose(2, 0, 1).reshape(dim_C, dim_A * dim_B)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                    
                # Re-normalize to keep scale stable
                if np.linalg.norm(U) > 0: U /= np.linalg.norm(U)
                if np.linalg.norm(V) > 0: V /= np.linalg.norm(V)
                if np.linalg.norm(W) > 0: W /= np.linalg.norm(W)
                
                err = verify_decomposition(U, V, W, rank)
                if err < 1e-8:
                    return U, V, W
                if err > prev_err * 1.01:
                    break
                prev_err = err
                
        return None

    best_result = None
    # Try rank 12 first
    res = search_rank(12, time_limit=100)
    if res is not None:
        U, V, W = res
        U_list = [[U[i, r] for r in range(12)] for i in range(dim_A)]
        V_list = [[V[i, r] for r in range(12)] for i in range(dim_B)]
        W_list = [[W[i, r] for r in range(12)] for i in range(dim_C)]
        
        is_valid = True
        for mat in [U_list, V_list, W_list]:
            for row in mat:
                for val in row:
                    k = round(val * 2)
                    if abs(val - k * 0.5) > 1e-3:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if not is_valid:
                break
        
        if is_valid:
            U_list = [[round(val * 2) * 0.5 for val in row] for row in U_list]
            V_list = [[round(val * 2) * 0.5 for val in row] for row in V_list]
            W_list = [[round(val * 2) * 0.5 for val in row] for row in W_list]
            U_np = np.array(U_list)
            V_np = np.array(V_list)
            W_np = np.array(W_list)
            if verify_decomposition(U_np, V_np, W_np, 12) < 1e-6:
                return {"U": U_list, "V": V_list, "W": W_list}

    # Try rank 13
    res = search_rank(13, time_limit=30)
    if res is not None:
        U, V, W = res
        U_list = [[U[i, r] for r in range(13)] for i in range(dim_A)]
        V_list = [[V[i, r] for r in range(13)] for i in range(dim_B)]
        W_list = [[W[i, r] for r in range(13)] for i in range(dim_C)]
        
        is_valid = True
        for mat in [U_list, V_list, W_list]:
            for row in mat:
                for val in row:
                    k = round(val * 2)
                    if abs(val - k * 0.5) > 1e-3:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if not is_valid:
                break
        
        if is_valid:
            U_list = [[round(val * 2) * 0.5 for val in row] for row in U_list]
            V_list = [[round(val * 2) * 0.5 for val in row] for row in V_list]
            W_list = [[round(val * 2) * 0.5 for val in row] for row in W_list]
            U_np = np.array(U_list)
            V_np = np.array(V_list)
            W_np = np.array(W_list)
            if verify_decomposition(U_np, V_np, W_np, 13) < 1e-6:
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to rank 14
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for B1 (cols 0,1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Strassen for B2 (cols 2,3)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
