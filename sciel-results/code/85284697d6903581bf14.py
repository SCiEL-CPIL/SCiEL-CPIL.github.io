# EVOLVE-BLOCK-START

def _build_expected_tensor(n, m, p):
    """Build the expected matmul tensor T[x, y, z] where:
    x = i*m+j (A index)
    y = j*p+k (B index)
    z = k*n+i (C index, column-major convention)
    T[x,y,z] = 1 if indices are consistent, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[x,r]*V[y,r]*W[z,r] equals the matmul tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Compute decomposed tensor
    D = np.einsum('xr,yr,zr->xyz', U_arr, V_arr, W_arr)
    
    # Build expected
    expected = _build_expected_tensor(n, m, p)
    
    return np.allclose(D, expected, atol=tol)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                U[x, col] = 1.0
                V[y, col] = 1.0
                W[z, col] = 1.0
                col += 1
    
    return U, V, W

def _try_als_decomposition(n, m, p, target_rank, seed=0, max_iters=100, deadline=None):
    """
    Try to find a rank-`target_rank` decomposition using Vectorized Alternating Least Squares.
    """
    np.random.seed(seed)
    
    N, M, P = n * m, m * p, p * n
    R = target_rank
    
    U = np.random.randn(N, R) * 0.1
    V = np.random.randn(M, R) * 0.1
    W = np.random.randn(P, R) * 0.1
    
    expected = _build_expected_tensor(n, m, p)
    
    # Pre-compute flattened expected tensor slices for faster access if needed, 
    # but we will use the 3D array directly with einsum or reshaping.
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            return None
        
        # 1. Solve for U
        # T_x = sum_r U[x,r] * (V[:,r] outer W[:,r])
        # vec(T_x) = sum_r U[x,r] * vec(V[:,r] outer W[:,r])
        # Let A_VW be a matrix of shape (M*P, R) where A_VW[idx, r] = V[y,r]*W[z,r]
        A_VW = np.einsum('yr,zr->yzr', V, W).reshape(M * P, R)
        
        # B_U is a matrix of shape (M*P, N) where B_U[idx, x] = T[x, y, z]
        # T is (N, M, P). We need to transpose to (M, P, N) then reshape to (M*P, N)
        B_U = expected.transpose(1, 2, 0).reshape(M * P, N)
        
        # Solve A_VW @ U.T = B_U
        # U.T is (R, N). A_VW is (M*P, R). B_U is (M*P, N).
        # This is a least squares problem: min || A_VW @ X - B_U || where X = U.T
        try:
            U.T, _, _, _ = np.linalg.lstsq(A_VW, B_U, rcond=None)
        except:
            pass

        # 2. Solve for V
        # T_y = sum_r V[y,r] * (U[:,r] outer W[:,r])
        # vec(T_y) = sum_r V[y,r] * vec(U[:,r] outer W[:,r])
        # A_UW shape (N*P, R)
        A_UW = np.einsum('xr,zr->x zr', U, W).reshape(N * P, R)
        
        # B_V shape (N*P, M) where B_V[idx, y] = T[x, y, z]
        # T is (N, M, P). Transpose to (N, P, M) then reshape to (N*P, M)
        B_V = expected.transpose(0, 2, 1).reshape(N * P, M)
        
        # Solve A_UW @ V.T = B_V
        try:
            V.T, _, _, _ = np.linalg.lstsq(A_UW, B_V, rcond=None)
        except:
            pass

        # 3. Solve for W
        # T_z = sum_r W[z,r] * (U[:,r] outer V[:,r])
        # vec(T_z) = sum_r W[z,r] * vec(U[:,r] outer V[:,r])
        # A_UV shape (N*M, R)
        A_UV = np.einsum('xr,yr->xyr', U, V).reshape(N * M, R)
        
        # B_W shape (N*M, P) where B_W[idx, z] = T[x, y, z]
        # T is (N, M, P). Transpose to (N, M, P) is already N*M*P?
        # We need index (x,y) to correspond to row, and z to column.
        # T is (N, M, P). Reshape to (N*M, P) directly.
        B_W = expected.reshape(N * M, P)
        
        # Solve A_UV @ W.T = B_W
        try:
            W.T, _, _, _ = np.linalg.lstsq(A_UV, B_W, rcond=None)
        except:
            pass

        # Check convergence
        if it % 10 == 0:
            D = np.einsum('xr,yr,zr->xyz', U, V, W)
            err = np.linalg.norm(D - expected)
            if err < 1e-8:
                return U, V, W
    
    # Final check
    D = np.einsum('xr,yr,zr->xyz', U, V, W)
    if np.allclose(D, expected, atol=1e-8):
        return U, V, W
    
    return None

def _try_sparse_search(n, m, p, target_rank, deadline):
    """
    Try a structured search for lower rank decompositions.
    """
    for r in range(target_rank, 55, -1):
        if time.time() > deadline - 10:
            break
        
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            result = _try_als_decomposition(n, m, p, r, seed=seed, max_iters=50, deadline=deadline)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return U, V, W
    
    return None

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 3x4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # We'll try a few candidate ranks below schoolbook
    # If rank is 60, this loop won't find anything better.
    # If rank is < 60, we hope to find it.
    for target_rank in range(best_rank - 1, max(best_rank - 10, 40), -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_sparse_search(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = np.round(best_U * 2).astype(int).tolist()
    V_list = np.round(best_V * 2).astype(int).tolist()
    W_list = np.round(best_W * 2).astype(int).tolist()
    
    U_list = [[val / 2.0 for val in row] for row in U_list]
    V_list = [[val / 2.0 for val in row] for row in V_list]
    W_list = [[val / 2.0 for val in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
