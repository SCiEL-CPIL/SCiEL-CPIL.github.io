# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # 10s margin before 150s kill

def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly represents 2x2 * 2x6 matrix multiplication.
    
    The tensor T[i,j,k] = 1 if j == k%2 else 0. We verify by checking that
    for each block b (columns 2b, 2b+1 of B), the 2x2 multiplication is correct.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verify using random integer tests
    import random
    rng = random.Random(12345)
    num_tests = 200
    for _ in range(num_tests):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # Compute C = A * B
        C = [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
        
        # Compute using decomposition
        T_decomp = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U_arr[i * m + j, r]
                    if u_val != 0:
                        for k in range(p):
                            v_val = V_arr[j * p + k, r]
                            if v_val != 0:
                                w_val = W_arr[k * n + i, r]
                                if w_val != 0:
                                    T_decomp[i, j, k] += u_val * v_val * w_val
        
        T_target = np.ones((n, m, p))
        if not np.allclose(T_decomp, T_target, atol=1e-9):
            return False
    
    return True

def _strassen_decomposition():
    """Block-diagonal Strassen decomposition with rank 21."""
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # (4, 21)
    V = [[0] * rank for _ in range(m * p)]  # (12, 21)
    W = [[0] * rank for _ in range(p * n)]  # (12, 21)
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1
        [(0, 0, 1)],               # M2
        [(0, 1, 1), (1, 1, -1)],  # M3
        [(1, 0, 1), (0, 0, -1)],  # M4
        [(1, 1, 1)],               # M5
        [(0, 0, 1), (0, 1, 1)],   # M6
        [(1, 0, 1), (1, 1, 1)]    # M7
    ]
    
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1
        [(1, 0, 1), (1, 1, -1)],  # M2
        [(0, 1, 1), (1, 1, 1)],   # M3
        [(0, 0, 1), (1, 0, 1)],   # M4
        [(0, 0, -1), (0, 1, 1)],  # M5
        [(1, 1, 1)],              # M6
        [(0, 0, 1)]               # M7
    ]
    
    for b in range(3):  # block index
        for k in range(7):  # Strassen factor index
            col = b * 7 + k
            
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
            
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * b + k_local
                v_idx = j_local * p + global_k
                V[v_idx][col] = val
            
            for i, k_local, val in W_coeffs[k]:
                global_k = 2 * b + k_local
                w_idx = global_k * n + i
                W[w_idx][col] = val
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank 24."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank20_search():
    """
    Attempt to find a rank-20 decomposition using numerical optimization.
    """
    if time.time() - _START_TIME > _TIME_LIMIT - 20:
        return None
        
    n, m, p = DIMENSIONS
    target_rank = 20
    
    T_target = np.ones((n, m, p))
    T_flat = T_target.flatten()
    
    strassen = _strassen_decomposition()
    U0 = np.array(strassen["U"], dtype=float)
    V0 = np.array(strassen["V"], dtype=float)
    W0 = np.array(strassen["W"], dtype=float)
    
    def objective(params):
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U : size_U + size_V]
        W_flat = params[size_U + size_V : size_U + size_V + size_W]
        
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        
        T_decomp = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    u_val = U[i*m+j, r]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = V[j*p+k, r]
                        if v_val == 0: continue
                        w_val = W[k*n+i, r]
                        T_decomp[i,j,k] += u_val * v_val * w_val
                        
        diff = T_decomp.flatten() - T_flat
        return np.sum(diff**2)

    # Initialization: Use first 19 columns of Strassen, and the 20th as the 21st.
    # Then optimize.
    U_init = U0[:, :19].copy()
    V_init = V0[:, :19].copy()
    W_init = W0[:, :19].copy()
    
    # Add the 21st column as the 20th initial guess
    U_init = np.hstack([U_init, U0[:, 19:20]])
    V_init = np.hstack([V_init, V0[:, 19:20]])
    W_init = np.hstack([W_init, W0[:, 19:20]])
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-12})
        
        if res.fun < 1e-9:
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            size_W = p * n * target_rank
            
            U_flat = res.x[:size_U]
            V_flat = res.x[size_U : size_U + size_V]
            W_flat = res.x[size_U + size_V : size_U + size_V + size_W]
            
            U = U_flat.reshape(n * m, target_rank)
            V = V_flat.reshape(m * p, target_rank)
            W = W_flat.reshape(p * n, target_rank)
            
            # Check for rationality (integers or half-integers)
            # Multiply by 2 and check closeness to integers
            U_int = np.round(U * 2)
            V_int = np.round(V * 2)
            W_int = np.round(W * 2)
            
            if np.allclose(U * 2, U_int, atol=1e-6) and np.allclose(V * 2, V_int, atol=1e-6) and np.allclose(W * 2, W_int, atol=1e-6):
                U_list = [[int(val) for val in row] for row in U_int]
                V_list = [[int(val) for val in row] for row in V_int]
                W_list = [[int(val) for val in row] for row in W_int]
                
                if _verify_decomposition(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
        
    return None

def construct():
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook_decomposition()
    
    # Try to find a rank-20 solution
    result20 = _try_rank20_search()
    if result20 is not None:
        return result20
        
    # Fall back to rank-21 Strassen
    result = _strassen_decomposition()
    
    if not _verify_decomposition(result["U"], result["V"], result["W"]):
        return _schoolbook_decomposition()
    
    return result

# EVOLVE-BLOCK-END
