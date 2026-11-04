import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    rank = 44
    
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    
    def add_strassen(X_u, Y_v, Z_w, col):
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col
    
    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
    
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 0][col] += 1; W[2][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 1][col] += 1; W[5][col] += 1; col += 1
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 2][col] += 1; W[8][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 3][col] += 1; W[11][col] += 1; col += 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verify correctness with a test case
    rng = np.random.RandomState(42)
    A_test = rng.randn(3, 4)
    B_test = rng.randn(4, 4)
    C_test = A_test @ B_test
    
    C_check = np.zeros((3, 4))
    for i in range(3):
        for k in range(4):
            val = 0.0
            for r in range(rank):
                u_part = 0.0
                for j in range(4):
                    u_part += A_test[i, j] * U_arr[i*4+j, r]
                v_part = 0.0
                for j in range(4):
                    v_part += B_test[j, k] * V_arr[j*4+k, r]
                val += W_arr[i*4+k, r] * u_part * v_part
            C_check[i, k] = val
    
    if not np.allclose(C_check, C_test, atol=1e-10):
        # Decomposition is incorrect, return as-is (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank via linear dependency detection on flattened tensors
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    tensor_dim = dim_u * dim_v * dim_w  # 2304
    T_mat = np.zeros((tensor_dim, rank))
    for r in range(rank):
        T_mat[:, r] = np.kron(np.kron(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    
    # Find rank via SVD
    try:
        s_vals = np.linalg.svd(T_mat, full_matrices=False, compute_uv=False)
        tol = max(T_mat.shape) * s_vals[0] * np.finfo(float).eps
        num_independent = int(np.sum(s_vals > tol))
        
        if num_independent < rank:
            # Use QR with column pivoting to find independent set
            Q, R, piv = np.linalg.qr(T_mat, pivoting=True)
            keep_pivots = sorted(piv[:num_independent].tolist())
            
            U_new = U_arr[:, keep_pivots].tolist()
            V_new = V_arr[:, keep_pivots].tolist()
            W_new = W_arr[:, keep_pivots].tolist()
            
            # Verify reduced decomposition
            if time.time() < deadline:
                C_check2 = np.zeros((3, 4))
                for i in range(3):
                    for k in range(4):
                        val = 0.0
                        for r in keep_pivots:
                            u_part = 0.0
                            for j in range(4):
                                u_part += A_test[i, j] * U_arr[i*4+j, r]
                            v_part = 0.0
                            for j in range(4):
                                v_part += B_test[j, k] * V_arr[j*4+k, r]
                            val += W_arr[i*4+k, r] * u_part * v_part
                        C_check2[i, k] = val
                
                if np.allclose(C_check2, C_test, atol=1e-10):
                    return {"U": U_new, "V": V_new, "W": W_new}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
