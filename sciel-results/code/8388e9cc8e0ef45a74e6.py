import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 150s limit

    def build_rank44():
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

        col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)
        col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
        col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)

        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1

        return U, V, W

    def verify(U, V, W, n, m, p):
        """Verify exact decomposition using integer arithmetic (scaled by 2)."""
        U_int = np.round(np.array(U) * 2).astype(np.int64)
        V_int = np.round(np.array(V) * 2).astype(np.int64)
        W_int = np.round(np.array(W) * 2).astype(np.int64)

        # Compute tensor product: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Expected: 4 for each (i,j,k) in the multiplication pattern, 0 otherwise
        # Scaled by 8: expected 8 for each term
        
        # Build expected tensor
        T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 8

        # Compute actual tensor
        T_actual = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)

        return np.array_equal(T_actual, T_expected)

    # Build the rank-44 decomposition
    U, V, W = build_rank44()
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # First verify the base decomposition is correct
    if not verify(U_np, V_np, W_np, n, m, p):
        # Fallback to schoolbook rank 48
        rank = 48
        U = [[0]*rank for _ in range(dim_u)]
        V = [[0]*rank for _ in range(dim_v)]
        W = [[0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank by checking linear dependence and removing redundant columns
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = 44

    # Form the combined matrix M where column r is [U[:,r], V[:,r], W[:,r]]
    def build_M(U_mat, V_mat, W_mat):
        return np.vstack([U_mat, V_mat, W_mat])

    # Iteratively try to remove columns that are linearly dependent
    # Use SVD to identify rank deficiency, then try removing one column at a time
    removed_count = 0
    max_removal_attempts = 50  # Limit total removal attempts

    for attempt in range(max_removal_attempts):
        if time.time() > deadline:
            break
        
        if current_rank <= 1:
            break

        M = build_M(current_U, current_V, current_W)
        
        # Check if any column is in the span of the others
        # Try removing each column and checking if the tensor still decomposes
        # But this is expensive, so use a heuristic: check SVD rank first
        try:
            s_vals = np.linalg.svd(M, full_matrices=False, compute_uv=False)
            tol = max(M.shape) * np.finfo(float).eps * (s_vals[0] if len(s_vals) > 0 else 1)
            num_rank = int(np.sum(s_vals > tol))
        except:
            num_rank = current_rank

        if num_rank >= current_rank:
            break  # No more linearly dependent columns

        # Try to find a column to remove
        found = False
        for r_idx in range(current_rank):
            if time.time() > deadline:
                break
            
            # Try removing column r_idx
            keep_indices = [i for i in range(current_rank) if i != r_idx]
            
            U_test = current_U[:, keep_indices]
            V_test = current_V[:, keep_indices]
            W_test = current_W[:, keep_indices]
            
            if verify(U_test, V_test, W_test, n, m, p):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                removed_count += 1
                found = True
                break
        
        if not found:
            break

    # Final verification
    if current_rank < 44:
        if verify(current_U, current_V, current_W, n, m, p):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

    return {"U": U, "V": V, "W": W}
