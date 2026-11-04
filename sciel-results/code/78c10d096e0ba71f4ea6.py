import numpy as np
import time

def construct():
    start_time = time.time()
    max_time = 140  # Safety margin
    
    n, m, p = 2, 3, 4
    
    num_U = n * m
    num_V = m * p
    num_W = p * n
    
    T = np.zeros((num_U, num_V, num_W))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def verify_decomposition(U, V, W, R):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)

    def try_rank(R):
        # Try to find rank R decomposition using ALS
        # We use multiple restarts
        max_restarts = 10
        max_iters = 30
        
        for restart in range(max_restarts):
            if time.time() - start_time > max_time - 10:
                return None
                
            U = np.random.uniform(-1, 1, (num_U, R))
            V = np.random.uniform(-1, 1, (num_V, R))
            W = np.random.uniform(-1, 1, (num_W, R))
            
            for _ in range(max_iters):
                if time.time() - start_time > max_time - 5:
                    return None

                # Update U
                # T[i, J, K] = sum_r U[i, r] V[J, r] W[K, r]
                # For fixed i, we solve for U[i, :]
                # b = T[i, :, :].flatten()
                # M[r, (J, K)] = V[J, r] * W[K, r]
                for i in range(num_U):
                    # Construct M of shape (R, num_V * num_W)
                    # M[r, J*num_W + K] = V[J, r] * W[K, r]
                    # This is the sum of outer products? No, for each r, it's a specific matrix.
                    # We can construct M by iterating r or using broadcasting.
                    # V[:, r] is (num_V,). W[:, r] is (num_W,).
                    # outer(V[:, r], W[:, r]) is (num_V, num_W).
                    # We want M[r, :] = outer(V[:, r], W[:, r]).flatten()
                    
                    # Efficient way:
                    # V_col = V.T  # (R, num_V)
                    # W_col = W.T  # (R, num_W)
                    # M = (V_col[:, :, None] * W_col[:, None, :]).reshape(R, -1)
                    # Check shapes:
                    # V_col[:, :, None] -> (R, num_V, 1)
                    # W_col[:, None, :] -> (R, 1, num_W)
                    # Product -> (R, num_V, num_W)
                    # Reshape -> (R, num_V * num_W)
                    # This matches M[r, J*num_W + K] = V_col[r, J] * W_col[r, K] = V[J, r] * W[K, r]. Correct.
                    
                    M = (V.T[:, :, None] * W.T[:, None, :]).reshape(R, -1)
                    b = T[i].flatten()
                    try:
                        U[i, :], residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        return None

                # Update V
                # T[I, j, K] = sum_r U[I, r] V[j, r] W[K, r]
                # For fixed j, solve for V[j, :]
                # b = T[:, j, :].flatten()
                # M[r, (I, K)] = U[I, r] * W[K, r]
                for j in range(num_V):
                    M = (U.T[:, :, None] * W.T[:, None, :]).reshape(R, -1)
                    b = T[:, j, :].flatten()
                    try:
                        V[j, :], residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        return None

                # Update W
                # T[I, J, k] = sum_r U[I, r] V[J, r] W[k, r]
                # For fixed k, solve for W[k, :]
                # b = T[:, :, k].flatten()
                # M[r, (I, J)] = U[I, r] * V[J, r]
                for k in range(num_W):
                    M = (U.T[:, :, None] * V.T[:, None, :]).reshape(R, -1)
                    b = T[:, :, k].flatten()
                    try:
                        W[k, :], residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                    except:
                        return None

            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify_decomposition(U_snap, V_snap, W_snap, R):
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
        
        return None

    # Try ranks from 1 upwards
    for R in range(1, 25):
        if time.time() - start_time > max_time:
            break
        dec = try_rank(R)
        if dec and verify_decomposition(np.array(dec["U"]), np.array(dec["V"]), np.array(dec["W"]), R):
            return dec
    
    # Fallback to schoolbook rank 24
    rank = 24
    U = [[0] * rank for _ in range(num_U)]
    V = [[0] * rank for _ in range(num_V)]
    W = [[0] * rank for _ in range(num_W)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                U[I][column] = 1
                V[J][column] = 1
                W[K][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
