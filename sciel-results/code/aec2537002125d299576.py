import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Helper to check if a decomposition is valid
    def check_validity_residual(U, V, W, tol=1e-6):
        R = U.shape[1]
        # Vectorized check: build the full multiplication tensor and compare
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 if i==k else 0  (since A is n x m, B is m x p, C = A@B is n x p)
        # Actually: C[i,k] = sum_j A[i,j]*B[j,k], so the tensor is T[i,j,k] = delta_{i,k}
        # Let's verify: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1 if i==k else 0
        
        # Efficient: for each r, outer product of U[:,r], V[:,r], W[:,r]
        # But that's R tensors of size n*m*p = 60, which is fine
        err = 0.0
        for r in range(R):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            # Contribution to C[i,k] from this rank: sum_j u_col[i,j] * v_col[j,k] * w_col[k,i]
            # = (u_col @ v_col) element-wise * w_col, then sum over k? No.
            # C_contrib[i,k] = sum_j u_col[i,j] * v_col[j,k] * w_col[k,i]
            # This is not a standard matmul. Let's do it explicitly.
            uv = u_col @ v_col  # (n, p)
            # C_contrib[i,k] = uv[i,k] * w_col[k,i]
            c_contrib = uv * w_col.T  # (n, p), element [i,k] = uv[i,k]*w_col[k,i]
            # Target: identity-like, C[i,k] = 1 if i==k else 0
            target = np.eye(n, p)  # n x p, 1 on diagonal
            err += np.sum((c_contrib - target) ** 2)
        return err

    def get_rank_55_solution():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # Part A: j=0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
            
        # Part B: j=1,2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
            
        return U, V, W

    U, V, W = get_rank_55_solution()
    
    # Verify validity
    err = check_validity_residual(U, V, W)
    if err > 1e-6:
        # Fallback to schoolbook if construction failed (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 55
    best_U, best_V, best_W = U, V, W
    
    # Try to find a rank-54 solution via random search in spare time
    if time.time() < deadline - 20:
        # Attempt a few random rank-54 constructions
        # Strategy: start from rank-55, try to merge two columns
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            # Random perturbation approach: try to find a rank-54 by
            # randomly modifying the rank-55 solution and checking if it still works
            # with one column zeroed out or merged
            
            # Alternative: direct random search for rank 54
            # This is unlikely to succeed quickly, so we do a quick check
            # by trying to nullify one column via linear combination
            if attempt == 0:
                # Try: can we express one column as a combination of others?
                # Check column rank of U, V, W simultaneously
                pass  # Too complex for quick check; skip
        
        # Quick random search: generate random rank-54 tensors and check
        # This is very unlikely to succeed in the time budget, so we skip
        # unless we have lots of time
        if time.time() < deadline - 60:
            np.random.seed(42)
            for trial in range(20):
                if time.time() > deadline - 60:
                    break
                r = 54
                U_try = np.random.randn(n*m, r)
                V_try = np.random.randn(m*p, r)
                W_try = np.random.randn(p*n, r)
                # Scale to make entries small integers/half-integers is hard
                # Just check if residual is small
                err_try = check_validity_residual(U_try, V_try, W_try)
                if err_try < 1e-6:
                    best_rank = 54
                    best_U, best_V, best_W = U_try, V_try, W_try
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
