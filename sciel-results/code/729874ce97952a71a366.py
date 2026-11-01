import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 34
    # The multiplication tensor T[i, j, k] = 1 if (i,j) in A and (j,k) in B
    # Actually T is defined by the indices.
    # T is a 3-way tensor of size n x m x p.
    # T[i, j, k] = 1 if i in [0, n), j in [0, m), k in [0, p).
    # Wait, the tensor for matrix multiplication is:
    # T[i, j, k] = 1 if the operation multiplies A[i, j] and B[j, k] to contribute to C[i, k].
    # So T[i, j, k] is 1 for all valid i, j, k.
    # The decomposition is T = sum_c U[:, c] \otimes V[:, c] \otimes W[:, c].
    # U has shape (n*m, rank). Index i*m+j.
    # V has shape (m*p, rank). Index j*p+k.
    # W has shape (p*n, rank). Index k*n+i.
    
    # The target tensor T has non-zero entries at:
    # For each i, j, k:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # Value: 1
    
    # We want to find U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
    
    # This is a system of equations.
    # Number of equations: n*m*p = 48.
    # Number of variables: rank * (n*m + m*p + p*n) = 34 * (8 + 24 + 12) = 34 * 44 = 1496.
    # Under-determined. We can find a solution.
    
    # We can set up a linear system if we fix two matrices?
    # No, it's trilinear.
    
    # Strategy: Random initialization and ALS (Alternating Least Squares)
    # or simply solving for one matrix given the others?
    # If we fix V and W, the equation for U is:
    # sum_c U[idx_u, c] * (V[idx_v, c] * W[idx_w, c]) = 1
    # This is linear in U.
    # Let M_u be a matrix of size (n*m*p) x (n*m * rank)? No.
    # For each (i, j, k), we have an equation.
    # Let's flatten the tensor to a vector y of size 48.
    # y[t] = 1 for all t.
    # The model is y[t] = sum_c U[i_t*m+j_t, c] * V[j_t*p+k_t, c] * W[k_t*n+i_t, c].
    
    # This is not linear in U, V, W simultaneously.
    # But if we fix V and W, it is linear in U.
    # Let A_u be a matrix where row t corresponds to (i, j, k) and column c' corresponds to (u_idx, c).
    # A_u[t, u_idx * rank + c] = V[j_t*p+k_t, c] * W[k_t*n+i_t, c]
    # Then y = A_u * vec(U).
    # We can solve for vec(U) using least squares.
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_result = None
    best_error = float('inf')
    
    # Try a few random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize V, W randomly
        # Using small integers or half-integers might help convergence to rational solutions
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Iterate ALS
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Solve for U
            # Construct A_u
            # Rows: n*m*p
            # Cols: (n*m) * rank
            A_u = np.zeros((n * m * p, n * m * target_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t = i * m * p + j * p + k
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        # For each rank component c
                        for c in range(target_rank):
                            val = V[v_idx, c] * W[w_idx, c]
                            A_u[t, u_idx * target_rank + c] = val
                            
            # Solve A_u * vec(U) = y (where y is all 1s)
            # Use lstsq
            y = np.ones(n * m * p)
            try:
                U_flat, residuals, rank_u, s = np.linalg.lstsq(A_u, y, rcond=None)
            except:
                continue
                
            U = U_flat.reshape(n * m, target_rank)
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        prod = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        error += (prod - 1.0) ** 2
                        
            if error < 1e-6:
                # Found a solution
                # Round to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify rounded solution
                error_r = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            prod = np.dot(U_int[u_idx, :], V_int[v_idx, :] * W_int[w_idx, :])
                            error_r += (prod - 1.0) ** 2
                            
                if error_r < 1e-9:
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                else:
                    # If rounding fails, keep the float solution? 
                    # The prompt says "integer/half-integer entries".
                    # If I return floats, it might be invalid.
                    # I will try to find a solution that is naturally close to half-integers.
                    pass
                    
            # Solve for V
            A_v = np.zeros((n * m * p, m * p * target_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t = i * m * p + j * p + k
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        for c in range(target_rank):
                            val = U[u_idx, c] * W[w_idx, c]
                            A_v[t, v_idx * target_rank + c] = val
                            
            try:
                V_flat, residuals, rank_v, s = np.linalg.lstsq(A_v, y, rcond=None)
            except:
                continue
            V = V_flat.reshape(m * p, target_rank)
            
            # Solve for W
            A_w = np.zeros((n * m * p, p * n * target_rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t = i * m * p + j * p + k
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        for c in range(target_rank):
                            val = U[u_idx, c] * V[v_idx, c]
                            A_w[t, w_idx * target_rank + c] = val
                            
            try:
                W_flat, residuals, rank_w, s = np.linalg.lstsq(A_w, y, rcond=None)
            except:
                continue
            W = W_flat.reshape(p * n, target_rank)

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
