import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = delta_{i, i'} delta_{j, j'} ... 
    # Actually T is defined by the multiplication rule:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i, j, k] is 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    # In the tensor decomposition C = sum A_k * B_k * W_k, 
    # U[i*m+j, k] corresponds to A[i,j]
    # V[j*p+k, k] corresponds to B[j,k]
    # W[k*n+i, k] corresponds to C[i,k]
    
    # Let's construct the target tensor in the flattened index space.
    # U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # The product is sum_r U[:,r] V[:,r] W[:,r]
    # This must equal the tensor M where M[i, j, k] = 1 if ...?
    # Actually, the tensor T for matrix multiplication is:
    # T[i, j, k] = 1 if i is row of A, j is col of A/row of B, k is col of B.
    # But the output is indexed by i (row A) and k (col B).
    # So T is a 3-tensor with indices (i, j, k) and value 1?
    # No, the decomposition is:
    # sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This sum must equal the coefficient of A[i,j] B[j,k] in C[i,k].
    # For a fixed (i,j,k), the term A[i,j] B[j,k] appears exactly once in C[i,k].
    # So the sum must be 1 for all valid (i,j,k).
    
    # So the target "tensor" is just the identity mapping in the sense that
    # for each triplet (i,j,k), the sum is 1.
    
    def objective(params):
        r = 20 # Try rank 20
        # Reshape params
        # U: n*m * r
        # V: m*p * r
        # W: p*n * r
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        u_flat = params[:size_u * r]
        v_flat = params[size_u * r : size_u * r + size_v * r]
        w_flat = params[size_u * r + size_v * r :]
        
        U = u_flat.reshape(size_u, r)
        V = v_flat.reshape(size_v, r)
        W = w_flat.reshape(size_w, r)
        
        # Calculate the reconstructed tensor
        # Rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want Rec[i, j, k] == 1 for all i,j,k.
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    error += (val - 1.0) ** 2
        return error

    # Initial guess
    # Start with schoolbook rank 24, then randomly project to rank 20?
    # Or just random init.
    
    best_params = None
    best_err = float('inf')
    
    for seed in range(10):
        np.random.seed(seed)
        # Initialize with small random values
        init_u = np.random.randn(n * m * 20) * 0.1
        init_v = np.random.randn(m * p * 20) * 0.1
        init_w = np.random.randn(p * n * 20) * 0.1
        params = np.concatenate([init_u, init_v, init_w])
        
        res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x
            
        if best_err < 1e-8:
            break
            
    if best_err < 1e-8:
        r = 20
        u_flat = best_params[:n * m * r]
        v_flat = best_params[n * m * r : n * m * r + m * p * r]
        w_flat = best_params[n * m * r + m * p * r :]
        
        U = u_flat.reshape(n * m, r)
        V = v_flat.reshape(m * p, r)
        W = w_flat.reshape(p * n, r)
        
        # Round to nearest half-integer or integer
        # Multiply by 2, round, divide by 2
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook if optimization fails
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
