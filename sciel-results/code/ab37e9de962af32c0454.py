import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Lower bound for rank is max(n*m, m*p, p*n)
    min_rank = max(n * m, m * p, p * n)
    
    # We try ranks starting from the lower bound upwards
    # Given time constraints, we try a few small ranks
    base_rank = min_rank
    
    for rank_offset in range(0, 20): # Try ranks from min_rank to min_rank + 19
        R = base_rank + rank_offset
        
        # Initialize with random values
        np.random.seed(42 + rank_offset) # Reproducibility
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Target tensor T of shape (n*m, m*p, p*n)
        # The equation is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        # We can reshape U, V, W to facilitate vectorization.
        
        # Let's use alternating least squares (ALS).
        # Fix V, W -> Solve for U.
        # Fix U, W -> Solve for V.
        # Fix U, V -> Solve for W.
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(100): # 100 iterations of ALS
            # Solve for U
            # For each (i,j), we need sum_r U[i,j,r] * (V[:, j, :] * W[:, i, :])_r = 1
            # This is a linear system for U[i,j, :]
            # Let A_u[i,j] be a vector of length R such that dot(A_u[i,j], U[i,j,:]) = 1
            # A_u[i,j]_r = sum_k V[j*p+k, r] * W[k*n+i, r]
            
            # Construct A_u of shape (n*m, R)
            A_u = np.zeros((n * m, R))
            for i in range(n):
                for j in range(m):
                    # V indices: j*p to j*p+p-1
                    # W indices: k*n+i for k in 0..p-1
                    # We need sum_k V[j*p+k, r] * W[k*n+i, r]
                    # This is a sum over k of outer products? No, element-wise product summed over k.
                    # V_slice = V[j*p:(j+1)*p, :] (shape p x R)
                    # W_slice = W[0:p*n:n, i]? No, W is indexed by k*n+i.
                    # So W_indices = [k*n + i for k in range(p)]
                    W_idx = np.array([k * n + i for k in range(p)])
                    W_slice = W[W_idx, :] # shape p x R
                    V_slice = V[j * p:(j + 1) * p, :] # shape p x R
                    # A_u[i*m+j, r] = sum_k V_slice[k, r] * W_slice[k, r]
                    A_u[i * m + j, :] = np.sum(V_slice * W_slice, axis=0)
            
            # Solve U * A_u^T = 1 (vector of ones)
            # U is n*m x R. A_u is n*m x R.
            # We want U[i,j,:] such that dot(U[i,j,:], A_u[i,j,:]) = 1
            # This is underdetermined if R > 1. We can use pseudo-inverse.
            # U = 1 * (A_u^T (A_u A_u^T)^-1) ? No.
            # Minimize || U A_u^T - 1 ||^2.
            # Solution: U = 1 * A_u / (A_u A_u^T) if we treat row-wise?
            # Actually, U is a matrix. Let's solve row by row or use vectorized ops.
            # U[i,j,:] = A_u[i,j,:] / (A_u[i,j,:] @ A_u[i,j,:])
            norms_sq = np.sum(A_u * A_u, axis=1)
            norms_sq[norms_sq == 0] = 1e-9
            U = A_u / norms_sq[:, np.newaxis]
            
            # Solve for V
            # For each (j,k), sum_r V[j,k,r] * (U[:, j, :] * W[:, k, :])_r = 1
            # A_v[j*p+k, r] = sum_i U[i*m+j, r] * W[k*n+i, r]
            A_v = np.zeros((m * p, R))
            for j in range(m):
                for k in range(p):
                    # U indices: i*m+j for i in 0..n-1
                    U_idx = np.array([i * m + j for i in range(n)])
                    U_slice = U[U_idx, :] # shape n x R
                    # W indices: k*n+i for i in 0..n-1
                    W_idx = np.array([k * n + i for i in range(n)])
                    W_slice = W[W_idx, :] # shape n x R
                    A_v[j * p + k, :] = np.sum(U_slice * W_slice, axis=0)
            
            norms_sq_v = np.sum(A_v * A_v, axis=1)
            norms_sq_v[norms_sq_v == 0] = 1e-9
            V = A_v / norms_sq_v[:, np.newaxis]
            
            # Solve for W
            # For each (k,i), sum_r W[k,i,r] * (U[:, i, :] * V[:, k, :])_r = 1
            # Note: W index is k*n+i.
            # A_w[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            A_w = np.zeros((p * n, R))
            for k in range(p):
                for i in range(n):
                    # U indices: i*m+j for j in 0..m-1
                    U_idx = np.array([i * m + j for j in range(m)])
                    U_slice = U[U_idx, :] # shape m x R
                    # V indices: j*p+k for j in 0..m-1
                    V_idx = np.array([j * p + k for j in range(m)])
                    V_slice = V[V_idx, :] # shape m x R
                    A_w[k * n + i, :] = np.sum(U_slice * V_slice, axis=0)
            
            norms_sq_w = np.sum(A_w * A_w, axis=1)
            norms_sq_w[norms_sq_w == 0] = 1e-9
            W = A_w / norms_sq_w[:, np.newaxis]
            
            # Calculate error
            # Error = sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_error < 1e-6:
                break

        if best_error < 1e-4:
            # Round to half-integers
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            
            # Verify exactness
            is_exact = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_int[i * m + j, :], V_int[j * p + k, :] * W_int[k * n + i, :])
                        if abs(val - 1.0) > 1e-9:
                            is_exact = False
                            break
                    if not is_exact:
                        break
                if not is_exact:
                    break
            
            if is_exact:
                # Convert to lists of lists
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = W_int.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If not exact, try next rank

    # Fallback to schoolbook if no low-rank solution found
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
# EVOLVE-BLOCK-END
