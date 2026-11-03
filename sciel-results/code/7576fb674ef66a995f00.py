import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = 2, 3, 3
    
    # Rank 17 solution (known valid)
    rank17 = 17
    U17 = np.zeros((n * m, rank17))
    V17 = np.zeros((m * p, rank17))
    W17 = np.zeros((p * n, rank17))
    
    # Part 1: Strassen for 2x2x2 sub-block
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Part 2: Cross terms A[:,2] * B[2,:]
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    # Part 3: Cross terms A[:,:2] * B[:,2]
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1
    
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    # Verify rank-17 solution
    if not _verify(U17, V17, W17, n, m, p, rank17):
        # Fallback to schoolbook
        U_fb = np.zeros((n * m, n * m * p))
        V_fb = np.zeros((m * p, n * m * p))
        W_fb = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to find rank-16 solution using numerical optimization
    best_rank = rank17
    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    
    # Lascu-Parter analysis: compute flattening ranks
    # Build the multiplication tensor T[i*m+j, j'*p+k, k'*n+i'] where T=1 if j=j' and i=i' and k=k'
    # Actually, the tensor is: T[a, b, c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 if the indices are consistent: j from a, j' from b must match, i from a and i' from c must match, k from b and k' from c must match
    
    # Let's build the tensor explicitly
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Lascu-Parter: matricize along each mode
    T_uvw = T.reshape(n*m, m*p*p*n)  # U | VW
    T_vuw = T.reshape(m*p, n*m*p*n)  # V | UW
    T_wuv = T.reshape(p*n, n*m*m*p)  # W | UV
    
    rank_uvw = np.linalg.matrix_rank(T_uvw)
    rank_vuw = np.linalg.matrix_rank(T_vuw)
    rank_wuv = np.linalg.matrix_rank(T_wuv)
    
    lp_lower_bound = max(rank_uvw, rank_vuw, rank_wuv)
    
    # If LP bound > 16, rank 16 is impossible
    if lp_lower_bound > 16:
        # Return rank 17
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try numerical search for rank 16
    rank16 = 16
    np.random.seed(42)
    
    # Initialize with random values, scaled appropriately
    U16 = np.random.randn(n*m, rank16) * 0.5
    V16 = np.random.randn(m*p, rank16) * 0.5
    W16 = np.random.randn(p*n, rank16) * 0.5
    
    # Simple gradient descent
    lr = 0.01
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute current decomposition error
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Target: C[i,k] should have C[i,k] = delta_{j,j'} for the appropriate terms
        
        # Actually, let's compute the residual differently.
        # The tensor equation: for all (a,b,c) where a=i*m+j, b=j'*p+k, c=k'*n+i'
        # sum_r U[a,r]*V[b,r]*W[c,r] = 1 if j=j' and i=i' and k=k', else 0
        
        # Let's compute the residual
        residual = np.zeros((n*m, m*p, p*n))
        for r in range(rank16):
            residual += np.einsum('a,b,c->abc', U16[:,r], V16[:,r], W16[:,r])
        
        # Target tensor
        target = T
        
        # Loss
        loss = np.sum((residual - target)**2)
        
        if loss < 1e-8:
            best_rank = rank16
            best_U, best_V, best_W = U16.copy(), V16.copy(), W16.copy()
            break
        
        # Compute gradients
        # dL/dU[a,r] = 2 * sum_{b,c} (residual[a,b,c] - target[a,b,c]) * V[b,r] * W[c,r]
        # dL/dV[b,r] = 2 * sum_{a,c} (residual[a,b,c] - target[a,b,c]) * U[a,r] * W[c,r]
        # dL/dW[c,r] = 2 * sum_{a,b} (residual[a,b,c] - target[a,b,c]) * U[a,r] * V[b,r]
        
        diff = residual - target
        
        dU = 2.0 * np.einsum('abc,br,cr->ar', diff, V16, W16)
        dV = 2.0 * np.einsum('abc,ar,cr->br', diff, U16, W16)
        dW = 2.0 * np.einsum('abc,ar,br->cr', diff, U16, V16)
        
        # Update
        U16 -= lr * dU
        V16 -= lr * dV
        W16 -= lr * dW
    
    # Check if we found a valid rank-16 solution
    if best_rank == rank16 and _verify(best_U, best_V, best_W, n, m, p, rank16):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Return rank 17
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
