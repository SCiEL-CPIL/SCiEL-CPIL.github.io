import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    size_a = n * m  # 8
    size_b = m * p  # 24
    size_c = p * n  # 12
    
    # Target tensor T of shape (size_a, size_b, size_c)
    # T[a, b, c] = 1 if the indices are consistent with matrix multiplication
    # a = i*m + j, b = j*p + k, c = k*n + i
    
    T = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a rank-minimal decomposition using ALS
    # The trivial rank is n*p = 12 (one term per output element, with U having 1s for the 4 relevant inputs)
    # Let's verify rank 12 works with a simple construction first, then try to beat it.
    
    # Simple rank 12 construction:
    # For each k in 0..5 (p=6) and i in 0..1 (n=2), we have an output c = k*n + i.
    # We need sum_{j=0..3} A[i, j] B[j, k] for each (i, k).
    # This is a dot product of 4 elements.
    # We can do this with 4 terms per (i, k) pair? No, that would be 6*2*4 = 48.
    # Wait, the "trivial" decomposition for matrix multiplication (n x m) * (m x p) has rank n*m*p.
    # Here n=2, m=4, p=6, so rank = 48.
    # My previous "rank 12" idea was flawed. Let's re-evaluate.
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    # For a fixed (i, k), c is fixed. We sum over j.
    # T[i*m+j, j*p+k, k*n+i] = 1.
    # This looks like we need to sum A[i, j] * B[j, k].
    # This is exactly matrix multiplication.
    # The rank of the matrix multiplication tensor M_{n, m, p} is at least max(n*m, m*p, p*n) = max(8, 24, 12) = 24.
    # It is at most n*m*p = 48.
    # Known lower bounds for small matrices:
    # M_{2, 4, 6} rank is between 24 and 48.
    # Let's try to find a rank < 48 using ALS.
    
    def als_decompose(T_tensor, rank, iters=200, tol=1e-12, seed=0, max_time=100):
        import time
        start_time = time.time()
        np.random.seed(seed)
        
        size_a, size_b, size_c = T_tensor.shape
        
        U = np.random.randn(size_a, rank)
        V = np.random.randn(size_b, rank)
        W = np.random.randn(size_c, rank)
        
        prev_err = np.inf
        
        for it in range(iters):
            if time.time() - start_time > max_time:
                break
                
            # Solve for U
            # T[a, b, c] = sum_l U[a, l] V[b, l] W[c, l]
            # Fix V, W. T_mat[a, b*size_c+c] = sum_l U[a, l] * (V[:,l] o W[:,l])[b*size_c+c]
            # Let Y_l = (V[:,l] o W[:,l]).flatten()
            # T_mat = U @ Y.T  (where T_mat is size_a x size_b*size_c)
            # U = (Y Y^T)^-1 Y T_mat^T
            
            Y = np.zeros((rank, size_b * size_c))
            for l in range(rank):
                Y[l, :] = np.outer(V[:, l], W[:, l]).flatten()
            
            T_mat = T_tensor.reshape(size_a, size_b * size_c)
            
            try:
                U = np.linalg.solve(Y @ Y.T, Y @ T_mat.T).T
            except:
                U = np.linalg.pinv(Y @ Y.T) @ (Y @ T_mat.T).T
                
            # Solve for V
            # T_mat2[a*size_c+c, b] = sum_l U[a, l] W[c, l] V[b, l]
            # Let Z_l = (U[:,l] o W[:,l]).flatten() (size_a*size_c)
            # T_mat2 = Z @ V.T
            # V = (Z^T Z)^-1 Z^T T_mat2
            
            Z = np.zeros((rank, size_a * size_c))
            for l in range(rank):
                Z[l, :] = np.outer(U[:, l], W[:, l]).flatten()
            
            T_mat2 = T_tensor.transpose(0, 2, 1).reshape(size_a * size_c, size_b)
            
            try:
                V = np.linalg.solve(Z.T @ Z, Z.T @ T_mat2)
            except:
                V = np.linalg.pinv(Z.T @ Z) @ (Z.T @ T_mat2)
                
            # Solve for W
            # T_mat3[a*size_b+b, c] = sum_l U[a, l] V[b, l] W[c, l]
            # Let X_l = (U[:,l] o V[:,l]).flatten() (size_a*size_b)
            # T_mat3 = X @ W.T
            # W = (X^T X)^-1 X^T T_mat3
            
            X = np.zeros((rank, size_a * size_b))
            for l in range(rank):
                X[l, :] = np.outer(U[:, l], V[:, l]).flatten()
            
            T_mat3 = T_tensor.reshape(size_a * size_b, size_c)
            
            try:
                W = np.linalg.solve(X.T @ X, X.T @ T_mat3)
            except:
                W = np.linalg.pinv(X.T @ X) @ (X.T @ T_mat3)
                
            # Calculate error
            T_approx = np.zeros_like(T_tensor)
            for l in range(rank):
                T_approx += np.einsum('a,b,c->abc', U[:, l], V[:, l], W[:, l])
            err = np.linalg.norm(T_tensor - T_approx)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        return U, V, W, err

    import time
    deadline = time.time() + 120
    best_result = None
    
    # Try ranks from 24 upwards
    for rank in range(24, 49):
        if time.time() > deadline - 10:
            break
            
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            try:
                U, V, W, err = als_decompose(T, rank, iters=50, tol=1e-10, seed=seed, max_time=10)
                if err < 1e-8:
                    # Try to round to half-integers
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    
                    T_approx = np.zeros_like(T)
                    for l in range(rank):
                        T_approx += np.einsum('a,b,c->abc', U_round[:, l], V_round[:, l], W_round[:, l])
                    err_round = np.linalg.norm(T - T_approx)
                    
                    if err_round < 1e-8:
                        best_result = {'U': U_round.tolist(), 'V': V_round.tolist(), 'W': W_round.tolist()}
                        found = True
                        break
            except Exception:
                pass
        if found:
            break

    if best_result:
        return best_result

    # Fallback to rank 48 (trivial decomposition)
    rank = 48
    U = [[0] * rank for _ in range(size_a)]
    V = [[0] * rank for _ in range(size_b)]
    W = [[0] * rank for _ in range(size_c)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
                
    return {'U': U, 'V': V, 'W': W}
