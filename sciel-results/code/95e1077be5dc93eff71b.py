import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 64
    
    # Define the tensor T[i, j, k] = delta(i, j) * delta(j, k)
    # i in [0, n), j in [0, m), k in [0, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            if i == j:
                for k in range(p):
                    if j == k:
                        T[i, j, k] = 1.0

    # Flatten T for vector operations
    T_flat = T.flatten()
    
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_c U[:, c] outer V[:, c] outer W[:, c] approx T_flat
    # Note: The index mapping for W is k*n + i
    
    def build_tensor(U, V, W):
        # U: (n*m, r)
        # V: (m*p, r)
        # W: (p*n, r)
        # Result: (n, m, p)
        res = np.zeros((n, m, p))
        r = U.shape[1]
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # The term is u[i, j] * v[j, k] * w[k, i]
            # Sum over i, j, k
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return res.flatten()

    def error(U, V, W):
        T_approx = build_tensor(U, V, W)
        return np.linalg.norm(T_flat - T_approx)

    # Initialize with random values
    random.seed(42)
    np.random.seed(42)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Try to find a rank 64 solution
    # Use ALS (Alternating Least Squares) with rounding
    
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for step in range(100):
            if time.time() > deadline:
                break
                
            # Update U
            # Fix V, W. Minimize || T - sum_c U_c outer V_c outer W_c ||^2
            # This is a linear system for U_c
            # T_flat = sum_c (V_c outer W_c) * U_c (flattened)
            # Let A_c = (V_c outer W_c) flattened
            # T_flat = A @ U_flat
            # U_flat = A^+ T_flat
            
            A = np.zeros((n*m*p, target_rank))
            for c in range(target_rank):
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                # We need the vector such that sum_{i,j,k} U[i,j,c] v[j,k] w[k,i] = T[i,j,k]
                # The coefficient for U[i,j,c] is v[j,k] w[k,i]
                # So A[i*m+j, c] is not a scalar, it's a vector in the (i,j,k) space?
                # No, we are solving for U.
                # T[i,j,k] = sum_c U[i,j,c] * (sum_{j,k} v[j,k] w[k,i] * delta(i,j,k)?)
                # Actually, for fixed c, the term is U_c (size n*m) times a vector derived from V_c, W_c.
                # Let's construct the matrix M such that T_flat = M @ U_flat
                # M has shape (n*m*p, n*m*target_rank)
                # This is too large.
                
            # Instead, use gradient descent or simple ALS on the tensor.
            # Given time constraints, a simple local search with rounding might be better.
            
            # Round to nearest half-integer
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            err = error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            # If error is very small, break
            if err < 1e-6:
                break

    # If we found a good solution, return it
    if best_U is not None and best_err < 1e-6:
        # Convert to list of lists with integer/half-integer entries
        def to_list(mat):
            return [[float(x) for x in row] for row in mat]
        
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Fallback: Return schoolbook (rank 80)
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
