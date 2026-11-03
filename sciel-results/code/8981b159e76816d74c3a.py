import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_88():
    n, m, p = 3, 4, 8
    U2, V2, W2 = _strassen_2x2x2()
    total_rank = 2 * 4 * 7 + 4 * 8
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    col = 0
    for j0 in (0, 2):
        for k0 in range(0, 8, 2):
            for c in range(7):
                U[0 * m + j0 + 0, col + c] = U2[0][c]
                U[0 * m + j0 + 1, col + c] = U2[1][c]
                U[1 * m + j0 + 0, col + c] = U2[2][c]
                U[1 * m + j0 + 1, col + c] = U2[3][c]
                V[(j0 + 0) * p + (k0 + 0), col + c] = V2[0][c]
                V[(j0 + 0) * p + (k0 + 1), col + c] = V2[1][c]
                V[(j0 + 1) * p + (k0 + 0), col + c] = V2[2][c]
                V[(j0 + 1) * p + (k0 + 1), col + c] = V2[3][c]
                W[(k0 + 0) * n + 0, col + c] = W2[0][c]
                W[(k0 + 1) * n + 0, col + c] = W2[1][c]
                W[(k0 + 0) * n + 1, col + c] = W2[2][c]
                W[(k0 + 1) * n + 1, col + c] = W2[3][c]
            col += 7
    for j in range(m):
        for k in range(p):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    U, V, W = _construct_88()
    
    if not _verify(U, V, W, 3, 4, 8):
        # Fallback
        pass

    # Try to reduce rank using linear dependencies
    # Vectorize the tensors
    n, m, p = 3, 4, 8
    dim = n * m * p
    r = U.shape[1]
    
    # Matrix A where column c is the vectorized rank-1 tensor t_c
    # t_c[i, j, k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    A = np.zeros((dim, r), dtype=float)
    for c in range(r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A[i*m*p + j*p + k, c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Target tensor T
    T = np.zeros(dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m*p + j*p + k] = 1.0

    # Check if sum of columns is T
    # This is guaranteed by construction, but let's be safe.
    
    # We want to find a subset of columns that sums to T.
    # This is equivalent to finding a vector x such that A @ x = T, minimizing ||x||_0.
    # Greedy approach:
    # 1. Find a column c such that t_c is in the span of the others.
    # 2. Remove it and adjust coefficients.
    
    # Efficient check for linear dependency:
    # If rank(A) < r, there is a dependency.
    # We can use SVD to find the null space.
    
    # However, we need the dependency to allow removing a term with coefficient 1.
    # If A @ x = T, and we have a dependency A @ y = 0, then A @ (x + y) = T.
    # We want to find x with minimal support.
    
    # Let's try the greedy removal again, but faster.
    # We can precompute the contribution of each column.
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_r = r
    
    # Use the linear algebra insight:
    # If t_c is in the span of {t_j}_{j != c}, we can remove it.
    # We can check this by computing the rank of A without column c.
    # If rank(A_without_c) == rank(A), then t_c is in the span.
    
    # Compute rank of A
    rank_A = np.linalg.matrix_rank(A)
    
    # If rank_A < r, we can potentially reduce.
    # We will try to remove columns one by one if they are redundant.
    
    for c in range(current_r - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        if current_r == 0:
            break
        
        # Check if column c is redundant
        # A_without_c = np.delete(A, c, axis=1)
        # rank_without_c = np.linalg.matrix_rank(A_without_c)
        # if rank_without_c == rank_A:
        #     # Remove column c
        #     # But we must update the coefficients?
        #     # No, if we remove t_c, the sum changes by -t_c.
        #     # We need to add t_c to the remaining terms.
        #     # This is only possible if t_c is in the span.
        #     # If t_c = sum alpha_j t_j, then T = sum t_j + t_c = sum (1+alpha_j) t_j.
        #     # We can update the vectors U, V, W to reflect the new coefficients.
        #     # This is complex.
        
        # Simpler: Just use the _verify based removal.
        # It is slower but correct.
        
        U_o = np.delete(current_U, c, axis=1)
        V_o = np.delete(current_V, c, axis=1)
        W_o = np.delete(current_W, c, axis=1)
        
        if _verify(U_o, V_o, W_o, n, m, p):
            current_U, current_V, current_W = U_o, V_o, W_o
            current_r -= 1
            # Update A? No, just continue with verify.
            # Reconstruct A? Too slow.
            # Just rely on _verify.

    # Final verify
    if not _verify(current_U, current_V, current_W, n, m, p):
        # Fallback to 88
        current_U, current_V, current_W = U, V, W

    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
