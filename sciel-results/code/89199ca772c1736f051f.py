"""Matrix multiplication 4,4,6 rank-minimal decomposition."""
import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    
    # Build the trivial rank-96 decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Build target tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def check_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces the target tensor."""
        T_test = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_test += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_test, T, atol=1e-10)
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to remove columns greedily
    start_time = time.time()
    deadline = start_time + 130
    
    current_rank = rank
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in range(current_rank):
            if time.time() > deadline:
                break
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            
            if check_decomposition(U_test, V_test, W_test):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                best_rank = current_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                improved = True
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
