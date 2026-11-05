import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Precompute target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Flatten T for solving
    # We will try to find rank 46 or 47.
    
    def try_find_decomposition(r, max_restarts=5, max_iters=10):
        # Initialize
        U = np.random.randint(-1, 2, size=(n*m, r))
        V = np.random.randint(-1, 2, size=(m*p, r))
        W = np.random.randint(-1, 2, size=(p*n, r))
        
        for _ in range(max_restarts):
            for _ in range(max_iters):
                # Solve for W
                # For each K, solve A_K * W[K,:] = b_K
                for K in range(p*n):
                    b = T[:, :, K].reshape(-1)
                    # Build A_K
                    # A_K[I*m*p + J, c] = U[I, c] * V[J, c]
                    # Only need non-zero parts of b? No, lstsq needs full A.
                    # But b is sparse.
                    # We can just build full A.
                    A = np.zeros((n*m * m*p, r))
                    idx = 0
                    for I in range(n*m):
                        for J in range(m*p):
                            A[idx, :] = U[I, :] * V[J, :]
                            idx += 1
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[K, :] = sol
                
                # Solve for U
                for I in range(n*m):
                    b = T[I, :, :].reshape(-1)
                    A = np.zeros((m*p * p*n, r))
                    idx = 0
                    for J in range(m*p):
                        for K in range(p*n):
                            A[idx, :] = V[J, :] * W[K, :]
                            idx += 1
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[I, :] = sol
                    
                # Solve for V
                for J in range(m*p):
                    b = T[:, J, :].reshape(-1)
                    A = np.zeros((n*m * p*n, r))
                    idx = 0
                    for I in range(n*m):
                        for K in range(p*n):
                            A[idx, :] = U[I, :] * W[K, :]
                            idx += 1
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[J, :] = sol
                
                # Check
                T_hat = np.einsum('ic,jc,kc->ijk', U, V, W)
                if np.max(np.abs(T_hat - T)) < 1e-6:
                    # Check half-integer
                    if (np.max(np.abs(U*2 - np.round(U*2))) < 1e-6 and
                        np.max(np.abs(V*2 - np.round(V*2))) < 1e-6 and
                        np.max(np.abs(W*2 - np.round(W*2))) < 1e-6):
                        return U, V, W
                        
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Try rank 46
    if time.time() < deadline:
        res = try_find_decomposition(46, max_restarts=3, max_iters=5)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try rank 47
    if time.time() < deadline:
        res = try_find_decomposition(47, max_restarts=3, max_iters=5)
        if res:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
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
