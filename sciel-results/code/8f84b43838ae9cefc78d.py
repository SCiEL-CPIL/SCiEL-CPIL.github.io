import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def find_decomposition_444(rank, max_iters=500, tol=1e-10):
        """Find a rank-r decomposition for 4x4x4 matrix multiplication."""
        n = m = p = 4
        
        def verify(U, V, W, tol=1e-10):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                        if abs(val - 1) > tol:
                            return False
            return True
        
        for seed in range(20):
            if time.time() > deadline - 30:
                break
            np.random.seed(seed)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            prev_err = float('inf')
            for it in range(max_iters):
                if time.time() > deadline - 30:
                    break
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        M = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        M = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        V[j*p+k, :], _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        M = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        W[k*n+i, :], _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                
                # Compute error
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                            err += (val - 1)**2
                
                if err < tol:
                    return U, V, W
                
                if abs(prev_err - err) < 1e-15:
                    break
                prev_err = err
        
        return None
    
    # Try to find rank-19 decomposition for 4x4x4
    result_444 = find_decomposition_444(19)
    
    if result_444 is not None:
        U444, V444, W444 = result_444
        # Extend to 4x4x8: use two copies
        n, m, p = 4, 4, 8
        rank = 38
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # First block (k=0..3)
        U[:, 0:19] = U444
        for j in range(m):
            for k in range(4):
                V[j*p+k, 0:19] = V444[j*4+k]
        for k in range(4):
            for i in range(n):
                W[k*n+i, 0:19] = W444[k*4+i]
        
        # Second block (k=4..7)
        U[:, 19:38] = U444
        for j in range(m):
            for k in range(4):
                V[j*p+(k+4), 19:38] = V444[j*4+k]
        for k in range(4):
            for i in range(n):
                W[(k+4)*n+i, 19:38] = W444[k*4+i]
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1) > 1e-8:
                        # Fallback
                        break
            else:
                continue
            break
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    n, m, p = 4, 4, 8
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
