"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, seed=42, max_iters=5000):
        """Try to find a rank-r decomposition using alternating least squares."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-10:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded, tol=1e-10):
                    return U_rounded, V_rounded, W_rounded, True
                elif verify(U, V, W, tol=1e-10):
                    return U, V, W, True
        
        # Check final error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-10:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if verify(U_rounded, V_rounded, W_rounded, tol=1e-10):
                return U_rounded, V_rounded, W_rounded, True
            elif verify(U, V, W, tol=1e-10):
                return U, V, W, True
        
        return None, None, None, False
    
    # Try rank 16 with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        U, V, W, success = try_rank(16, seed=seed, max_iters=2000)
        if success:
            U_list = [[float(U[i][j]) for j in range(16)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(16)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(16)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 17
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        U, V, W, success = try_rank(17, seed=seed, max_iters=2000)
        if success:
            U_list = [[float(U[i][j]) for j in range(17)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(17)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(17)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 18
    for seed in range(5):
        if time.time() > deadline - 30:
            break
        U, V, W, success = try_rank(18, seed=seed, max_iters=2000)
        if success:
            U_list = [[float(U[i][j]) for j in range(18)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(18)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(18)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
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
# EVOLVE-BLOCK-END
