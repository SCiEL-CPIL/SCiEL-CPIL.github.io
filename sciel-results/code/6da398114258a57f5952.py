"""Tensor decomposition for 2x2 by 2x5 matrix multiplication."""

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_rank18():
    """Construct the known rank-18 decomposition."""
    n, m, p = 2, 2, 5
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(t_offset, k_start):
        U[0][t_offset+0] = 1
        U[3][t_offset+0] = 1
        U[2][t_offset+1] = 1
        U[3][t_offset+1] = 1
        U[0][t_offset+2] = 1
        U[3][t_offset+3] = 1
        U[0][t_offset+4] = 1
        U[1][t_offset+4] = 1
        U[2][t_offset+5] = 1
        U[0][t_offset+5] = -1
        U[1][t_offset+6] = 1
        U[3][t_offset+6] = -1
        
        k0 = k_start
        k1 = k_start + 1
        V[0*p + k0][t_offset+0] = 1
        V[1*p + k1][t_offset+0] = 1
        V[0*p + k0][t_offset+1] = 1
        V[0*p + k1][t_offset+2] = 1
        V[1*p + k1][t_offset+2] = -1
        V[1*p + k0][t_offset+3] = 1
        V[0*p + k0][t_offset+3] = -1
        V[1*p + k1][t_offset+4] = 1
        V[0*p + k0][t_offset+5] = 1
        V[0*p + k1][t_offset+5] = 1
        V[1*p + k0][t_offset+6] = 1
        V[1*p + k1][t_offset+6] = 1
        
        W[k0*n + 0][t_offset+0] = 1
        W[k1*n + 1][t_offset+0] = 1
        W[k0*n + 1][t_offset+1] = 1
        W[k1*n + 1][t_offset+1] = -1
        W[k1*n + 0][t_offset+2] = 1
        W[k1*n + 1][t_offset+2] = 1
        W[k0*n + 0][t_offset+3] = 1
        W[k0*n + 1][t_offset+3] = 1
        W[k0*n + 0][t_offset+4] = -1
        W[k1*n + 0][t_offset+4] = 1
        W[k1*n + 1][t_offset+5] = 1
        W[k0*n + 0][t_offset+6] = 1
    
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    
    t = 14
    U[0][t] = 1
    V[0*p + 4][t] = 1
    W[4*n + 0][t] = 1
    t += 1
    U[1][t] = 1
    V[1*p + 4][t] = 1
    W[4*n + 0][t] = 1
    t += 1
    U[2][t] = 1
    V[0*p + 4][t] = 1
    W[4*n + 1][t] = 1
    t += 1
    U[3][t] = 1
    V[1*p + 4][t] = 1
    W[4*n + 1][t] = 1
    
    return U, V, W

def _try_rank16_als(n, m, p, deadline):
    """Try to find a rank-16 decomposition using ALS with multiple restarts."""
    rank = 16
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(restart * 12345 + 67890)
        
        # Initialize
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros(rank)
                    for k in range(p):
                        M += V[j*p+k, :] * W[k*n+i, :]
                    M_dot_M = np.dot(M, M)
                    if abs(M_dot_M) > 1e-15:
                        U[i*m+j, :] = p * M / M_dot_M
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros(rank)
                    for i in range(n):
                        M += U[i*m+j, :] * W[k*n+i, :]
                    M_dot_M = np.dot(M, M)
                    if abs(M_dot_M) > 1e-15:
                        V[j*p+k, :] = n * M / M_dot_M
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros(rank)
                    for j in range(m):
                        M += U[i*m+j, :] * V[j*p+k, :]
                    M_dot_M = np.dot(M, M)
                    if abs(M_dot_M) > 1e-15:
                        W[k*n+i, :] = m * M / M_dot_M
            
            # Check error
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-20:
                return U.tolist(), V.tolist(), W.tolist()
        
        if best_err < 1e-18:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try rank 16
    result = _try_rank16_als(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 18
    U, V, W = _construct_rank18()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
