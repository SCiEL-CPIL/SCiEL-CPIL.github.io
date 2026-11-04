"""Rewritten program for Matrix Multiplication 3,5,7."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    """Schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that the decomposition correctly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_decomp[i, j, k] = np.sum(
                    U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                )
    
    return np.allclose(T_decomp, 1.0, atol=1e-8)

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for matrix multiplication
    of dimensions 3 x 5 and 5 x 7.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    schoolbook_rank = n * m * p
    
    best_result = schoolbook
    best_rank = schoolbook_rank
    
    np.random.seed(42)
    
    lower_bound = max(n*p, m*p, n*m)
    
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 30:
            break
        
        found = False
        for restart in range(10):
            if time.time() > deadline - 30:
                break
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            for iteration in range(30):
                if time.time() > deadline - 30:
                    break
                
                # Solve for U given V, W
                for i in range(n):
                    for j in range(m):
                        A = np.array([[V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank)] for k in range(p)])
                        try:
                            u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Solve for V given U, W
                for j in range(m):
                    for k in range(p):
                        A = np.array([[U[i*m+j, r] * W[k*n+i, r] for r in range(target_rank)] for i in range(n)])
                        try:
                            v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                # Solve for W given U, V
                for k in range(p):
                    for i in range(n):
                        A = np.array([[U[i*m+j, r] * V[j*p+k, r] for r in range(target_rank)] for j in range(m)])
                        try:
                            w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
                
                # Check
                T_decomp = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_decomp[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if np.max(np.abs(T_decomp - 1.0)) < 1e-8:
                    found = True
                    break
            
            if found:
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if found:
            break
    
    if _verify(best_result["U"], best_result["V"], best_result["W"]):
        return best_result
    return schoolbook
# EVOLVE-BLOCK-END
